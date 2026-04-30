#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# =============================================================================
# Full end-to-end pipeline for:
#   "A Fuzzy Convolutional Neural Network for Multi-Focus Image Fusion"
#   Bhalla et al., JVCIR 2022
#
# Stages:
#   0. Dependency check / install
#   1. Ingest real datasets (Dataset 1: Lytro, Dataset 2: MFIF)
#   2. Download COCO source images (ImageNet proxy, ~5 GB)
#   3. Generate 5-level Gaussian blur (σ=0.5, sequential)
#   4. Extract 16×16 non-overlapping patches
#   5. Build training dataset NPZ (5,000 pairs, balanced)
#   6. Train Siamese FCNN (SGD, lr=0.002, 30 epochs)
#   7. Verify full pipeline output
#   8. Run Fusion on all test pairs (stride=2, paper-exact SM=253×253)
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                        # full run
#   ./run_pipeline.sh --skip-download        # skip COCO download (already done)
#   ./run_pipeline.sh --epochs 50            # custom epoch count
#   ./run_pipeline.sh --workers 8            # more CPU workers
#   ./run_pipeline.sh --stage 8              # run fusion only (model already trained)
#   ./run_pipeline.sh --fast-stride          # use stride=8 for fast inference (~30s/pair)
#
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner()  { echo -e "\n${CYAN}${BOLD}╔══ $1 ══╗${NC}"; }
ok()      { echo -e "  ${GREEN}[OK]${NC}  $1"; }
info()    { echo -e "  ${CYAN}[INFO]${NC} $1"; }
warn()    { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "  ${RED}[FAIL]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}──── Stage $1: $2 ────${NC}"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS=4
EPOCHS=30
BATCH_SIZE=64
N_IMAGES=1000          # source images (use 1000 for disk safety; 5000 for full paper)
START_STAGE=0
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
FAST_STRIDE=false      # false=stride 2 (paper-exact, slow), true=stride 8 (fast)
ZIP_PATH=""            # path to Multi-focus-Image-Fusion-Dataset-master.zip
OUT_DIR="outputs"      # root output directory for fusion results

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --skip-train)    SKIP_TRAIN=true;    shift ;;
    --fast-stride)   FAST_STRIDE=true;   shift ;;
    --workers)       WORKERS="$2";       shift 2 ;;
    --epochs)        EPOCHS="$2";        shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
    --n-images)      N_IMAGES="$2";      shift 2 ;;
    --stage)         START_STAGE="$2";   shift 2 ;;
    --zip)           ZIP_PATH="$2";      shift 2 ;;
    --out-dir)       OUT_DIR="$2";       shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --skip-download    Skip COCO image download (already done)"
      echo "  --skip-train       Skip model training"
      echo "  --fast-stride      Use stride=8 for fusion (~30s/pair vs ~8min)"
      echo "  --workers N        Number of CPU workers  (default: 4)"
      echo "  --epochs N         Training epochs        (default: 30)"
      echo "  --batch-size N     Training batch size    (default: 64)"
      echo "  --n-images N       Source images to use   (default: 1000)"
      echo "  --stage N          Resume from stage N    (default: 0)"
      echo "  --zip PATH         Path to dataset zip    (auto-detected if not set)"
      echo "  --out-dir PATH     Fusion output directory (default: ./outputs)"
      exit 0 ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

# ── Stride selection ──────────────────────────────────────────────────────────
if [[ "$FAST_STRIDE" == "true" ]]; then
  STRIDE=8
  STRIDE_NOTE="stride=8 (fast, SM=64×64)"
else
  STRIDE=2
  STRIDE_NOTE="stride=2 (paper-exact, SM=253×253)"
fi

# ── Header ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}"
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  FCNN-MFIF  Full Pipeline                           │"
echo "  │  Bhalla et al., JVCIR 2022                          │"
echo "  └─────────────────────────────────────────────────────┘"
echo -e "${NC}"
info "Working directory : $SCRIPT_DIR"
info "Workers           : $WORKERS"
info "Source images     : $N_IMAGES"
info "Training epochs   : $EPOCHS"
info "Batch size        : $BATCH_SIZE"
info "Start stage       : $START_STAGE"
info "Fusion stride     : $STRIDE_NOTE"
info "Output directory  : $OUT_DIR"
echo ""

cd "$SCRIPT_DIR"

# ── Helper: stage gate ────────────────────────────────────────────────────────
should_run() { [[ "$1" -ge "$START_STAGE" ]]; }

# =============================================================================
# STAGE 0: Dependency Check
# =============================================================================
if should_run 0; then
  step 0 "Dependency Check"

  # Python
  PYTHON=$(command -v python3 || command -v python || true)
  [[ -z "$PYTHON" ]] && fail "python3 not found. Install Python 3.9+ and retry."
  PY_VER=$($PYTHON --version 2>&1)
  ok "Python: $PY_VER"

  # pip packages
  info "Checking Python packages..."
  MISSING=""
  for pkg in numpy Pillow tqdm requests torch scipy; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
      MISSING="$MISSING $pkg"
    fi
  done

  if [[ -n "$MISSING" ]]; then
    info "Installing missing packages:$MISSING"
    if $PYTHON -c "import torch" 2>/dev/null; then
      pip install $MISSING -q --break-system-packages 2>/dev/null || \
      pip install $MISSING -q
    else
      # Install PyTorch CPU first
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
          -q --break-system-packages 2>/dev/null || \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
      [[ -n "$(echo $MISSING | tr ' ' '\n' | grep -v torch | tr '\n' ' ')" ]] && \
        pip install $(echo $MISSING | tr ' ' '\n' | grep -v torch) \
            -q --break-system-packages 2>/dev/null || true
    fi
  fi

  # Verify key packages
  TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
  [[ "$TORCH_VER" == "MISSING" ]] && fail "PyTorch install failed."
  ok "PyTorch : $TORCH_VER"

  SCIPY_VER=$($PYTHON -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "MISSING")
  [[ "$SCIPY_VER" == "MISSING" ]] && fail "scipy install failed (needed for guided filter)."
  ok "scipy   : $SCIPY_VER"

  # matplotlib (optional — for plots)
  $PYTHON -c "import matplotlib" 2>/dev/null && ok "matplotlib: available" \
    || warn "matplotlib not found — pipeline visualizations will be skipped."

  # Disk space check (need at least 5 GB free)
  AVAIL_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2{gsub("G",""); print $4}')
  if [[ "$AVAIL_GB" -lt 3 ]]; then
    warn "Only ${AVAIL_GB} GB free. Pipeline needs ~3 GB for $N_IMAGES images."
    warn "Consider reducing --n-images or freeing space."
  else
    ok "Disk: ${AVAIL_GB} GB available"
  fi
fi

# =============================================================================
# STAGE 1: Ingest Real Datasets
# =============================================================================
if should_run 1; then
  step 1 "Ingest Real Multi-Focus Datasets"

  # Auto-detect zip
  if [[ -z "$ZIP_PATH" ]]; then
    ZIP_PATH=$(find "$SCRIPT_DIR/datasets" -maxdepth 1 -name "*.zip" 2>/dev/null | head -1 || true)
  fi

  DS1="$SCRIPT_DIR/datasets/real/dataset1_lytro"
  DS2="$SCRIPT_DIR/datasets/real/dataset2_mfif"

  # Dataset 1 (Lytro)
  if [[ -d "$DS1" ]] && [[ $(ls "$DS1"/pair_*/A.png 2>/dev/null | wc -l) -ge 20 ]]; then
    ok "Dataset 1 (Lytro): already ingested ($(ls -d $DS1/pair_*/ 2>/dev/null | wc -l) pairs)"
  elif [[ -n "$ZIP_PATH" && -f "$ZIP_PATH" ]]; then
    info "Ingesting Dataset 1 from $ZIP_PATH ..."
    $PYTHON scripts/ingest_lytro_zip.py --zip "$ZIP_PATH" \
      || fail "Dataset 1 ingestion failed."
    ok "Dataset 1 ingested."
  else
    warn "No zip found for Dataset 1. Skipping real dataset ingestion."
    warn "Run:  python3 scripts/ingest_lytro_zip.py --zip <path-to-zip>"
  fi

  # Dataset 2 (MFIF cross-level pairs)
  if [[ -d "$DS2" ]] && [[ $(ls "$DS2"/pair_*/A.png 2>/dev/null | wc -l) -ge 21 ]]; then
    ok "Dataset 2 (MFIF): already ingested ($(ls -d $DS2/pair_*/ 2>/dev/null | wc -l) pairs)"
  elif [[ -n "$ZIP_PATH" && -f "$ZIP_PATH" ]]; then
    info "Ingesting Dataset 2 from $ZIP_PATH ..."
    $PYTHON scripts/ingest_dataset2_from_zip.py --zip "$ZIP_PATH" \
      || fail "Dataset 2 ingestion failed."
    ok "Dataset 2 ingested."
  else
    warn "Skipping Dataset 2 ingestion."
  fi
fi

# =============================================================================
# STAGE 2: Download COCO Source Images
# =============================================================================
if should_run 2; then
  step 2 "Download Source Images (COCO/ImageNet proxy, n=$N_IMAGES)"

  RAW_DIR="$SCRIPT_DIR/datasets/synthetic/raw_imagenet"
  EXISTING=0
  [[ -d "$RAW_DIR" ]] && EXISTING=$(ls "$RAW_DIR"/*.png 2>/dev/null | wc -l || echo 0)

  if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
    info "Skipping download (--skip-download set). Found $EXISTING images."
  elif [[ "$EXISTING" -ge "$N_IMAGES" ]]; then
    ok "Already have $EXISTING source images — skipping download."
  else
    info "Downloading $N_IMAGES images (have $EXISTING so far) ..."
    info "This may take 1–3 hours depending on connection speed."
    $PYTHON scripts/02_download_imagenet_proxy.py \
      --n-images "$N_IMAGES" \
      || fail "Source image download failed."
    ok "Source images ready: $(ls $RAW_DIR/*.png 2>/dev/null | wc -l) images"
  fi
fi

# =============================================================================
# STAGE 3: Generate Blur Levels
# =============================================================================
if should_run 3; then
  step 3 "Generate 5-Level Gaussian Blur (σ=0.5, sequential)"

  BLUR_DIR="$SCRIPT_DIR/datasets/synthetic/blurred"
  BL1="$BLUR_DIR/blur_level_1"
  RAW_DIR="$SCRIPT_DIR/datasets/synthetic/raw_imagenet"
  N_SRC=$(ls "$RAW_DIR"/*.png 2>/dev/null | wc -l || echo 0)

  if [[ -d "$BL1" ]] && [[ $(ls "$BL1"/*.png 2>/dev/null | wc -l) -ge "$N_SRC" && "$N_SRC" -gt 0 ]]; then
    ok "Blur levels already generated ($(ls $BL1/*.png | wc -l) images per level)"
  else
    info "Generating blur levels for $N_SRC source images ..."
    info "Estimated time: ~$(( N_SRC / 25 / 60 + 1 )) minutes"
    $PYTHON scripts/03_generate_blur_levels.py \
      --workers "$WORKERS" \
      || fail "Blur generation failed."
    ok "Blur levels generated."
  fi
fi

# =============================================================================
# STAGE 4: Extract 16×16 Patches
# =============================================================================
if should_run 4; then
  step 4 "Extract 16×16 Non-Overlapping Patches"

  PATCHES_DIR="$SCRIPT_DIR/datasets/synthetic/patches/focused"
  RAW_DIR="$SCRIPT_DIR/datasets/synthetic/raw_imagenet"
  N_SRC=$(ls "$RAW_DIR"/*.png 2>/dev/null | wc -l || echo 0)

  if [[ -d "$PATCHES_DIR" ]] && [[ $(ls "$PATCHES_DIR"/*.npy 2>/dev/null | wc -l) -ge "$N_SRC" && "$N_SRC" -gt 0 ]]; then
    ok "Patches already extracted ($(ls $PATCHES_DIR/*.npy | wc -l) files)"
  else
    info "Extracting patches from $N_SRC images ..."
    $PYTHON scripts/04_extract_patches.py \
      --workers "$WORKERS" \
      || fail "Patch extraction failed."
    ok "Patches extracted."
  fi
fi

# =============================================================================
# STAGE 5: Build Training Dataset NPZ
# =============================================================================
if should_run 5; then
  step 5 "Build Training Dataset (5,000 pairs, balanced)"

  NPZ="$SCRIPT_DIR/datasets/synthetic/training_dataset_norm.npz"

  if [[ -f "$NPZ" ]]; then
    N_SAMPLES=$($PYTHON -c "
import numpy as np
d = np.load('$NPZ')
print(len(d['labels']))
" 2>/dev/null || echo 0)
    ok "Training NPZ already exists ($N_SAMPLES samples)"
  else
    info "Building training dataset (2500 positive + 2500 negative) ..."
    $PYTHON scripts/05_build_training_dataset.py \
      --n-positive 2500 \
      --n-negative 2500 \
      --augment \
      || fail "Training dataset build failed."
    ok "Training dataset built."
  fi
fi

# =============================================================================
# STAGE 6: Train Siamese FCNN
# =============================================================================
if should_run 6 && [[ "$SKIP_TRAIN" == "false" ]]; then
  step 6 "Train Siamese FCNN (SGD lr=0.002, epochs=$EPOCHS)"

  BEST_CKPT="$SCRIPT_DIR/checkpoints/fcnn_best.pt"

  if [[ -f "$BEST_CKPT" ]]; then
    info "Checkpoint found: $BEST_CKPT"
    read -rp "  Resume training? [y/N]: " RESUME
    if [[ "$RESUME" =~ ^[Yy]$ ]]; then
      info "Resuming from checkpoint..."
      $PYTHON train.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --resume "$BEST_CKPT" \
        || fail "Training failed."
    else
      info "Skipping training (existing checkpoint kept)."
    fi
  else
    info "Starting training from scratch..."
    info "Estimated time: ~$(( EPOCHS * 2 )) minutes on CPU"
    $PYTHON train.py \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      || fail "Training failed."
    ok "Training complete."
  fi
elif [[ "$SKIP_TRAIN" == "true" ]]; then
  info "Skipping training (--skip-train set)."
fi

# =============================================================================
# STAGE 7: Verify Pipeline
# =============================================================================
if should_run 7; then
  step 7 "Verify Full Pipeline"

  info "Running pipeline verification..."
  $PYTHON verify_pipeline.py --min-source-count 100 2>&1 | \
    grep -E "^\[|^OVERALL" | \
    sed "s/\[PASS\]/$(echo -e "${GREEN}[PASS]${NC}")/g" | \
    sed "s/\[FAIL\]/$(echo -e "${RED}[FAIL]${NC}")/g" | \
    sed "s/\[SKIP\]/$(echo -e "${YELLOW}[SKIP]${NC}")/g"

  VERIFY_EXIT="${PIPESTATUS[0]}"
  if [[ "$VERIFY_EXIT" -eq 0 ]]; then
    ok "All verification checks passed."
  else
    warn "Some verification checks failed — see above."
  fi
fi

# =============================================================================
# STAGE 8: Run Fusion on All Test Pairs
# =============================================================================
if should_run 8; then
  step 8 "Run Fusion on All Test Pairs ($STRIDE_NOTE)"

  BEST_CKPT="$SCRIPT_DIR/checkpoints/fcnn_best.pt"
  [[ ! -f "$BEST_CKPT" ]] && fail "No trained checkpoint found at $BEST_CKPT. Run stage 6 first."

  DS1="$SCRIPT_DIR/datasets/real/dataset1_lytro"
  DS2="$SCRIPT_DIR/datasets/real/dataset2_mfif"
  N_PAIRS=0
  for ds in "$DS1" "$DS2"; do
    [[ -d "$ds" ]] && N_PAIRS=$(( N_PAIRS + $(ls -d "$ds"/pair_*/ 2>/dev/null | wc -l) ))
  done

  if [[ "$N_PAIRS" -eq 0 ]]; then
    warn "No real dataset pairs found. Run stage 1 first to ingest datasets."
  else
    info "Found $N_PAIRS test pairs across Dataset 1 (Lytro) and Dataset 2 (MFIF)"
    if [[ "$STRIDE" -eq 2 ]]; then
      info "Using stride=2 (paper-exact). Est. time: ~$(( N_PAIRS * 8 )) min on CPU"
      warn "Tip: use --fast-stride to reduce this to ~$(( N_PAIRS / 2 )) min (stride=8)"
    else
      info "Using stride=8 (fast mode). Est. time: ~$(( N_PAIRS / 2 )) min on CPU"
    fi

    $PYTHON inference/fusion.py \
      --model "$BEST_CKPT" \
      --all-pairs \
      --out-dir "$OUT_DIR" \
      --stride "$STRIDE" \
      --batch-size 512 \
      || fail "Fusion inference failed."

    N_FUSED=$(find "$OUT_DIR" -name "*_fused.png" 2>/dev/null | wc -l)
    ok "Fusion complete — $N_FUSED fused images saved to $OUT_DIR/"
  fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗"
echo    "║         PIPELINE COMPLETE                    ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""

NPZ_NORM="$SCRIPT_DIR/datasets/synthetic/training_dataset_norm.npz"
BEST_PT="$SCRIPT_DIR/checkpoints/fcnn_best.pt"
REPORT="$SCRIPT_DIR/pipeline_verification_report.txt"
FUSED_COUNT=$(find "$OUT_DIR" -name "*_fused.png" 2>/dev/null | wc -l)

echo -e "${BOLD}Key outputs:${NC}"
[[ -f "$NPZ_NORM"      ]] && ok "Training data  : $NPZ_NORM"      || warn "Training NPZ   : not found"
[[ -f "$BEST_PT"       ]] && ok "Best model     : $BEST_PT"       || warn "Best model     : not trained yet"
[[ -f "$REPORT"        ]] && ok "Verify report  : $REPORT"
[[ "$FUSED_COUNT" -gt 0 ]] && ok "Fused images   : $FUSED_COUNT images in $OUT_DIR/" \
                           || warn "Fused images   : none yet (run stage 8)"

echo ""
echo -e "${BOLD}Pipeline equations implemented:${NC}"
echo "  Fuzzification   S-membership Eq.(3)"
echo "  Score Map       Siamese CNN, stride=$STRIDE → SM"
echo "  Focus Map       Overlap averaging Eq.(8)"
echo "  Binary Map      BM = (FM > 0.4) Eq.(9-10)"
echo "  Small regions   bwareaopen(BM, 0.01·H·W) Eq.(11-12)"
echo "  Guided filter   r=7, ε=0.2, guidance=mean(A',B')"
echo "  Fusion          F = FD·A' + (1−FD)·B'  Eq.(13)"

echo ""
echo -e "${BOLD}Useful commands:${NC}"
echo "  Fusion only:    ./run_pipeline.sh --stage 8 --skip-train"
echo "  Fast mode:      ./run_pipeline.sh --stage 8 --fast-stride --skip-train"
echo "  Re-train:       python3 train.py --epochs 50 --resume checkpoints/fcnn_best.pt"
echo "  Single pair:    python3 inference/fusion.py --model checkpoints/fcnn_best.pt \\"
echo "                    --pair-dir datasets/real/dataset1_lytro/pair_001 --stride 2"
echo ""
