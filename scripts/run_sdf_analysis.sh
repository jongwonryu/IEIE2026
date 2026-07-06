#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-runwayml/stable-diffusion-v1-5}"
SOURCE_PROMPT="${SOURCE_PROMPT:-a red car}"
TARGET_PROMPT="${TARGET_PROMPT:-a blue car}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/semantic_difference}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
STEP_INDICES="${STEP_INDICES:-5,15,25,35}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

python "$REPO_ROOT/semantic_difference_fields.py" \
  --model "$MODEL" \
  --source-prompt "$SOURCE_PROMPT" \
  --target-prompt "$TARGET_PROMPT" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --step-indices "$STEP_INDICES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR"
