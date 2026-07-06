# Semantic Difference Fields in Diffusion Models

This repository contains lightweight software for analyzing semantic
difference fields in text-guided diffusion models.

The accompanying survey is titled:

**Semantic Difference Fields in Diffusion Models: A Survey**

The main idea is simple. Given the same noisy latent `x_t`, a source prompt
produces a noise prediction `n_s`, and a target prompt produces a noise
prediction `n_t`. Their difference

```text
n_d = n_t - n_s
```

can be interpreted as the denoising response caused by the semantic change
from the source prompt to the target prompt. The spatial magnitude of this
difference can also be used as an edit-localization signal:

```text
M_t(x, y) = ||n_d^(t)(x, y)||
```

## Key Files

- `semantic_difference_fields.py`: computes prompt-conditioned noise
  prediction differences and localization maps.
- `scripts/run_sdf_analysis.sh`: example command for running the analyzer.
- `software_register.txt`: Korean software registration description.
- `requirements.txt`: minimal Python dependencies.

## Install

```bash
pip install -r requirements.txt
```

The script uses Hugging Face Diffusers and can run with a local Stable
Diffusion checkpoint or a Hub model name.

## Run

```bash
SOURCE_PROMPT="a red car" \
TARGET_PROMPT="a blue car" \
OUTPUT_DIR="outputs/red_to_blue" \
bash scripts/run_sdf_analysis.sh
```

You can also run the Python file directly:

```bash
python semantic_difference_fields.py \
  --model runwayml/stable-diffusion-v1-5 \
  --source-prompt "a red car" \
  --target-prompt "a blue car" \
  --num-inference-steps 50 \
  --step-indices 5,15,25,35 \
  --output-dir outputs/red_to_blue
```

The output directory contains:

- `semantic_difference_map.npy`: aggregated spatial magnitude map.
- `semantic_difference_map.png`: normalized heatmap visualization.
- `per_step_maps.npy`: per-timestep magnitude maps.
- `metadata.json`: prompts, timesteps, and text-difference statistics.

## Notes

This code is intended as an analysis tool, not a full image editing pipeline.
It implements the survey's core measurement: comparing source- and
target-conditioned denoising responses under the same noisy latent.
