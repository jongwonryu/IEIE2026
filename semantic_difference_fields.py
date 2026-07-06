import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image


@dataclass
class DifferenceFieldResult:
    aggregate_map: np.ndarray
    per_step_maps: np.ndarray
    timesteps: List[int]
    text_difference_norm: float
    source_prompt: str
    target_prompt: str


def parse_step_indices(value: str) -> List[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--step-indices must contain at least one index.")
    return indices


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def save_heatmap(array: np.ndarray, path: Path) -> None:
    normalized = normalize_map(array)
    red = (normalized * 255).astype(np.uint8)
    green = np.zeros_like(red, dtype=np.uint8)
    blue = ((1.0 - normalized) * 255).astype(np.uint8)
    heatmap = np.stack([red, green, blue], axis=-1)
    Image.fromarray(heatmap).save(path)


class SemanticDifferenceFieldAnalyzer:
    def __init__(self, model: str, device: str = "cuda", dtype: str = "auto"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        torch_dtype = self._resolve_dtype(dtype)
        self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch_dtype)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == "auto":
            return torch.float16 if self.device.type == "cuda" else torch.float32
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp32":
            return torch.float32
        raise ValueError("dtype must be one of: auto, fp16, bf16, fp32")

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds

    def sample_latent(self, height: int, width: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        channels = self.pipe.unet.config.in_channels
        shape = (1, channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.pipe.unet.dtype)

    @torch.no_grad()
    def predict_noise(self, latent: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        latent_input = self.pipe.scheduler.scale_model_input(latent, timestep)
        return self.pipe.unet(latent_input, timestep, encoder_hidden_states=prompt_embeds).sample

    @torch.no_grad()
    def compute(
        self,
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int,
        step_indices: List[int],
        height: int,
        width: int,
        seed: int,
    ) -> DifferenceFieldResult:
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        schedule = self.pipe.scheduler.timesteps

        max_index = len(schedule) - 1
        for index in step_indices:
            if index < 0 or index > max_index:
                raise ValueError(f"step index {index} is outside the schedule range [0, {max_index}]")

        source_embeds = self.encode_prompt(source_prompt)
        target_embeds = self.encode_prompt(target_prompt)
        text_difference_norm = float(torch.linalg.vector_norm((target_embeds - source_embeds).float()).item())

        latent = self.sample_latent(height=height, width=width, seed=seed)
        per_step_maps = []
        selected_timesteps = []

        for index in step_indices:
            timestep = schedule[index]
            source_noise = self.predict_noise(latent, timestep, source_embeds)
            target_noise = self.predict_noise(latent, timestep, target_embeds)
            noise_difference = target_noise - source_noise

            magnitude = noise_difference.float().square().mean(dim=1, keepdim=True).sqrt()
            magnitude = F.interpolate(magnitude, size=(height, width), mode="bilinear", align_corners=False)
            per_step_maps.append(magnitude.squeeze().cpu().numpy())
            selected_timesteps.append(int(timestep.item()))

        stacked = np.stack(per_step_maps, axis=0).astype(np.float32)
        aggregate = stacked.mean(axis=0)

        return DifferenceFieldResult(
            aggregate_map=aggregate,
            per_step_maps=stacked,
            timesteps=selected_timesteps,
            text_difference_norm=text_difference_norm,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
        )


def save_result(result: DifferenceFieldResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "semantic_difference_map.npy", result.aggregate_map)
    np.save(output_dir / "per_step_maps.npy", result.per_step_maps)
    save_heatmap(result.aggregate_map, output_dir / "semantic_difference_map.png")

    metadata = {
        "source_prompt": result.source_prompt,
        "target_prompt": result.target_prompt,
        "timesteps": result.timesteps,
        "text_difference_norm": result.text_difference_norm,
        "map_shape": list(result.aggregate_map.shape),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute semantic difference fields in diffusion noise space.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--source-prompt", type=str, required=True)
    parser.add_argument("--target-prompt", type=str, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--step-indices", type=parse_step_indices, default=parse_step_indices("5,15,25,35"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_difference"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = SemanticDifferenceFieldAnalyzer(model=args.model, device=args.device, dtype=args.dtype)
    result = analyzer.compute(
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        num_inference_steps=args.num_inference_steps,
        step_indices=args.step_indices,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    save_result(result, args.output_dir)
    print(f"Saved semantic difference field outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
