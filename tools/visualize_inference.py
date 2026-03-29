#!/usr/bin/env python3
"""Visualize CycleGAN / AWADA style transfer on a set of images.

For each input image the script produces a side-by-side PNG showing:
  left  – original source image
  right – style-translated output

Example usage::

    python visualize_inference.py \\
        visualize.checkpoint=outputs/awada_gan/awada_epoch_200.pth \\
        visualize.input_dir=/data/sim10k/images \\
        visualize.output_dir=outputs/visualizations

    # Use the inverse generator (B → A) instead of the default (A → B):
    python visualize_inference.py \\
        visualize.checkpoint=outputs/awada_gan/awada_epoch_200.pth \\
        visualize.input_dir=/data/cityscapes/leftImg8bit/val/aachen \\
        visualize.output_dir=outputs/visualizations \\
        visualize.direction=BA
"""

import os

import hydra
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from awada.models.generator import ResNetGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_generator(checkpoint_path: str, direction: str, device: torch.device) -> ResNetGenerator:
    """Load the requested generator (G_AB or G_BA) from a checkpoint."""
    gen = ResNetGenerator().to(device)
    state = torch.load(checkpoint_path, map_location=device)

    key = "G_AB" if direction == "AB" else "G_BA"
    if isinstance(state, dict) and key in state:
        gen.load_state_dict(state[key])
    elif isinstance(state, dict) and "model_state_dict" in state:
        gen.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and not any(
        k in state for k in ("G_AB", "G_BA", "model_state_dict")
    ):
        # Assume the dict is already a state_dict
        gen.load_state_dict(state)
    else:
        raise ValueError(
            f"Cannot find key '{key}' in checkpoint. Available keys: {list(state.keys())}"
        )

    gen.eval()
    return gen


def pad_to_multiple(img: Image.Image, multiple: int = 4) -> tuple:
    """Pad image so that width and height are divisible by *multiple*.

    Returns:
        padded_img: the padded PIL image
        original_size: (w, h) of the original image (used for cropping back)
    """
    w, h = img.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    if pad_w > 0 or pad_h > 0:
        img = T.Pad((0, 0, pad_w, pad_h))(img)
    return img, (w, h)


def translate_image(
    generator: ResNetGenerator,
    img: Image.Image,
    device: torch.device,
) -> Image.Image:
    """Run a single PIL image through the generator and return the output."""
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    padded, (orig_w, orig_h) = pad_to_multiple(img.convert("RGB"), multiple=4)
    tensor = normalize(to_tensor(padded)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = generator(tensor)

    # Denormalize [-1, 1] → [0, 1] and crop back to original size
    output = (output.squeeze(0).cpu() + 1.0) / 2.0
    output = output.clamp(0.0, 1.0)[:, :orig_h, :orig_w]

    return TF.to_pil_image(output)


def make_side_by_side(
    original: Image.Image,
    translated: Image.Image,
    label_original: str = "Original",
    label_translated: str = "Translated",
    label_height: int = 24,
) -> Image.Image:
    """Concatenate two images horizontally with caption labels."""
    w1, h1 = original.size
    w2, h2 = translated.size

    # Resize translated to match the height of the original if they differ
    if h1 != h2:
        translated = translated.resize((int(w2 * h1 / h2), h1), Image.LANCZOS)
        w2, h2 = translated.size

    canvas_w = w1 + w2
    canvas_h = h1 + label_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    canvas.paste(original, (0, label_height))
    canvas.paste(translated, (w1, label_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((4, 4), label_original, fill=(220, 220, 220), font=font)
    draw.text((w1 + 4, 4), label_translated, fill=(220, 220, 220), font=font)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _visualize(cfg: DictConfig) -> None:
    """Run visualization from a Hydra config."""
    os.makedirs(cfg.visualize.output_dir, exist_ok=True)
    device = torch.device(cfg.hardware.device)

    print(f"Loading generator ({cfg.visualize.direction}) from {cfg.visualize.checkpoint} …")
    generator = load_generator(cfg.visualize.checkpoint, cfg.visualize.direction, device)

    image_files = sorted(
        f
        for f in os.listdir(cfg.visualize.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if cfg.visualize.num_images is not None:
        image_files = image_files[: cfg.visualize.num_images]

    print(f"Translating {len(image_files)} image(s) → {cfg.visualize.output_dir}")
    for fname in tqdm(image_files, desc="Visualizing"):
        src_path = os.path.join(cfg.visualize.input_dir, fname)
        original = Image.open(src_path).convert("RGB")

        translated = translate_image(generator, original, device)

        vis = make_side_by_side(
            original,
            translated,
            label_original="Original",
            label_translated=f"Translated ({cfg.visualize.direction})",
        )

        # Always save as PNG regardless of input format
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(cfg.visualize.output_dir, f"{stem}_vis.png")
        vis.save(out_path)

    print(f"Done. {len(image_files)} visualization(s) saved to {cfg.visualize.output_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="visualize_inference")
def main(cfg: DictConfig) -> None:
    _visualize(cfg)


if __name__ == "__main__":
    main()
