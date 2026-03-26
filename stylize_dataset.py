#!/usr/bin/env python3
"""Stylize source domain images using a trained CycleGAN/AWADA generator."""

import argparse
import os

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.models.generator import ResNetGenerator


def main():
    parser = argparse.ArgumentParser(description="Stylize source domain images")
    parser.add_argument("--generator_checkpoint", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    generator = ResNetGenerator().to(device)
    state = torch.load(args.generator_checkpoint, map_location=device)
    # Checkpoint might be full CycleGAN or just G_AB
    if isinstance(state, dict):
        if "G_AB" in state:
            generator.load_state_dict(state["G_AB"])
        elif "model_state_dict" in state:
            generator.load_state_dict(state["model_state_dict"])
        else:
            generator.load_state_dict(state)
    generator.eval()

    # Collect images recursively, preserving relative paths for nested structures
    # (e.g. Cityscapes stores images under leftImg8bit/train/<city>/*.png)
    image_rel_paths = []
    for dirpath, _dirnames, filenames in os.walk(args.source_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_path = os.path.relpath(os.path.join(dirpath, fname), args.source_dir)
                image_rel_paths.append(rel_path)
    image_rel_paths = sorted(image_rel_paths)

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    with torch.no_grad():
        for rel_path in tqdm(image_rel_paths, desc="Stylizing"):
            img_path = os.path.join(args.source_dir, rel_path)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Pad to be divisible by 4 (generator stride)
            pad_w = (4 - w % 4) % 4
            pad_h = (4 - h % 4) % 4
            if pad_w > 0 or pad_h > 0:
                img = T.Pad((0, 0, pad_w, pad_h))(img)

            tensor = normalize(to_tensor(img)).unsqueeze(0).to(device)
            output = generator(tensor)

            # Denormalize from [-1,1] to [0,1]
            output = (output.squeeze(0).cpu() + 1) / 2
            output = output.clamp(0, 1)

            # Crop back to original size
            output = output[:, :h, :w]

            out_img = TF.to_pil_image(output)
            out_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out_img.save(out_path)

    print(f"Stylized {len(image_rel_paths)} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
