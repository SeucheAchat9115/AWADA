#!/usr/bin/env python3
"""Train standard CycleGAN for domain translation."""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.unpaired_dataset import UnpairedImageDataset
from src.models.cyclegan import CycleGAN
from src.utils.train_utils import get_lambda_lr, load_config


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config",
        default="configs/cyclegan.yaml",
        help="Path to YAML config file with hyperparameters",
    )
    # Hyperparameters – CLI flags override the config file when provided
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda_cyc", type=float)
    parser.add_argument("--lambda_gan", type=float)
    parser.add_argument("--lambda_idt", type=float, help="Identity loss weight (0 = disabled)")
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    # Load defaults from config file, then apply CLI overrides
    cfg: dict = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 200)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)
    lr = args.lr if args.lr is not None else cfg.get("lr", 0.0002)
    lambda_cyc = args.lambda_cyc if args.lambda_cyc is not None else cfg.get("lambda_cyc", 10.0)
    lambda_gan = args.lambda_gan if args.lambda_gan is not None else cfg.get("lambda_gan", 1.0)
    lambda_idt = args.lambda_idt if args.lambda_idt is not None else cfg.get("lambda_idt", 0.0)
    patch_size = args.patch_size if args.patch_size is not None else cfg.get("patch_size", 128)
    betas = tuple(cfg.get("betas", [0.5, 0.999]))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = args.device if args.device is not None else cfg.get("device", default_device)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(device_str)

    dataset = UnpairedImageDataset(args.source_dir, args.target_dir, patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = CycleGAN(device=str(device))

    n_epochs_decay = epochs // 2
    n_epochs_stable = epochs - n_epochs_decay

    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=lr,
        betas=betas,
    )
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=lr, betas=betas
    )

    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )

    for epoch in range(epochs):
        model.G_AB.train()
        model.G_BA.train()
        model.D_A.train()
        model.D_B.train()

        for iteration, (real_A, real_B) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            model.set_input(real_A, real_B)
            model.forward()

            # Update generators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            g_losses = model.compute_generator_loss(lambda_cyc, lambda_gan, lambda_idt)
            g_losses["total_G"].backward()
            opt_G.step()

            # Update discriminators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad()
            d_losses = model.compute_discriminator_loss()
            d_losses["total_D"].backward()
            opt_D.step()

            if (iteration + 1) % 100 == 0:
                print(
                    f"  [Epoch {epoch + 1}, Iter {iteration + 1}] "
                    f"G={g_losses['total_G'].item():.3f} "
                    f"D={d_losses['total_D'].item():.3f} "
                    f"cyc={g_losses['cycle_A'].item() + g_losses['cycle_B'].item():.3f}"
                )

        sched_G.step()
        sched_D.step()

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "G_AB": model.G_AB.state_dict(),
            "G_BA": model.G_BA.state_dict(),
            "D_A": model.D_A.state_dict(),
            "D_B": model.D_B.state_dict(),
        }
        ckpt_path = os.path.join(args.output_dir, f"cyclegan_epoch_{epoch + 1}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
