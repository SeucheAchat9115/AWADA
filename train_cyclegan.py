#!/usr/bin/env python3
"""Train standard CycleGAN for domain translation."""

import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.cyclegan import CycleGAN


class UnpairedImageDataset(Dataset):
    def __init__(self, dir_A, dir_B, patch_size=128):
        self.patch_size = patch_size
        self.files_A = sorted(
            [
                os.path.join(dir_A, f)
                for f in os.listdir(dir_A)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.files_B = sorted(
            [
                os.path.join(dir_B, f)
                for f in os.listdir(dir_B)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.transform = T.Compose(
            [
                T.RandomCrop(patch_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A = Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[idx % len(self.files_B)]).convert("RGB")
        return self.transform(img_A), self.transform(img_B)


def get_lambda_lr(epoch, n_epochs, n_epochs_decay):
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / float(n_epochs_decay + 1))


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lambda_cyc", type=float, default=10.0)
    parser.add_argument("--lambda_idt", type=float, default=5.0)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = UnpairedImageDataset(args.source_dir, args.target_dir, args.patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = CycleGAN(device=str(device))

    n_epochs_decay = args.epochs // 2
    n_epochs_stable = args.epochs - n_epochs_decay

    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )

    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )

    for epoch in range(args.epochs):
        model.G_AB.train()
        model.G_BA.train()
        model.D_A.train()
        model.D_B.train()

        for iteration, (real_A, real_B) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        ):
            model.set_input(real_A, real_B)
            model.forward()

            # Update generators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            g_losses = model.compute_generator_loss(
                args.lambda_cyc, args.lambda_idt, args.lambda_gan
            )
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
                    f'  [Epoch {epoch+1}, Iter {iteration+1}] '
                    f'G={g_losses["total_G"].item():.3f} '
                    f'D={d_losses["total_D"].item():.3f} '
                    f'cyc={g_losses["cycle_A"].item() + g_losses["cycle_B"].item():.3f}'
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
        ckpt_path = os.path.join(args.output_dir, f"cyclegan_epoch_{epoch+1}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
