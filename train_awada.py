#!/usr/bin/env python3
"""Train AWADA CycleGAN with attention-masked adversarial losses."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.attention_dataset import AttentionPairedDataset
from src.models.awada_cyclegan import AWADACycleGAN


def get_lambda_lr(epoch, n_epochs, n_epochs_decay, offset=0):
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / float(n_epochs_decay + 1))


def main():
    parser = argparse.ArgumentParser(description='Train AWADA CycleGAN')
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--attention_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lambda_cyc', type=float, default=10.0)
    parser.add_argument('--lambda_idt', type=float, default=5.0)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = AttentionPairedDataset(
        args.source_dir, args.target_dir, args.attention_dir,
        patch_size=args.patch_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    model = AWADACycleGAN(device=str(device))

    n_epochs_decay = args.epochs // 2
    n_epochs_stable = args.epochs - n_epochs_decay

    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )

    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )

    for epoch in range(args.epochs):
        model.G_AB.train(); model.G_BA.train()
        model.D_A.train(); model.D_B.train()

        for iteration, (real_A, real_B, att_A) in enumerate(
            tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        ):
            # att_B is not used (no target domain attention); pass None
            model.set_input(real_A, real_B, attention_A=att_A, attention_B=None)
            model.forward()

            # Update generators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            g_losses = model.compute_generator_loss(args.lambda_cyc, args.lambda_idt)
            g_losses['total_G'].backward()
            opt_G.step()

            # Update discriminators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad()
            d_losses = model.compute_discriminator_loss()
            d_losses['total_D'].backward()
            opt_D.step()

            if (iteration + 1) % 100 == 0:
                print(f'  [Epoch {epoch+1}, Iter {iteration+1}] '
                      f'G={g_losses["total_G"].item():.3f} '
                      f'D={d_losses["total_D"].item():.3f} '
                      f'cyc={g_losses["cycle_A"].item() + g_losses["cycle_B"].item():.3f}')

        sched_G.step(); sched_D.step()

        ckpt = {
            'epoch': epoch + 1,
            'G_AB': model.G_AB.state_dict(),
            'G_BA': model.G_BA.state_dict(),
            'D_A': model.D_A.state_dict(),
            'D_B': model.D_B.state_dict(),
        }
        ckpt_path = os.path.join(args.output_dir, f'awada_epoch_{epoch+1}.pth')
        torch.save(ckpt, ckpt_path)
        print(f'Checkpoint saved: {ckpt_path}')

    print('AWADA training complete.')


if __name__ == '__main__':
    main()
