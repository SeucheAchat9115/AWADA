#!/usr/bin/env python3
"""Train CyCada (CycleGAN + semantic consistency loss) for domain translation."""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from awada.config import DEFAULT_DEVICE
from awada.datasets.unpaired_dataset import UnpairedImageDataset
from awada.models.cycada import CyCada
from awada.utils.train_utils import get_lambda_lr, load_config, set_seed

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train CyCada")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config",
        default="configs/cycada.yaml",
        help="Path to YAML config file with hyperparameters",
    )
    # Hyperparameters – CLI flags override the config file when provided
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda_cyc", type=float)
    parser.add_argument("--lambda_gan", type=float)
    parser.add_argument("--lambda_idt", type=float, help="Identity loss weight (0 = disabled)")
    parser.add_argument(
        "--lambda_sem",
        type=float,
        help="Semantic consistency loss weight (0 = disabled, no DeepLabV3 loaded)",
    )
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (also always saves the final epoch)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable Automatic Mixed Precision (AMP) training",
    )
    args = parser.parse_args()

    set_seed(args.seed)

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
    lambda_sem = args.lambda_sem if args.lambda_sem is not None else cfg.get("lambda_sem", 1.0)
    patch_size = args.patch_size if args.patch_size is not None else cfg.get("patch_size", 128)
    betas = tuple(cfg.get("betas", [0.5, 0.999]))
    device_str = args.device if args.device is not None else cfg.get("device", DEFAULT_DEVICE)
    buffer_size = cfg.get("buffer_size", 50)
    buffer_return_prob = cfg.get("buffer_return_prob", 0.5)
    disc_loss_avg_factor = cfg.get("disc_loss_avg_factor", 0.5)
    log_interval = cfg.get("log_interval", 100)

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

    model = CyCada(
        device=str(device),
        lambda_sem=lambda_sem,
        buffer_size=buffer_size,
        buffer_return_prob=buffer_return_prob,
        disc_loss_avg_factor=disc_loss_avg_factor,
    )

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

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.G_AB.load_state_dict(ckpt["G_AB"])
        model.G_BA.load_state_dict(ckpt["G_BA"])
        model.D_A.load_state_dict(ckpt["D_A"])
        model.D_B.load_state_dict(ckpt["D_B"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch = ckpt["epoch"]
        logger.info("Resumed from checkpoint: %s (epoch %d)", args.resume, start_epoch)

    for epoch in range(start_epoch, epochs):
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
            g_losses = model.compute_generator_loss(lambda_cyc, lambda_gan, lambda_idt, lambda_sem)
            for loss_name, loss_val in g_losses.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(
                        f"Non-finite generator loss '{loss_name}' detected at epoch {epoch + 1}, "
                        f"iteration {iteration + 1}: {loss_val.item()}"
                    )
            g_losses["total_G"].backward()
            opt_G.step()

            # Update discriminators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad()
            d_losses = model.compute_discriminator_loss()
            for loss_name, loss_val in d_losses.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(
                        f"Non-finite discriminator loss '{loss_name}' detected at epoch {epoch + 1}, "
                        f"iteration {iteration + 1}: {loss_val.item()}"
                    )
            d_losses["total_D"].backward()
            opt_D.step()

            if (iteration + 1) % log_interval == 0:
                logger.info(
                    "  [Epoch %d, Iter %d] G=%.3f D=%.3f cyc=%.3f",
                    epoch + 1,
                    iteration + 1,
                    g_losses["total_G"].item(),
                    d_losses["total_D"].item(),
                    g_losses["cycle_A"].item() + g_losses["cycle_B"].item(),
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
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict(),
            "sched_D": sched_D.state_dict(),
        }
        ckpt_path = os.path.join(args.output_dir, f"cycada_epoch_{epoch + 1}.pth")
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == epochs:
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    logger.info("CyCada training complete.")


if __name__ == "__main__":
    main()
