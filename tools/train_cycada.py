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
from awada.utils.train_utils import get_lambda_lr, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    setup_logging(args.output_dir)
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

    scaler_G = torch.cuda.amp.GradScaler(enabled=args.amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=args.amp)

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
        if "scaler_G" in ckpt:
            scaler_G.load_state_dict(ckpt["scaler_G"])
        if "scaler_D" in ckpt:
            scaler_D.load_state_dict(ckpt["scaler_D"])
        start_epoch = ckpt["epoch"]
        logger.info("Resumed from checkpoint: %s (epoch %d)", args.resume, start_epoch)

    for epoch in range(start_epoch, epochs):
        logger.info("--- Epoch %d/%d ---", epoch + 1, epochs)
        model.G_AB.train()
        model.G_BA.train()
        model.D_A.train()
        model.D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_sem = 0.0
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_sem = 0.0
        num_iters = 0

        for iteration, (real_A, real_B) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            with torch.cuda.amp.autocast(enabled=args.amp):
                model.set_input(real_A, real_B)
                model.forward()

            # Update generators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                g_losses = model.compute_generator_loss(
                    lambda_cyc, lambda_gan, lambda_idt, lambda_sem
                )
            for loss_name, loss_val in g_losses.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(
                        f"Non-finite generator loss '{loss_name}' detected at epoch {epoch + 1}, "
                        f"iteration {iteration + 1}: {loss_val.item()}"
                    )
            scaler_G.scale(g_losses["total_G"]).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # Update discriminators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                d_losses = model.compute_discriminator_loss()
            for loss_name, loss_val in d_losses.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(
                        f"Non-finite discriminator loss '{loss_name}' detected at epoch {epoch + 1}, "
                        f"iteration {iteration + 1}: {loss_val.item()}"
                    )
            scaler_D.scale(d_losses["total_D"]).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            epoch_loss_G += g_losses["total_G"].item()
            epoch_loss_D += d_losses["total_D"].item()
            running_loss_G += g_losses["total_G"].item()
            running_loss_D += d_losses["total_D"].item()
            if "sem_AB" in g_losses:
                sem_val = g_losses["sem_AB"].item() + g_losses["sem_BA"].item()
                epoch_loss_sem += sem_val
                running_loss_sem += sem_val
            num_iters += 1

            if (iteration + 1) % log_interval == 0:
                if lambda_sem > 0:
                    logger.info(
                        "  [Epoch %d, Iter %d] G=%.3f D=%.3f cyc=%.3f sem=%.3f",
                        epoch + 1,
                        iteration + 1,
                        running_loss_G / log_interval,
                        running_loss_D / log_interval,
                        g_losses["cycle_A"].item() + g_losses["cycle_B"].item(),
                        running_loss_sem / log_interval,
                    )
                else:
                    logger.info(
                        "  [Epoch %d, Iter %d] G=%.3f D=%.3f cyc=%.3f",
                        epoch + 1,
                        iteration + 1,
                        running_loss_G / log_interval,
                        running_loss_D / log_interval,
                        g_losses["cycle_A"].item() + g_losses["cycle_B"].item(),
                    )
                running_loss_G = 0.0
                running_loss_D = 0.0
                running_loss_sem = 0.0

        if lambda_sem > 0:
            logger.info(
                "Epoch %d complete | avg G loss=%.4f | avg D loss=%.4f | avg sem loss=%.4f",
                epoch + 1,
                epoch_loss_G / max(num_iters, 1),
                epoch_loss_D / max(num_iters, 1),
                epoch_loss_sem / max(num_iters, 1),
            )
        else:
            logger.info(
                "Epoch %d complete | avg G loss=%.4f | avg D loss=%.4f",
                epoch + 1,
                epoch_loss_G / max(num_iters, 1),
                epoch_loss_D / max(num_iters, 1),
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
            "scaler_G": scaler_G.state_dict(),
            "scaler_D": scaler_D.state_dict(),
        }
        ckpt_path = os.path.join(args.output_dir, f"cycada_epoch_{epoch + 1}.pth")
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == epochs:
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    logger.info("CyCada training complete.")


if __name__ == "__main__":
    main()
