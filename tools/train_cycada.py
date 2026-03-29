#!/usr/bin/env python3
"""Train CyCada (CycleGAN + semantic consistency loss) for domain translation."""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from awada.datasets.unpaired_dataset import UnpairedImageDataset
from awada.models.cycada import CyCada
from awada.utils.train_utils import get_lambda_lr, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _train(cfg: DictConfig) -> None:
    """Run CyCada training from a Hydra config."""
    set_seed(cfg.training.seed)

    os.makedirs(cfg.data.output_dir, exist_ok=True)
    setup_logging(cfg.data.output_dir)
    device = torch.device(cfg.hardware.device)

    dataset = UnpairedImageDataset(
        cfg.data.source_dir, cfg.data.target_dir, cfg.training.patch_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = CyCada(
        device=str(device),
        lambda_sem=cfg.model.lambda_sem,
        buffer_size=cfg.model.buffer_size,
        buffer_return_prob=cfg.model.buffer_return_prob,
        disc_loss_avg_factor=cfg.model.disc_loss_avg_factor,
    )

    epochs = cfg.training.epochs
    n_epochs_decay = epochs // 2
    n_epochs_stable = epochs - n_epochs_decay

    betas = tuple(cfg.training.betas)
    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=cfg.training.lr,
        betas=betas,
    )
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=cfg.training.lr,
        betas=betas,
    )

    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda ep: get_lambda_lr(ep, n_epochs_stable, n_epochs_decay)
    )

    scaler_G = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    start_epoch = 0
    if cfg.training.resume:
        ckpt = torch.load(cfg.training.resume, map_location=device)
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
        logger.info("Resumed from checkpoint: %s (epoch %d)", cfg.training.resume, start_epoch)

    for epoch in range(start_epoch, epochs):
        model.G_AB.train()
        model.G_BA.train()
        model.D_A.train()
        model.D_B.train()

        for iteration, (real_A, real_B) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            with torch.cuda.amp.autocast(enabled=cfg.training.amp):
                model.set_input(real_A, real_B)
                model.forward()

            # Update generators
            for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.training.amp):
                g_losses = model.compute_generator_loss(
                    cfg.model.lambda_cyc,
                    cfg.model.lambda_gan,
                    cfg.model.lambda_idt,
                    cfg.model.lambda_sem,
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
            with torch.cuda.amp.autocast(enabled=cfg.training.amp):
                d_losses = model.compute_discriminator_loss()
            for loss_name, loss_val in d_losses.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(
                        f"Non-finite discriminator loss '{loss_name}' detected at epoch "
                        f"{epoch + 1}, iteration {iteration + 1}: {loss_val.item()}"
                    )
            scaler_D.scale(d_losses["total_D"]).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            if (iteration + 1) % cfg.training.log_interval == 0:
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
        ckpt_path = os.path.join(cfg.data.output_dir, f"cycada_epoch_{epoch + 1}.pth")
        if (epoch + 1) % cfg.training.save_every == 0 or (epoch + 1) == epochs:
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    logger.info("CyCada training complete.")


@hydra.main(version_base=None, config_path="../configs", config_name="train_cycada")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _train(cfg)


if __name__ == "__main__":
    main()
