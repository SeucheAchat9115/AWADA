import torch
import torch.nn.functional as F

from .cycada import CyCada


class AWADA(CyCada):
    """AWADA: Attention-Weighted Adversarial Domain Adaptation.

    Extends CyCada with attention-masked adversarial losses.  Foreground
    attention masks (derived from RPN proposals) bias the GAN loss towards
    semantically meaningful regions, while cycle and semantic losses remain
    unmasked global regularisers.
    """

    def __init__(
        self,
        device: str = "cuda",
        lambda_sem: float = 0.0,
        buffer_size: int = 50,
        buffer_return_prob: float = 0.5,
        disc_loss_avg_factor: float = 0.5,
    ):
        super().__init__(
            device=device,
            lambda_sem=lambda_sem,
            buffer_size=buffer_size,
            buffer_return_prob=buffer_return_prob,
            disc_loss_avg_factor=disc_loss_avg_factor,
        )

    def set_input(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        attention_A: torch.Tensor | None = None,
        attention_B: torch.Tensor | None = None,
    ) -> None:
        super().set_input(real_A, real_B)
        # attention_A: binary mask for source, attention_B: binary mask for target
        # Both [B, 1, H, W] float tensors
        self.attention_A = attention_A.to(self.device) if attention_A is not None else None
        self.attention_B = attention_B.to(self.device) if attention_B is not None else None

    def _masked_mse_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """MSE loss with spatial mask applied to discriminator output map."""
        if mask is None:
            return ((pred - target) ** 2).mean()
        # Resize mask to match discriminator output spatial size
        mask_resized = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
        # Weight: foreground = 1, background = 0 (masked out)
        weight = mask_resized
        return (weight * (pred - target) ** 2).mean()

    def compute_generator_loss(
        self,
        lambda_cyc: float = 10.0,
        lambda_gan: float = 1.0,
        lambda_idt: float = 0.0,
        lambda_sem: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # Build losses using CyCada (cycle, identity, semantic) with unmasked GAN
        losses = super().compute_generator_loss(lambda_cyc, lambda_gan, lambda_idt, lambda_sem)
        total = losses.pop("total_G")

        # Replace unmasked GAN losses with MASKED versions
        total = total - losses["G_AB"] - losses["G_BA"]

        # GAN loss (MASKED by attention — foreground-focused adversarial objective)
        pred_fake_B = self.D_B(self.fake_B)
        loss_G_AB = (
            self._masked_mse_loss(pred_fake_B, torch.ones_like(pred_fake_B), self.attention_A)
            * lambda_gan
        )
        pred_fake_A = self.D_A(self.fake_A)
        loss_G_BA = (
            self._masked_mse_loss(pred_fake_A, torch.ones_like(pred_fake_A), self.attention_B)
            * lambda_gan
        )

        losses["G_AB"] = loss_G_AB
        losses["G_BA"] = loss_G_BA
        losses["total_G"] = total + loss_G_AB + loss_G_BA
        return losses

    def compute_discriminator_loss(self) -> dict[str, torch.Tensor]:
        # D_B with attention mask from source (A -> B translation)
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B.detach())
        pred_real_B = self.D_B(self.real_B)
        loss_D_B_real = self._masked_mse_loss(
            pred_real_B, torch.ones_like(pred_real_B), self.attention_B
        )
        pred_fake_B = self.D_B(fake_B)
        loss_D_B_fake = self._masked_mse_loss(
            pred_fake_B, torch.zeros_like(pred_fake_B), self.attention_A
        )
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * self.disc_loss_avg_factor

        # D_A with attention mask
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A.detach())
        pred_real_A = self.D_A(self.real_A)
        loss_D_A_real = self._masked_mse_loss(
            pred_real_A, torch.ones_like(pred_real_A), self.attention_A
        )
        pred_fake_A = self.D_A(fake_A)
        loss_D_A_fake = self._masked_mse_loss(
            pred_fake_A, torch.zeros_like(pred_fake_A), self.attention_B
        )
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * self.disc_loss_avg_factor

        return {"D_A": loss_D_A, "D_B": loss_D_B, "total_D": loss_D_A + loss_D_B}
