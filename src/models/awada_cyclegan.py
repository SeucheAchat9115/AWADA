import torch
import torch.nn.functional as F
from .cyclegan import CycleGAN


class AWADACycleGAN(CycleGAN):
    def set_input(self, real_A, real_B, attention_A=None, attention_B=None):
        super().set_input(real_A, real_B)
        # attention_A: binary mask for source, attention_B: binary mask for target
        # Both [B, 1, H, W] float tensors
        self.attention_A = attention_A.to(self.device) if attention_A is not None else None
        self.attention_B = attention_B.to(self.device) if attention_B is not None else None

    def _masked_mse_loss(self, pred, target, mask):
        """MSE loss with spatial mask applied to discriminator output map."""
        if mask is None:
            return ((pred - target) ** 2).mean()
        # Resize mask to match discriminator output spatial size
        mask_resized = F.interpolate(mask, size=pred.shape[2:], mode='nearest')
        # Weight: foreground = 2, background = 1, normalized
        weight = 1.0 + mask_resized  # values: 1 (background) or 2 (foreground)
        weight = weight / weight.mean()
        return (weight * (pred - target) ** 2).mean()

    def compute_generator_loss(self, lambda_cyc=10.0, lambda_idt=5.0):
        # Identity loss (UNMASKED)
        idt_A = self.G_BA(self.real_A)
        idt_B = self.G_AB(self.real_B)
        loss_idt_A = self.criterion_identity(idt_A, self.real_A) * lambda_idt
        loss_idt_B = self.criterion_identity(idt_B, self.real_B) * lambda_idt

        # GAN loss (MASKED by attention)
        pred_fake_B = self.D_B(self.fake_B)
        loss_G_AB = self._masked_mse_loss(pred_fake_B, torch.ones_like(pred_fake_B), self.attention_A)
        pred_fake_A = self.D_A(self.fake_A)
        loss_G_BA = self._masked_mse_loss(pred_fake_A, torch.ones_like(pred_fake_A), self.attention_B)

        # Cycle consistency loss (UNMASKED)
        loss_cyc_A = self.criterion_cycle(self.rec_A, self.real_A) * lambda_cyc
        loss_cyc_B = self.criterion_cycle(self.rec_B, self.real_B) * lambda_cyc

        total = loss_G_AB + loss_G_BA + loss_cyc_A + loss_cyc_B + loss_idt_A + loss_idt_B
        return {
            'G_AB': loss_G_AB, 'G_BA': loss_G_BA,
            'cycle_A': loss_cyc_A, 'cycle_B': loss_cyc_B,
            'idt_A': loss_idt_A, 'idt_B': loss_idt_B,
            'total_G': total,
        }

    def compute_discriminator_loss(self):
        # D_B with attention mask from source (A -> B translation)
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B.detach())
        pred_real_B = self.D_B(self.real_B)
        loss_D_B_real = self._masked_mse_loss(pred_real_B, torch.ones_like(pred_real_B), self.attention_B)
        pred_fake_B = self.D_B(fake_B)
        loss_D_B_fake = self._masked_mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B), self.attention_A)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        # D_A with attention mask
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A.detach())
        pred_real_A = self.D_A(self.real_A)
        loss_D_A_real = self._masked_mse_loss(pred_real_A, torch.ones_like(pred_real_A), self.attention_A)
        pred_fake_A = self.D_A(fake_A)
        loss_D_A_fake = self._masked_mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A), self.attention_B)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        return {'D_A': loss_D_A, 'D_B': loss_D_B, 'total_D': loss_D_A + loss_D_B}
