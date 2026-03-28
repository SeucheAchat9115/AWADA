import random

import torch
import torch.nn as nn

from .discriminator import PatchGANDiscriminator
from .generator import ResNetGenerator


class ImageBuffer:
    """Replay buffer to stabilize discriminator training.
    Size of 50 matches the original CycleGAN implementation (Zhu et al., 2017).
    """

    def __init__(self, max_size: int = 50, return_prob: float = 0.5):
        self.max_size = max_size
        self.return_prob = return_prob
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > self.return_prob:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


class CycleGAN(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        buffer_size: int = 50,
        buffer_return_prob: float = 0.5,
        disc_loss_avg_factor: float = 0.5,
    ):
        super().__init__()
        self.device = device
        self.disc_loss_avg_factor = disc_loss_avg_factor
        self.G_AB = ResNetGenerator().to(device)
        self.G_BA = ResNetGenerator().to(device)
        self.D_A = PatchGANDiscriminator().to(device)
        self.D_B = PatchGANDiscriminator().to(device)
        self.fake_A_buffer = ImageBuffer(max_size=buffer_size, return_prob=buffer_return_prob)
        self.fake_B_buffer = ImageBuffer(max_size=buffer_size, return_prob=buffer_return_prob)
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()

    def set_input(self, real_A: torch.Tensor, real_B: torch.Tensor) -> None:
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self) -> None:
        self.fake_B = self.G_AB(self.real_A)
        self.rec_A = self.G_BA(self.fake_B)
        self.fake_A = self.G_BA(self.real_B)
        self.rec_B = self.G_AB(self.fake_A)

    def compute_generator_loss(
        self,
        lambda_cyc: float = 10.0,
        lambda_gan: float = 1.0,
        lambda_idt: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # GAN loss (generators try to fool discriminators)
        pred_fake_B = self.D_B(self.fake_B)
        loss_G_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B)) * lambda_gan
        pred_fake_A = self.D_A(self.fake_A)
        loss_G_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A)) * lambda_gan

        # Cycle consistency loss
        loss_cyc_A = self.criterion_cycle(self.rec_A, self.real_A) * lambda_cyc
        loss_cyc_B = self.criterion_cycle(self.rec_B, self.real_B) * lambda_cyc

        total = loss_G_AB + loss_G_BA + loss_cyc_A + loss_cyc_B
        losses = {
            "G_AB": loss_G_AB,
            "G_BA": loss_G_BA,
            "cycle_A": loss_cyc_A,
            "cycle_B": loss_cyc_B,
        }

        # Identity loss (unmasked global regulariser; skipped when weight is zero).
        # Each line performs an extra forward pass through one generator.
        if lambda_idt > 0.0:
            loss_idt_A = self.criterion_cycle(self.G_BA(self.real_A), self.real_A) * lambda_idt
            loss_idt_B = self.criterion_cycle(self.G_AB(self.real_B), self.real_B) * lambda_idt
            total = total + loss_idt_A + loss_idt_B
            losses["idt_A"] = loss_idt_A
            losses["idt_B"] = loss_idt_B

        losses["total_G"] = total
        return losses

    def compute_discriminator_loss(self) -> dict[str, torch.Tensor]:
        # D_B
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B.detach())
        pred_real_B = self.D_B(self.real_B)
        loss_D_B_real = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.D_B(fake_B)
        loss_D_B_fake = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * self.disc_loss_avg_factor

        # D_A
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A.detach())
        pred_real_A = self.D_A(self.real_A)
        loss_D_A_real = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.D_A(fake_A)
        loss_D_A_fake = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * self.disc_loss_avg_factor

        return {"D_A": loss_D_A, "D_B": loss_D_B, "total_D": loss_D_A + loss_D_B}
