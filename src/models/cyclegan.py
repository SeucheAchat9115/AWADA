import random
import torch
import torch.nn as nn
from .generator import ResNetGenerator
from .discriminator import PatchGANDiscriminator


class ImageBuffer:
    """Replay buffer to stabilize discriminator training.
    Size of 50 matches the original CycleGAN implementation (Zhu et al., 2017).
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


class CycleGAN(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.G_AB = ResNetGenerator().to(device)
        self.G_BA = ResNetGenerator().to(device)
        self.D_A = PatchGANDiscriminator().to(device)
        self.D_B = PatchGANDiscriminator().to(device)
        self.fake_A_buffer = ImageBuffer()
        self.fake_B_buffer = ImageBuffer()
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        self.fake_B = self.G_AB(self.real_A)
        self.rec_A = self.G_BA(self.fake_B)
        self.fake_A = self.G_BA(self.real_B)
        self.rec_B = self.G_AB(self.fake_A)

    def compute_generator_loss(self, lambda_cyc=10.0, lambda_idt=5.0, lambda_gan=1.0):
        # Identity loss
        idt_A = self.G_BA(self.real_A)
        idt_B = self.G_AB(self.real_B)
        loss_idt_A = self.criterion_identity(idt_A, self.real_A) * lambda_idt
        loss_idt_B = self.criterion_identity(idt_B, self.real_B) * lambda_idt

        # GAN loss (generators try to fool discriminators)
        pred_fake_B = self.D_B(self.fake_B)
        loss_G_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B)) * lambda_gan
        pred_fake_A = self.D_A(self.fake_A)
        loss_G_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A)) * lambda_gan

        # Cycle consistency loss
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
        # D_B
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B.detach())
        pred_real_B = self.D_B(self.real_B)
        loss_D_B_real = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.D_B(fake_B)
        loss_D_B_fake = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        # D_A
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A.detach())
        pred_real_A = self.D_A(self.real_A)
        loss_D_A_real = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.D_A(fake_A)
        loss_D_A_fake = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        return {'D_A': loss_D_A, 'D_B': loss_D_B, 'total_D': loss_D_A + loss_D_B}
