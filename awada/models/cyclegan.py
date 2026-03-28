import random

import torch
import torch.nn as nn

from awada.config import DEFAULT_DEVICE

from .discriminator import PatchGANDiscriminator
from .generator import ResNetGenerator


class ImageBuffer:
    """Replay buffer to stabilize discriminator training.
    Size of 50 matches the original CycleGAN implementation (Zhu et al., 2017).
    """

    def __init__(self, max_size: int = 50, return_prob: float = 0.5) -> None:
        """Initialise the image replay buffer.

        Args:
            max_size: Maximum number of images stored in the buffer.
            return_prob: Probability of returning a stored image instead of the
                incoming one when the buffer is full.
        """
        self.max_size = max_size
        self.return_prob = return_prob
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        """Push new images into the buffer and return a mixed batch.

        For each image in the batch: if the buffer is not full the image is
        added and returned as-is; otherwise with probability ``return_prob``
        the incoming image is returned directly, and with probability
        ``1 - return_prob`` a randomly stored image is returned and the
        stored image is replaced by the incoming one.

        Args:
            data: Batch of images of shape ``[B, C, H, W]``.

        Returns:
            Mixed batch of the same shape ``[B, C, H, W]``.
        """
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
    """CycleGAN: unpaired image-to-image translation with cycle consistency.

    Trains two generators (A→B and B→A) and two discriminators (for domain A
    and B) jointly.  Cycle-consistency loss ensures round-trip reconstruction
    fidelity without paired training data.

    References:
        Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
        Adversarial Networks", ICCV 2017.

    Args:
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        buffer_size: Number of images stored in each replay buffer.
        buffer_return_prob: Probability of returning a stored image from the
            replay buffer (see :class:`ImageBuffer`).
        disc_loss_avg_factor: Scalar applied to the sum of real and fake
            discriminator losses (typically 0.5).
    """

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        buffer_size: int = 50,
        buffer_return_prob: float = 0.5,
        disc_loss_avg_factor: float = 0.5,
    ) -> None:
        """Initialise CycleGAN with two generators and two discriminators.

        Args:
            device: Torch device string.
            buffer_size: Capacity of each fake-image replay buffer.
            buffer_return_prob: Replay probability for the image buffers.
            disc_loss_avg_factor: Averaging factor for discriminator loss.
        """
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
        """Store and transfer input tensors to the configured device.

        Args:
            real_A: Batch of source-domain images, shape ``[B, C, H, W]``.
            real_B: Batch of target-domain images, shape ``[B, C, H, W]``.
        """
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self) -> None:
        """Perform the full cycle-translation forward pass.

        Computes ``fake_B = G_AB(real_A)``, ``rec_A = G_BA(fake_B)``,
        ``fake_A = G_BA(real_B)``, and ``rec_B = G_AB(fake_A)``.
        Results are stored as instance attributes for use by the loss methods.
        """
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
        """Compute the combined generator loss.

        Args:
            lambda_cyc: Weight for the cycle-consistency loss.
            lambda_gan: Weight for the adversarial GAN loss.
            lambda_idt: Weight for the identity loss (0 disables it).

        Returns:
            Dictionary with individual loss components and ``"total_G"``.
        """
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
        """Compute the combined discriminator loss for both domains.

        Uses the replay buffers to stabilise training.  Losses are averaged
        over real and fake predictions using ``disc_loss_avg_factor``.

        Returns:
            Dictionary with ``"D_A"``, ``"D_B"``, and ``"total_D"`` losses.
        """
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
