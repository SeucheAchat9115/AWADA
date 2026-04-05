from typing import Optional

import torch

from awada.config import DEFAULT_DEVICE

from .cyclegan import CycleGAN
from .semantic_loss import SemanticConsistencyLoss


class CyCada(CycleGAN):
    """CyCada: CycleGAN extended with a semantic consistency loss.

    Adds an unmasked, global semantic regularisation term on top of standard
    CycleGAN losses.  A frozen DeepLabV3 backbone is used to compare
    segmentation logits of the translated and original images, encouraging the
    generator to preserve semantic structure across domains.

    References:
        Hoffman et al., "CyCADA: Cycle-Consistent Adversarial Domain
        Adaptation", ICML 2018.
    """

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        lambda_sem: float = 0.0,
        buffer_size: int = 50,
        buffer_return_prob: float = 0.5,
        disc_loss_avg_factor: float = 0.5,
    ) -> None:
        """Initialise CyCada.

        Args:
            device: Torch device string.
            lambda_sem: Weight for the semantic consistency loss.  Pass ``0``
                (default) to skip instantiating the DeepLabV3 backbone.
            buffer_size: Capacity of each fake-image replay buffer.
            buffer_return_prob: Replay probability for the image buffers.
            disc_loss_avg_factor: Averaging factor for discriminator loss.
        """
        super().__init__(
            device=device,
            buffer_size=buffer_size,
            buffer_return_prob=buffer_return_prob,
            disc_loss_avg_factor=disc_loss_avg_factor,
        )
        # Semantic consistency loss: only instantiated when weight is non-zero
        self.criterion_sem: Optional[SemanticConsistencyLoss] = (
            SemanticConsistencyLoss(device=device) if lambda_sem > 0.0 else None
        )

    def compute_generator_loss(
        self,
        lambda_cyc: float = 10.0,
        lambda_gan: float = 1.0,
        lambda_idt: float = 0.0,
        lambda_sem: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined generator loss including the semantic term.

        Args:
            lambda_cyc: Weight for the cycle-consistency loss.
            lambda_gan: Weight for the adversarial GAN loss.
            lambda_idt: Identity-loss ratio relative to the cycle loss (0
                disables it).  Effective weight = ``lambda_idt × lambda_cyc``;
                the canonical value of 0.5 gives 0.5 × 10 = 5.
            lambda_sem: Weight for the semantic consistency loss (0 disables it).

        Returns:
            Dictionary with individual loss components and ``"total_G"``.
        """
        losses = super().compute_generator_loss(lambda_cyc, lambda_gan, lambda_idt)
        total = losses.pop("total_G")

        # Semantic consistency loss (unmasked global regulariser; skipped when weight is zero)
        if lambda_sem > 0.0 and self.criterion_sem is not None:
            loss_sem_AB = self.criterion_sem(self.fake_B, self.real_A) * lambda_sem
            loss_sem_BA = self.criterion_sem(self.fake_A, self.real_B) * lambda_sem
            total = total + loss_sem_AB + loss_sem_BA
            losses["sem_AB"] = loss_sem_AB
            losses["sem_BA"] = loss_sem_BA

        losses["total_G"] = total
        return losses
