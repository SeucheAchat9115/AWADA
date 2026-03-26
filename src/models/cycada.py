from typing import Optional

import torch

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

    def __init__(self, device="cuda", lambda_sem: float = 0.0):
        super().__init__(device=device)
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
    ):
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
