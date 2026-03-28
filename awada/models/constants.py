"""Named constants for CycleGAN model configuration.

Centralizes magic numbers used across the model and training scripts so that
their intent is explicit and they can be changed in a single place.
"""

# ---------------------------------------------------------------------------
# Replay buffer (ImageBuffer)
# ---------------------------------------------------------------------------

IMAGE_BUFFER_SIZE: int = 50
"""Size of the image replay buffer.  Matches the original CycleGAN paper (Zhu et al., 2017)."""

BUFFER_RETURN_PROBABILITY: float = 0.5
"""Probability of returning a stored (historical) image rather than the newly generated one."""

# ---------------------------------------------------------------------------
# Discriminator loss
# ---------------------------------------------------------------------------

DISCRIMINATOR_LOSS_AVERAGING_FACTOR: float = 0.5
"""Factor used to average the real and fake discriminator losses."""

# ---------------------------------------------------------------------------
# Training logging
# ---------------------------------------------------------------------------

LOG_INTERVAL: int = 100
"""Number of iterations between loss log lines during training."""
