"""End-to-end smoke test for the AWADA training loop."""

import torch

from awada.models.awada import AWADA

DEVICE = "cpu"
IMG_SIZE = 64
BATCH_SIZE = 1


def test_training_loop_generator_loss_is_finite():
    """Two complete forward/backward iterations with synthetic data stay finite."""
    model = AWADA(device=DEVICE)

    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    g_loss_total = None

    for _ in range(2):
        real_A = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        real_B = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        att_A = torch.ones(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
        att_B = torch.ones(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

        model.set_input(real_A, real_B, attention_A=att_A, attention_B=att_B)
        model.forward()

        # Generator update
        for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
            p.requires_grad_(False)
        opt_G.zero_grad()
        g_losses = model.compute_generator_loss()
        g_losses["total_G"].backward()
        opt_G.step()
        for p in list(model.D_A.parameters()) + list(model.D_B.parameters()):
            p.requires_grad_(True)

        # Discriminator update
        opt_D.zero_grad()
        d_losses = model.compute_discriminator_loss()
        d_losses["total_D"].backward()
        opt_D.step()

        g_loss_total = g_losses["total_G"]
        d_loss_total = d_losses["total_D"]

    assert torch.isfinite(g_loss_total)
    assert torch.isfinite(d_loss_total)
