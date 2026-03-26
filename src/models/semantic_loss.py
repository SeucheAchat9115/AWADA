import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

# ImageNet normalisation constants used by DeepLabV3
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class SemanticConsistencyLoss(nn.Module):
    """Semantic consistency loss using a frozen DeepLabV3-ResNet50 (COCO) backbone.

    Encourages the generator to preserve semantic structure by minimising the L1
    distance between the segmentation logits of the translated image and those of
    the original.  The network weights are frozen — only the generator receives
    gradients from this loss.

    CycleGAN images are expected in the [-1, 1] range; they are re-normalised to
    ImageNet statistics before being passed to DeepLabV3.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        net = deeplabv3_resnet50(weights=weights)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device)
        self.device = device
        # Register normalisation tensors as buffers so they move with .to()
        self.register_buffer("_mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("_std", _IMAGENET_STD.view(1, 3, 1, 1))

    def _to_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Re-normalise a [-1, 1] CycleGAN tensor to ImageNet statistics."""
        x = (x + 1.0) / 2.0  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        return (x - self._mean) / self._std

    def forward(self, translated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Compute the semantic consistency loss.

        Args:
            translated: Generator output in [-1, 1], shape [B, 3, H, W].
            original:   Corresponding input image in [-1, 1], shape [B, 3, H, W].

        Returns:
            Scalar loss tensor.
        """
        # Target logits: computed from the (detached) original — no grad needed
        with torch.no_grad():
            logits_orig = self.net(self._to_imagenet(original.detach()))["out"]

        # Translated logits: gradients flow back to the generator through translated
        logits_trans = self.net(self._to_imagenet(translated))["out"]

        return F.l1_loss(logits_trans, logits_orig)
