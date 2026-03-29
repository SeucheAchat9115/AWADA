import logging
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_attention_maps(
    detector: Any,
    dataloader: Any,
    output_dir: str,
    score_threshold: float = 0.5,
    device: str = "cuda",
) -> None:
    """Run source-domain images through Faster R-CNN and save binary attention maps.

    Registers a forward hook on the RPN module (``detector.rpn``) to capture
    box proposals and their objectness scores.  For each image, pixels inside
    RPN boxes whose objectness score meets ``score_threshold`` are set to
    ``1``; all other pixels are ``0``.  Maps are saved as uint8 ``.npy``
    files with the same ``H × W`` spatial dimensions as the input image.

    Args:
        detector: A torchvision Faster R-CNN model (or compatible) whose
            ``rpn`` attribute is a ``torch.nn.Module``.  The RPN's forward
            pass must return a tuple whose first element is a list of
            ``[N, 4]`` box tensors (one per image) and whose second element
            is either a list of ``[N]`` score tensors or an empty dict (as
            returned by the stock torchvision RPN in eval mode; in that case
            all captured proposals are treated as having score ``1.0``).
        dataloader: DataLoader yielding batches of images (or ``(images, targets)``
            tuples).  When targets are present they must be dicts with an
            optional ``"image_id"`` key used to name the output file.
        output_dir: Directory where ``.npy`` attention maps are written.
        score_threshold: Minimum objectness score for a proposal box to
            contribute to the attention map (default: ``0.5``).
        device: Torch device string used to move images and the model
            (default: ``"cuda"``).
    """
    os.makedirs(output_dir, exist_ok=True)
    detector.eval()
    detector.to(device)

    rpn_proposals: dict[str, Any] = {}

    def rpn_hook(module: Any, input: Any, output: Any) -> None:
        """Forward hook that records boxes and scores from the RPN module."""
        boxes = output[0]
        raw_scores = output[1]
        # torchvision RPN returns (boxes, losses_dict) in eval mode; models
        # that expose objectness scores return (boxes, scores_list) instead.
        if isinstance(raw_scores, list):
            scores: list[torch.Tensor] = raw_scores
        else:
            scores = [torch.ones(len(b), device=b.device) for b in boxes]
        rpn_proposals["boxes"] = boxes
        rpn_proposals["scores"] = scores

    hook_handle = detector.rpn.register_forward_hook(rpn_hook)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating attention maps")):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch
                else:
                    images = batch
                    targets = None

                images = [img.to(device) for img in images]

                # Run forward pass to trigger the RPN hook
                _ = detector(images)

                boxes_list = rpn_proposals.get("boxes", [])
                scores_list = rpn_proposals.get("scores", [])

                for i, (img, boxes, scores) in enumerate(zip(images, boxes_list, scores_list)):
                    h, w = img.shape[1], img.shape[2]
                    attention_map = np.zeros((h, w), dtype=np.float32)

                    boxes_np = boxes.cpu().numpy()
                    scores_np = scores.cpu().numpy()

                    # Keep only proposals whose objectness score meets the threshold
                    keep = scores_np >= score_threshold
                    boxes_np = boxes_np[keep]

                    for box in boxes_np:
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        attention_map[y1:y2, x1:x2] = 1.0

                    # Determine output filename
                    if targets is not None and isinstance(targets, (list, tuple)):
                        t = targets[i]
                        if "image_id" in t:
                            img_id = int(t["image_id"].item())
                        else:
                            img_id = batch_idx * len(images) + i
                    else:
                        img_id = batch_idx * len(images) + i

                    out_path = os.path.join(output_dir, f"{img_id:06d}.npy")
                    np.save(out_path, attention_map.astype(np.uint8))
    finally:
        hook_handle.remove()

    logger.info("Saved %d attention maps to %s", len(dataloader.dataset), output_dir)
