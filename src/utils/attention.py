import os

import numpy as np
import torch
from tqdm import tqdm


def generate_attention_maps(detector, dataloader, output_dir, score_threshold=0.5, device="cuda"):
    """
    Run source domain images through Faster R-CNN, extract RPN proposals,
    create binary attention maps where pixels inside RPN boxes with objectness
    score >= score_threshold = 1, else 0.
    Save as .npy files (float32, 0 or 1, same H x W as input image).
    """
    os.makedirs(output_dir, exist_ok=True)
    detector.eval()
    detector.to(device)

    rpn_proposals = {}

    # Monkey-patch filter_proposals to capture both boxes and objectness scores
    original_filter_proposals = detector.rpn.filter_proposals

    def capturing_filter_proposals(*args, **kwargs):
        result = original_filter_proposals(*args, **kwargs)
        rpn_proposals["boxes"] = result[0]
        rpn_proposals["scores"] = result[1]
        return result

    detector.rpn.filter_proposals = capturing_filter_proposals

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating attention maps")):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch
                else:
                    images = batch
                    targets = None

                images = [img.to(device) for img in images]

                # Run forward pass to trigger the capturing wrapper
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
                    np.save(out_path, attention_map)
    finally:
        detector.rpn.filter_proposals = original_filter_proposals

    print(f"Saved {len(dataloader.dataset)} attention maps to {output_dir}")
