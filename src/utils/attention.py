import os
import numpy as np
import torch
from tqdm import tqdm


def generate_attention_maps(detector, dataloader, output_dir, top_k=10, device='cuda'):
    """
    Run source domain images through Faster R-CNN, extract RPN proposals,
    create binary attention maps where pixels inside top-k RPN boxes = 1, else 0.
    Save as .npy files (float32, 0 or 1, same H x W as input image).
    """
    os.makedirs(output_dir, exist_ok=True)
    detector.eval()
    detector.to(device)

    # Hook to capture RPN proposals
    rpn_proposals = {}

    def rpn_hook(module, input, output):
        # output from RPN: (boxes, scores) per image
        # In torchvision FasterRCNN, RPN forward returns (proposals, losses)
        # proposals is a list of tensors [N, 4] in (x1, y1, x2, y2) format
        rpn_proposals['boxes'] = output[0]  # list of [N, 4] tensors

    # Register hook on the RPN
    hook = detector.rpn.register_forward_hook(rpn_hook)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Generating attention maps')):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                images = batch
                targets = None

            if isinstance(images, torch.Tensor):
                images = [img.to(device) for img in images]
            else:
                images = [img.to(device) for img in images]

            # Run forward pass to trigger RPN hook
            _ = detector(images)

            boxes_list = rpn_proposals.get('boxes', [])

            for i, (img, boxes) in enumerate(zip(images, boxes_list)):
                h, w = img.shape[1], img.shape[2]
                attention_map = np.zeros((h, w), dtype=np.float32)

                # Sort by score not available here; just take top_k boxes by area (heuristic)
                boxes_np = boxes.cpu().numpy()
                if len(boxes_np) > top_k:
                    # Use first top_k (RPN already sorts by objectness score internally)
                    boxes_np = boxes_np[:top_k]

                for box in boxes_np:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    attention_map[y1:y2, x1:x2] = 1.0

                # Determine output filename
                if targets is not None and isinstance(targets, (list, tuple)):
                    t = targets[i]
                    if 'image_id' in t:
                        img_id = int(t['image_id'].item())
                    else:
                        img_id = batch_idx * len(images) + i
                else:
                    img_id = batch_idx * len(images) + i

                out_path = os.path.join(output_dir, f'{img_id:06d}.npy')
                np.save(out_path, attention_map)

    hook.remove()
    print(f'Saved {len(dataloader.dataset)} attention maps to {output_dir}')
