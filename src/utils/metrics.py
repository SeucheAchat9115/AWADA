import torch
from typing import List, Dict


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p = max([p for r, p in zip(recalls, precisions) if r >= t], default=0.0)
        ap += p / 11.0
    return ap


def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    num_classes: int = None,
) -> Dict[str, float]:
    """
    Compute mAP@iou_threshold for object detection.

    Args:
        predictions: list of dicts with 'boxes' [N,4], 'labels' [N], 'scores' [N]
        targets: list of dicts with 'boxes' [M,4], 'labels' [M]
        iou_threshold: IoU threshold for a true positive
        num_classes: number of classes; inferred from data if None

    Returns:
        dict with 'mAP' and per-class APs
    """
    if num_classes is None:
        all_labels = []
        for t in targets:
            if len(t['labels']) > 0:
                all_labels.extend(t['labels'].tolist())
        num_classes = max(all_labels) + 1 if all_labels else 1

    # Gather per-class detections and ground truths
    class_detections = {c: [] for c in range(num_classes)}
    class_gt_counts = {c: 0 for c in range(num_classes)}

    for pred, target in zip(predictions, targets):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        pred_boxes = pred.get('boxes', torch.zeros(0, 4))
        pred_labels = pred.get('labels', torch.zeros(0, dtype=torch.long))
        pred_scores = pred.get('scores', torch.zeros(0))

        # Count ground truths per class
        for c in range(num_classes):
            class_gt_counts[c] += (gt_labels == c).sum().item()

        matched_gt = set()
        # Sort predictions by score descending
        if len(pred_scores) > 0:
            order = pred_scores.argsort(descending=True)
        else:
            order = torch.arange(0)

        for i in order:
            c = pred_labels[i].item()
            score = pred_scores[i].item()
            box = pred_boxes[i]

            # Find best matching GT box
            best_iou = iou_threshold - 1e-6
            best_j = -1
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label.item() != c or j in matched_gt:
                    continue
                iou = compute_iou(box.tolist(), gt_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                matched_gt.add(best_j)
                class_detections[c].append((score, 1))  # TP
            else:
                class_detections[c].append((score, 0))  # FP

    aps = {}
    for c in range(num_classes):
        dets = sorted(class_detections[c], key=lambda x: -x[0])
        n_gt = class_gt_counts[c]
        if n_gt == 0:
            aps[c] = 0.0
            continue
        tp_cumsum = 0
        fp_cumsum = 0
        recalls, precisions = [], []
        for score, is_tp in dets:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            recalls.append(tp_cumsum / n_gt)
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))

        aps[c] = compute_ap(recalls, precisions)

    mean_ap = sum(aps.values()) / num_classes if num_classes > 0 else 0.0

    result = {'mAP': mean_ap}
    for c, ap in aps.items():
        result[f'AP_class_{c}'] = ap
    return result


def compute_map_range(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int = None,
) -> Dict[str, float]:
    """Compute mAP@0.5 and mAP@0.5:0.95."""
    map50 = compute_map(predictions, targets, iou_threshold=0.5, num_classes=num_classes)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    maps = [
        compute_map(predictions, targets, iou_threshold=t, num_classes=num_classes)['mAP']
        for t in thresholds
    ]
    return {
        'mAP@0.5': map50['mAP'],
        'mAP@0.5:0.95': sum(maps) / len(maps),
        **{f'AP_class_{c}': v for c, v in map50.items() if isinstance(c, int)},
    }
