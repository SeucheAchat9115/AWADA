import contextlib
import io
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_coco_format(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> tuple[dict, list]:
    """Convert predictions and targets to COCO-compatible structures.

    Args:
        predictions: List of prediction dicts, each with ``"boxes"`` ``[N, 4]``,
            ``"labels"`` ``[N]``, and ``"scores"`` ``[N]``.
        targets: List of ground-truth dicts, each with ``"boxes"`` ``[M, 4]``
            and ``"labels"`` ``[M]``.
        num_classes: Number of foreground classes (used to build the category list).

    Returns:
        Tuple of ``(gt_dataset, results)`` where ``gt_dataset`` is a COCO-format
        ground-truth dict and ``results`` is a list of COCO-format detection dicts.
    """
    images = []
    annotations = []
    results = []
    ann_id = 1

    # Build category list (1-indexed to match Faster R-CNN label convention)
    categories = [{"id": c, "name": str(c)} for c in range(1, num_classes + 1)]

    for img_id, (pred, target) in enumerate(zip(predictions, targets)):
        images.append({"id": img_id})

        gt_boxes = target.get("boxes", torch.zeros(0, 4))
        gt_labels = target.get("labels", torch.zeros(0, dtype=torch.long))

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            cat = int(label.item())
            if cat < 1 or cat > num_classes:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        pred_boxes = pred.get("boxes", torch.zeros(0, 4))
        pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
        pred_scores = pred.get("scores", torch.zeros(0))

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            cat = int(label.item())
            if cat < 1 or cat > num_classes:
                continue
            results.append(
                {
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x1, y1, w, h],
                    "score": float(score.item()),
                }
            )

    gt_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return gt_dataset, results


def compute_map_range(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute mAP@0.5 and mAP@0.5:0.95 using pycocotools.

    Args:
        predictions: list of dicts with 'boxes' [N,4], 'labels' [N], 'scores' [N]
        targets: list of dicts with 'boxes' [M,4], 'labels' [M]
        num_classes: number of foreground classes; inferred from data if None

    Returns:
        dict with 'mAP@0.5', 'mAP@0.5:0.95', and 'per_class_AP' (a dict mapping
        category id to AP@0.5:0.95 for that category).
    """
    if num_classes is None:
        all_labels = []
        for t in targets:
            if len(t.get("labels", [])) > 0:
                all_labels.extend(t["labels"].tolist())
        num_classes = int(max(all_labels)) if all_labels else 1

    gt_dataset, results = _to_coco_format(predictions, targets, num_classes)

    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    if len(results) == 0:
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "per_class_AP": {}}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[0] = mAP @ 0.50:0.95, stats[1] = mAP @ 0.50
    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    # Extract per-category AP@0.5:0.95 from precision array.
    # precision has shape [T, R, K, A, M]:
    #   T = IoU thresholds, R = recall thresholds, K = categories,
    #   A = area ranges, M = max detections.
    # Use A=0 (all areas) and M=2 (corresponding to maxDets=100).
    per_class_AP: Dict[int, float] = {}
    if coco_eval.eval and "precision" in coco_eval.eval:
        precision = coco_eval.eval["precision"]  # numpy array [T, R, K, A, M]
        for k, cat_id in enumerate(coco_eval.params.catIds):
            prec = precision[:, :, k, 0, 2]
            valid = prec[prec > -1]
            per_class_AP[int(cat_id)] = float(valid.mean()) if len(valid) > 0 else 0.0

    return {"mAP@0.5": map_50, "mAP@0.5:0.95": map_50_95, "per_class_AP": per_class_AP}
