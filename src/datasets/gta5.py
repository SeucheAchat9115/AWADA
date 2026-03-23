import os
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# GTA5 label index -> detection class index mapping
GTA5_LABEL_MAP = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    32: 5,  # motorcycle
    33: 6,  # bicycle
}
CLASS_NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
# Minimum number of foreground pixels for a connected component to be kept as a detection box
MIN_PIXELS_THRESHOLD = 10


class GTA5Dataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.transforms = transforms
        img_dir = os.path.join(root, 'images')
        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        n = len(all_images)
        if split == 'train':
            self.image_files = all_images[:int(0.8 * n)]
        elif split == 'val':
            self.image_files = all_images[int(0.8 * n):]
        else:
            self.image_files = all_images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.root, 'images', fname)
        lbl_path = os.path.join(self.root, 'labels', fname)

        image = Image.open(img_path).convert('RGB')
        label = np.array(Image.open(lbl_path))

        boxes, labels = [], []
        for gta_id, det_label in GTA5_LABEL_MAP.items():
            mask = (label == gta_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            labeled, num_features = ndimage.label(mask)
            for comp_idx in range(1, num_features + 1):
                ys, xs = np.where(labeled == comp_idx)
                if len(ys) < MIN_PIXELS_THRESHOLD:
                    continue
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(det_label)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        image_t = TF.to_tensor(image)
        target = {'boxes': boxes_t, 'labels': labels_t, 'image_id': torch.tensor([idx])}
        if self.transforms:
            image_t, target = self.transforms(image_t, target)
        return image_t, target
