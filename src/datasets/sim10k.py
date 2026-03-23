import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


CLASS_NAMES = ['car']


class Sim10kDataset(Dataset):
    """Driving in the Matrix (sim10k) synthetic driving dataset for object detection.

    Annotations are in PASCAL VOC XML format and contain only the 'car' class.
    Expected directory structure::

        root/
        ├── images/       # JPEG images (e.g. 00001.jpg)
        └── Annotations/  # PASCAL VOC XML files (e.g. 00001.xml)
    """

    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.transforms = transforms
        img_dir = os.path.join(root, 'images')
        all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
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
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(self.root, 'images', fname)
        ann_path = os.path.join(self.root, 'Annotations', stem + '.xml')

        image = Image.open(img_path).convert('RGB')

        boxes, labels = self._parse_annotation(ann_path)

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

    @staticmethod
    def _parse_annotation(ann_path):
        """Parse a PASCAL VOC XML annotation file and return boxes and labels."""
        boxes, labels = [], []
        if not os.path.exists(ann_path):
            return boxes, labels
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name != 'car':
                continue
            bndbox = obj.find('bndbox')
            x1 = float(bndbox.find('xmin').text)
            y1 = float(bndbox.find('ymin').text)
            x2 = float(bndbox.find('xmax').text)
            y2 = float(bndbox.find('ymax').text)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_NAMES.index('car'))
        return boxes, labels
