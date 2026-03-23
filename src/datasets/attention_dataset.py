import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class AttentionPairedDataset(Dataset):
    """
    Dataset for AWADA GAN training.
    Yields random 128x128 patches from source images + corresponding attention patches,
    plus random 128x128 patches from target images.
    """
    def __init__(self, source_root, target_root, attention_root, patch_size=128):
        self.patch_size = patch_size
        self.source_files = sorted([
            os.path.join(source_root, f) for f in os.listdir(source_root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.target_files = sorted([
            os.path.join(target_root, f) for f in os.listdir(target_root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.attention_root = attention_root

    def __len__(self):
        return len(self.source_files)

    def _load_attention(self, src_path):
        stem = os.path.splitext(os.path.basename(src_path))[0]
        npy_path = os.path.join(self.attention_root, stem + '.npy')
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(np.float32)
        # If no attention map found, return None (will use all-ones later)
        return None

    def __getitem__(self, idx):
        src_path = self.source_files[idx]
        tgt_path = random.choice(self.target_files)
        p = self.patch_size

        src_img = Image.open(src_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        src_w, src_h = src_img.size
        tgt_w, tgt_h = tgt_img.size

        # Random crop source + attention at same location
        src_x = random.randint(0, max(0, src_w - p))
        src_y = random.randint(0, max(0, src_h - p))
        src_patch = TF.to_tensor(src_img.crop((src_x, src_y, src_x + p, src_y + p)))

        attention = self._load_attention(src_path)
        if attention is not None:
            att_patch = attention[src_y:src_y + p, src_x:src_x + p]
            # Pad if near border
            pad_h = max(0, p - att_patch.shape[0])
            pad_w = max(0, p - att_patch.shape[1])
            if pad_h > 0 or pad_w > 0:
                att_patch = np.pad(att_patch, ((0, pad_h), (0, pad_w)), mode='constant')
            att_patch = torch.from_numpy(att_patch).unsqueeze(0)  # [1, p, p]
        else:
            att_patch = torch.ones(1, p, p)

        # Random crop target
        tgt_x = random.randint(0, max(0, tgt_w - p))
        tgt_y = random.randint(0, max(0, tgt_h - p))
        tgt_patch = TF.to_tensor(tgt_img.crop((tgt_x, tgt_y, tgt_x + p, tgt_y + p)))

        # Normalize to [-1, 1]
        src_patch = src_patch * 2 - 1
        tgt_patch = tgt_patch * 2 - 1

        return src_patch, tgt_patch, att_patch
