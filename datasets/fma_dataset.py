import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class FMADataset(Dataset):
    """Dataset for paired blurred(sharp) images with masks.

    Expects folder structure under root like:
      train_blur/001/xxx.png
      train_sharp/001/xxx.png
      train_mask/001/xxx.png

    Returns dict with 'inp' (3xHxW tensor), 'gt' (3xHxW tensor), 'mask' (1xHxW tensor).
    Automatically handles grayscale images by converting to RGB (3 channels).
    """

    def __init__(self, root, split='train', subfolders=None, patch_size=128, augment=True, gray_mode='replicate'):
        super().__init__()
        self.root = root
        self.split = split
        self.augment = augment
        self.patch_size = patch_size
        # gray_mode: 'replicate' -> convert gray->RGB by replication (default)
        #            'single'    -> return single-channel tensors (1,H,W)
        self.gray_mode = gray_mode

        self.blur_dir = os.path.join(root, f"{split}_blur")
        self.sharp_dir = os.path.join(root, f"{split}_sharp")
        self.mask_dir = os.path.join(root, f"{split}_mask")

        if not os.path.isdir(self.blur_dir):
            raise RuntimeError(f"Blur dir not found: {self.blur_dir}")

        self.items = []
        for sub in sorted(os.listdir(self.blur_dir)):
            bsub = os.path.join(self.blur_dir, sub)
            ssub = os.path.join(self.sharp_dir, sub)
            msub = os.path.join(self.mask_dir, sub)
            if not os.path.isdir(bsub):
                continue
            # find fallback mask in msub (single mask image for the whole subfolder)
            fallback_mask = None
            if os.path.isdir(msub):
                for mf in sorted(os.listdir(msub)):
                    if mf.lower().endswith('.png') or mf.lower().endswith('.jpg') or mf.lower().endswith('.jpeg'):
                        fallback_mask = os.path.join(msub, mf)
                        break

            for fn in sorted(os.listdir(bsub)):
                fb = os.path.join(bsub, fn)
                fs = os.path.join(ssub, fn)
                fm = os.path.join(msub, fn)
                if os.path.isfile(fb) and os.path.isfile(fs):
                    if os.path.isfile(fm):
                        self.items.append((fb, fs, fm))
                    elif fallback_mask is not None:
                        # use the single mask image for all files in this subfolder
                        self.items.append((fb, fs, fallback_mask))
                    else:
                        # no mask available; append with None and handle later
                        self.items.append((fb, fs, None))

    def __len__(self):
        return len(self.items)

    def _load_image(self, path, mode='RGB'):
        # Use PIL to keep consistent behavior across platforms
        img = Image.open(path)
        if mode == 'RGB':
            img = img.convert('RGB')
        elif mode == 'L':
            img = img.convert('L')
        return img

    def _random_crop(self, img_list, size):
        w, h = img_list[0].size
        th, tw = size, size
        if w == tw and h == th:
            return img_list
        if w < tw or h < th:
            # pad if smaller
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            for i in range(len(img_list)):
                img_list[i] = TF.pad(img_list[i], (0, 0, pad_w, pad_h))
            w, h = img_list[0].size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in img_list]

    def __getitem__(self, idx):
        fb, fs, fm = self.items[idx]
        # load images; keep original channels for flexibility
        img_b = Image.open(fb)
        img_s = Image.open(fs)

        # determine mask source: fm may be a path to an image, a csv, or None
        if fm is None:
            mask = Image.new('L', img_b.size, 0)
        else:
            fm_lower = str(fm).lower()
            if fm_lower.endswith('.csv'):
                # build mask from csv coords
                mask_arr = np.zeros((img_b.size[1], img_b.size[0]), dtype=np.uint8)
                try:
                    with open(fm, 'r', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 'x' in row and 'y' in row:
                                try:
                                    x = int(float(row['x']))
                                    y = int(float(row['y']))
                                except Exception:
                                    continue
                                if 0 <= x < img_b.size[0] and 0 <= y < img_b.size[1]:
                                    mask_arr[y, x] = 255
                except Exception:
                    mask_arr = np.zeros((img_b.size[1], img_b.size[0]), dtype=np.uint8)
                mask = Image.fromarray(mask_arr, mode='L')
            else:
                # fm is likely an image path
                try:
                    mask = self._load_image(fm, mode='L')
                except Exception:
                    mask = Image.new('L', img_b.size, 0)

        # handle grayscale datasets if requested
        if self.gray_mode == 'single':
            img_b = img_b.convert('L')
            img_s = img_s.convert('L')
        else:
            img_b = img_b.convert('RGB')
            img_s = img_s.convert('RGB')

        if self.patch_size is not None:
            img_b, img_s, mask = self._random_crop([img_b, img_s, mask], self.patch_size)

        # data augmentation
        if self.augment:
            if random.random() < 0.5:
                img_b = TF.hflip(img_b)
                img_s = TF.hflip(img_s)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img_b = TF.vflip(img_b)
                img_s = TF.vflip(img_s)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                img_b = img_b.rotate(90, expand=True)
                img_s = img_s.rotate(90, expand=True)
                mask = mask.rotate(90, expand=True)

        tb = TF.to_tensor(img_b)  # C,H,W in [0,1]
        ts = TF.to_tensor(img_s)
        tm = TF.to_tensor(mask)  # 1,H,W
        # normalize mask to 0/1 if needed
        tm = (tm > 0.5).float()

        # If single-channel mode, ensure gt is 1xHxW and inp is 1xHxW
        return {'inp': tb, 'gt': ts, 'mask': tm}
