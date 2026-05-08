import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab

def build_transform(split, size, is_optimized=False):
    """Return appropriate transforms based on split and optimization state."""
    # Lightness Augmentation: Helps the model handle shadows and bright spots in videos.
    lightness_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    
    if is_optimized:
        ops = [transforms.RandomHorizontalFlip(), lightness_aug] if split == 'train' else []
        return transforms.Compose(ops)
    
    ops = [transforms.Resize((size, size), Image.BICUBIC)]
    if split == 'train':
        ops += [
            transforms.RandomHorizontalFlip(),
            lightness_aug,
            transforms.RandomCrop(size, padding=4),
            transforms.Resize((size, size), Image.BICUBIC),
        ]
    return transforms.Compose(ops)

def pil_to_lab_tensors(pil_img, transform=None):
    """Convert PIL image to normalized L and ab tensors."""
    img = pil_img.convert("RGB")
    if transform:
        img = transform(img)
    
    img_np = np.array(img).astype("float32")
    if img_np.max() > 1.0: # If uint8, normalize to [0, 1] for skimage
        img_np /= 255.0
        
    img_lab = rgb2lab(img_np).astype("float32")
    img_lab = transforms.ToTensor()(img_lab) # (3, H, W)
    
    L  = img_lab[[0], ...] / 50. - 1.   # Normalize L  → [-1, 1]
    ab = img_lab[[1, 2], ...] / 110.    # Normalize ab → [-1, 1]
    return {'L': L, 'ab': ab}

class ColorizationDataset(Dataset):
    """
    A universal dataset class for all colorization tasks.
    Supports Tiny ImageNet, COCO, and Custom folders.
    """
    def __init__(self, root_dir=None, paths=None, split='train', size=128, is_optimized=False):
        self.split = split
        self.size = size
        self.is_optimized = is_optimized
        
        if paths:
            self.paths = paths
        elif root_dir:
            if os.path.isdir(os.path.join(root_dir, "train")): # Tiny ImageNet structure
                self.paths = []
                train_path = os.path.join(root_dir, "train")
                for cls in os.listdir(train_path):
                    img_dir = os.path.join(train_path, cls, "images")
                    if os.path.isdir(img_dir):
                        self.paths += [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
            else: # Flat folder (COCO or Optimized)
                self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.transform = build_transform(split, size, is_optimized)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx])
            return pil_to_lab_tensors(img, self.transform)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.paths)-1))

def get_image_paths(dir_path):
    """Recursively get all valid image file paths."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                paths.append(os.path.join(root, file))
    return paths
