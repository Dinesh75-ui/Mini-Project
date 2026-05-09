import os
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

def resize_image(args):
    src_path, dst_path, size = args
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((size, size), Image.BICUBIC)
            img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def optimize(src_root="data/coco_subset", dst_root="data/coco_128_subset", subset_size=30000, img_size=128):
    src_path = Path(src_root)
    if not src_path.exists():
        print(f"Error: Source directory not found at {src_path}")
        return

    os.makedirs(dst_root, exist_ok=True)
    
    all_imgs = [
        path for path in src_path.rglob("*")
        if path.suffix.lower() in {'.jpeg', '.jpg', '.png'}
    ]
    print(f"Found {len(all_imgs)} images. Optimizing...")
    sample = random.sample(all_imgs, min(len(all_imgs), subset_size))
    tasks = [(str(path), os.path.join(dst_root, path.name), img_size) for path in sample]

    print(f"Prepared {len(tasks)} tasks. Starting parallel processing on {cpu_count()} cores...")
    
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(resize_image, tasks), total=len(tasks)))
    
    success_count = sum(results)
    print(f"\nOptimization complete! Processed {success_count} images into {dst_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data/coco_subset")
    parser.add_argument("--dst", type=str, default="data/coco_128_subset")
    parser.add_argument("--size", type=int, default=30000)
    args = parser.parse_args()
    
    optimize(src_root=args.src, dst_root=args.dst, subset_size=args.size)
