import os
import random
from PIL import Image
from tqdm import tqdm
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

def optimize(src_root="data/tiny-imagenet-200", dst_root="data/optimized_128_subset", subset_size=30000, img_size=128, is_coco=False):
    if is_coco:
        # COCO subset is just a flat folder of images
        train_dir = src_root
    else:
        # Tiny ImageNet has class subfolders
        train_dir = os.path.join(src_root, "train")
    
    if not os.path.exists(train_dir):
        print(f"Error: Source directory not found at {train_dir}")
        return

    os.makedirs(dst_root, exist_ok=True)
    
    tasks = []
    
    if not is_coco:
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        imgs_per_class = subset_size // len(classes)
        print(f"Found {len(classes)} classes. Selecting ~{imgs_per_class} images per class...")
        
        for cls in classes:
            cls_dir = os.path.join(train_dir, cls, "images")
            all_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            sample = random.sample(all_imgs, min(len(all_imgs), imgs_per_class))
            for img_name in sample:
                tasks.append((os.path.join(cls_dir, img_name), os.path.join(dst_root, f"{cls}_{img_name}"), img_size))
    else:
        # Flat folder for COCO
        all_imgs = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        print(f"Found {len(all_imgs)} images in COCO subset. Optimizing...")
        sample = random.sample(all_imgs, min(len(all_imgs), subset_size))
        for img_name in sample:
            tasks.append((os.path.join(train_dir, img_name), os.path.join(dst_root, img_name), img_size))

    print(f"Prepared {len(tasks)} tasks. Starting parallel processing on {cpu_count()} cores...")
    
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(resize_image, tasks), total=len(tasks)))
    
    success_count = sum(results)
    print(f"\nOptimization complete! Processed {success_count} images into {dst_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data/tiny-imagenet-200")
    parser.add_argument("--dst", type=str, default="data/tiny_imagenet_128_subset")
    parser.add_argument("--coco", action="store_true", help="Use COCO flat folder structure")
    parser.add_argument("--size", type=int, default=30000)
    args = parser.parse_args()
    
    optimize(src_root=args.src, dst_root=args.dst, subset_size=args.size, is_coco=args.coco)
