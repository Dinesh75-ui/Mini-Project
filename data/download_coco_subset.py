import os
import requests
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_image(args):
    img_id, dst_dir = args
    # COCO 2017 images are 12 digits, zero-padded
    img_name = f"{str(img_id).zfill(12)}.jpg"
    url = f"http://images.cocodataset.org/train2017/{img_name}"
    dst_path = os.path.join(dst_dir, img_name)
    
    if os.path.exists(dst_path):
        return True
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(dst_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def download_coco_subset(count=100000, dst_dir="data/coco_subset"):
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Preparing to download {count} images from COCO 2017...")
    
    # We'll try a range of IDs. COCO IDs are not perfectly sequential but 
    # many exist in the lower ranges.
    # We'll generate a pool of potential IDs to check.
    potential_ids = list(range(1, 581930))
    random.seed(42)
    random.shuffle(potential_ids)
    
    selected_ids = potential_ids[:count*2] # Get extra in case some links are 404
    tasks = [(id, dst_dir) for id in selected_ids]
    
    success_count = 0
    with tqdm(total=count, desc="Downloading COCO") as pbar:
        with ThreadPoolExecutor(max_workers=30) as executor:
            for result in executor.map(download_image, tasks):
                if result:
                    success_count += 1
                    pbar.update(1)
                if success_count >= count:
                    break
                    
    print(f"\nDownload complete! {success_count} images saved to {dst_dir}")

if __name__ == "__main__":
    download_coco_subset()
