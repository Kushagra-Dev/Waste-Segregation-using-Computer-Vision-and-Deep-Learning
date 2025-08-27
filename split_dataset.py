import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "/Users/kushagra/Downloads/garbage-dataset"
OUTPUT_DIR = "dataset_split"

SPLIT_RATIOS = (0.7, 0.15, 0.15)

def split_and_copy():
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for cls in classes:
        class_dir = os.path.join(DATASET_DIR, cls)
        images = [os.path.join(class_dir, f)
                  for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        train_val_ratio = SPLIT_RATIOS[0] + SPLIT_RATIOS[1] 
        
        train_val_imgs, test_imgs = train_test_split(
            images, test_size=SPLIT_RATIOS[2], random_state=42)
        
        val_ratio_adjusted = SPLIT_RATIOS[1] / train_val_ratio  
        
        train_imgs, val_imgs = train_test_split(
            train_val_imgs, test_size=val_ratio_adjusted, random_state=42)
        
        for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            out_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for src_path in split_imgs:
                dest_path = os.path.join(out_dir, os.path.basename(src_path))
                shutil.copy(src_path, dest_path)
        
        print(f"Class {cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

if __name__ == "__main__":
    split_and_copy()
