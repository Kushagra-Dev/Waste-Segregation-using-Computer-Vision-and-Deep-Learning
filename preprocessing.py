import os
import cv2
import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


INPUT_DIR = "dataset_split/val"  # Change to val or test folder to preprocess
OUTPUT_DIR = f"processed_{os.path.basename(INPUT_DIR)}"


def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose([
        Resize(height=size[0], width=size[1]),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    augmented = transform(image=image)
    return augmented['image']


def preprocess_folder(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        class_name = os.path.basename(root)
        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        for file in image_files:
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_class_dir, file + ".pt")
            tensor = preprocess_image(in_path)
            torch.save(tensor, out_path)

        print(f"Processed class folder: {class_name}")


if __name__ == "__main__":
    preprocess_folder(INPUT_DIR, OUTPUT_DIR)
    print(f"Done preprocessing {INPUT_DIR} -> {OUTPUT_DIR}")
