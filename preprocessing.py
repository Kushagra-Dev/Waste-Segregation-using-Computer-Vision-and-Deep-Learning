import os
import cv2
import torch
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ColorJitter
from albumentations.pytorch import ToTensorV2


INPUT_DIR = "dataset_split/train"  # Change to train, val or test folder to preprocess
OUTPUT_DIR = f"processed_{os.path.basename(INPUT_DIR)}"


def preprocess_image(image_path, size=(224, 224), train=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if train:
        transform = Compose([
            Resize(height=size[0], width=size[1]),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        transform = Compose([
            Resize(height=size[0], width=size[1]),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    augmented = transform(image=image)
    return augmented['image']


def preprocess_folder(input_dir, output_dir, train=False):
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
            tensor = preprocess_image(in_path, train=train)
            torch.save(tensor, out_path)

        print(f"Processed class folder: {class_name}")


if __name__ == "__main__":
    preprocess_folder(INPUT_DIR, OUTPUT_DIR, train=True)
    print(f"Done preprocessing {INPUT_DIR} -> {OUTPUT_DIR}")
