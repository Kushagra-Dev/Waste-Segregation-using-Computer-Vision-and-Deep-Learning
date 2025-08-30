import os
import cv2
import torch
import numpy as np
import requests
from torchvision import transforms
from torch.autograd import Variable

model_path = "u2net.pth"

from u2net import U2NET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
net.to(device)
net.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_mask(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = net(img)

    pred = d1.squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize 0-1
    mask = (pred * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    return binary_mask

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, f)
                mask = get_mask(input_path)
                if mask is not None:
                    save_path = os.path.join(save_dir, os.path.splitext(f)[0] + '.png')
                    cv2.imwrite(save_path, mask)
                    print(f"Saved mask: {save_path}")

if __name__ == "__main__":
    input_images_dir = "dataset_split/train"
    masks_output_dir = "dataset_split/train_masks"

    process_folder(input_images_dir, masks_output_dir)
