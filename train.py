import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ColorJitter,
    MotionBlur, GaussianBlur, GaussNoise, CoarseDropout, Affine, Perspective
)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR


class ResNetSwinHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

        swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.swin_features = nn.Sequential(*list(swin.children())[:-1])

        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, x):
        resnet_feat = self.resnet_features(x).flatten(1)
        swin_feat = self.swin_features(x).squeeze(-1).squeeze(-1)
        combined = torch.cat([resnet_feat, swin_feat], dim=1)
        out = self.classifier(combined)
        return out


class WasteDataset(Dataset):
    def __init__(self, image_root, mask_root=None, bg_root=None, train=False, size=(224, 224)):
        self.samples = []
        self.class_to_idx = {}
        classes = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(image_root, cls)
            for f in os.listdir(cls_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, f), idx))

        self.train = train
        self.size = size
        self.mask_root = mask_root
        self.bg_images = []
        if bg_root and os.path.isdir(bg_root):
            for root, _, files in os.walk(bg_root):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        bg_path = os.path.join(root, f)
                        bg_img = cv2.imread(bg_path)
                        if bg_img is not None:
                            self.bg_images.append(bg_img)

        if train:
            self.transform = Compose([
                Resize(height=size[0], width=size[1]),
                Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), shear=(-15, 15), p=0.5),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Perspective(scale=(0.05, 0.1), p=0.3),
                MotionBlur(blur_limit=5, p=0.2),
                GaussianBlur(blur_limit=5, p=0.2),
                GaussNoise(p=0.2),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                CoarseDropout(p=0.3),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(height=size[0], width=size[1]),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def replace_background(self, image, mask):
        if not self.bg_images or mask is None:
            return image

        bg = random.choice(self.bg_images)
        bg = cv2.resize(bg, (image.shape[1], image.shape[0]))

        if mask.dtype != np.float32:
            mask = mask.astype(np.float32) / 255.0

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        composite = image.astype(np.float32) * mask + bg.astype(np.float32) * (1 - mask)
        composite = composite.astype(np.uint8)
        return composite

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train and self.mask_root and self.bg_images:
            mask_path = path.replace(os.path.basename(os.path.dirname(path)),
                                     os.path.basename(self.mask_root.rstrip(os.sep)))
            mask_path = os.path.splitext(mask_path)[0] + '.png'
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    image = self.replace_background(image, mask)

        augmented = self.transform(image=image)
        tensor = augmented['image']
        return tensor, label


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))

    return new_data, targets, shuffled_targets, lam


def remix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)

    batch_size, _, H, W = data.size()

    mask = (torch.rand(batch_size, 1, H, W) < lam).float().to(data.device)

    new_data = data * mask + shuffled_data * (1 - mask)
    lam = mask.mean().item()

    return new_data, targets, shuffled_targets, lam


def train(model, data_loader, criterion, optimizer, device, epoch, writer, cutmix_prob=0.3, remix_prob=0.1):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        r = random.random()
        if r < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif r < cutmix_prob + remix_prob:
            inputs, targets_a, targets_b, lam = remix(inputs, targets)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if batch_idx % 10 == 0:
            writer.add_scalar('training_loss', avg_loss, epoch * len(data_loader) + batch_idx)

    return running_loss / len(data_loader)


def validate(model, data_loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    print(f"Validation loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('val_accuracy', accuracy, epoch)
    return avg_loss, accuracy


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = WasteDataset(
        image_root="dataset_split/train",
        mask_root="dataset_split/train_masks",
        bg_root="/Users/kushagra/Downloads/indoor + outdoor (cear + hazy) bg dataset",
        train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)

    val_dataset = WasteDataset(
        image_root="dataset_split/val",
        train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

    num_classes = len(train_dataset.class_to_idx)
    model = ResNetSwinHybrid(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir="runs/waste_classification_hybrid_experiment")

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch + 1}/{epochs} — Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val accuracy: {val_acc * 100:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("Saved best hybrid model")

    writer.close()


if __name__ == "__main__":
    main()
