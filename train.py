import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ColorJitter
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class WasteDataset(Dataset):
    def __init__(self, image_root, train=False, size=(224, 224)):
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

        if train:
            self.transform = Compose([
                Resize(height=size[0], width=size[1]),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        tensor = augmented['image']
        return tensor, label


def train(model, data_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if batch_idx % 10 == 0:
            writer.add_scalar('training_loss', avg_loss, epoch * len(data_loader) + batch_idx)

    return running_loss / len(data_loader)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = WasteDataset("dataset_split/train", train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    val_dataset = WasteDataset("dataset_split/val", train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir="runs/waste_classification_experiment")

    epochs = 5
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Training Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "model_resnet50.pth")
    print("Model saved to model_resnet50.pth")

    writer.close()


if __name__ == "__main__":
    main()
