import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class WasteDataset(Dataset):
    def __init__(self, tensor_root):
        self.samples = []
        self.class_to_idx = {}
        classes = sorted(os.listdir(tensor_root))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(tensor_root, cls)
            for f in os.listdir(cls_folder):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(cls_folder, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
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

    dataset = WasteDataset("processed_train")
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir="runs/waste_classification_experiment")

    epochs = 5
    for epoch in range(epochs):
        loss = train(model, data_loader, criterion, optimizer, device, epoch, writer)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

    writer.close()

if __name__ == "__main__":
    main()