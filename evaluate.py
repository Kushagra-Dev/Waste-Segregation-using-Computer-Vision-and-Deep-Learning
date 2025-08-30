import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

class ResNetSwinHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc

        swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.swin_features = nn.Sequential(*list(swin.children())[:-1])  # Remove head

        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, x):
        resnet_feat = self.resnet_features(x).flatten(1)  # [batch, 2048]
        swin_feat = self.swin_features(x).squeeze(-1).squeeze(-1)  # [batch, 768]
        combined = torch.cat([resnet_feat, swin_feat], dim=1)
        out = self.classifier(combined)
        return out

class WasteDataset(Dataset):
    def __init__(self, image_root, size=(224, 224)):
        self.samples = []
        self.class_to_idx = {}
        classes = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(image_root, cls)
            for f in os.listdir(cls_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_folder, f), idx))

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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = WasteDataset("dataset_split/val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)

    num_classes = len(val_dataset.class_to_idx)
    model = ResNetSwinHybrid(num_classes).to(device)
    model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1_score * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    main()
