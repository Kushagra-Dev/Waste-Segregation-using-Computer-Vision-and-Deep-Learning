import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from evaluate import WasteDataset  # Reuse your hybrid dataset/class logic
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

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

def load_model(device, num_classes, model_path="best_hybrid_model.pth"):
    model = ResNetSwinHybrid(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_predictions(dataset, device, model):
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = "processed_test"
    dataset = WasteDataset(dataset_dir)

    num_classes = len(dataset.class_to_idx)
    class_names = sorted(dataset.class_to_idx.keys())

    model = load_model(device, num_classes)

    true_labels, predictions = get_predictions(dataset, device, model)

    plot_confusion_matrix(true_labels, predictions, class_names)

if __name__ == "__main__":
    main()
