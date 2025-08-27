import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from evaluate import WasteDataset  # Reuse dataset class if needed
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

def load_model(device, num_classes, model_path="model.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
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
