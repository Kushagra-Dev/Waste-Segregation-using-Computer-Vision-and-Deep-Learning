import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = WasteDataset("processed_test")
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.class_to_idx))
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
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