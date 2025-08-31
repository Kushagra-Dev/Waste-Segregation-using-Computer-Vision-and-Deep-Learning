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
import matplotlib.pyplot as plt
import seaborn as sns

#CLASS_NAMES = ['battery','biological','cardboard','clothes','glass','metal', 'paper','plastic', 'shoes','Textile Trash', 'trash']

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


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load validation dataset
    val_dataset = WasteDataset("dataset_split/val")

    # Dynamically build CLASS_NAMES list from dataset's class_to_idx mapping
    class_names = [None] * len(val_dataset.class_to_idx)
    for cls_name, idx in val_dataset.class_to_idx.items():
        class_names[idx] = cls_name
    print("CLASS_NAMES:", class_names)

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

    # Compute overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall: {recall * 100:.2f}%")
    print(f"Weighted F1 Score: {f1_score * 100:.2f}%")

    print("\nPer-class Precision, Recall, F1 Score:")
    for idx, cls_name in enumerate(class_names):
        print(f"{cls_name}: Precision={precision_cls[idx]*100:.2f}%, Recall={recall_cls[idx]*100:.2f}%, F1={f1_cls[idx]*100:.2f}%")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Visualize confusion matrix (optional)
    plot_confusion_matrix(conf_matrix, class_names)



if __name__ == "__main__":
    main()
