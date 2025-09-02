import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
)
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize


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


class WasteDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_root):
        self.samples = []
        self.class_to_idx = {}
        classes = sorted(
            [d for d in os.listdir(tensor_root) if os.path.isdir(os.path.join(tensor_root, d))]
        )
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(tensor_root, cls)
            for f in os.listdir(cls_folder):
                if f.lower().endswith('.pt'):
                    self.samples.append((os.path.join(cls_folder, f), idx))
        self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label


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
    all_probs = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names, normalized=False, save_path=None):
    cm = confusion_matrix(labels, preds)
    if normalized:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm_display = cm
        title = 'Confusion Matrix'
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to: {save_path}")
    plt.show()
    return cm  # Return non-normalized cm


def plot_per_class_metrics(labels, preds, class_names, save_path=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=range(len(class_names))
    )
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, precision * 100, width, label='Precision')
    ax.bar(x, recall * 100, width, label='Recall')
    ax.bar(x + width, f1 * 100, width, label='F1-Score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Percentage')
    ax.set_title('Per-Class Precision, Recall, and F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Per-class metrics plot saved to: {save_path}")
    plt.show()


def plot_roc_auc(labels, probs, class_names, save_path=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=range(n_classes))
    plt.figure(figsize=(12, 10))

    colors = plt.get_cmap('tab10').colors  
    auc_values = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append((roc_auc, i, fpr, tpr))

    auc_values.sort(reverse=True)
    for idx, (roc_auc, i, fpr, tpr) in enumerate(auc_values):
        color = colors[i % len(colors)]
        plt.plot(
            fpr, tpr, lw=3, color=color,
            label=f"{class_names[i]} (AUC = {roc_auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random (AUC = 0.50)")
    # ZOOM IN!
    plt.xlim([0.0, 0.1])
    plt.ylim([0.9, 1.01])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curve (Zoomed Top Left)', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), fontsize=13, borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, dpi=175, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    plt.show()



def plot_confidence_histograms(labels, probs, class_names, save_path=None):
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes // 3 + 1, 3, figsize=(18, 12))
    axes = axes.flatten()

    for class_idx, ax in enumerate(axes):
        if class_idx >= n_classes:
            ax.axis('off')
            continue
        class_probs = probs[labels == class_idx, class_idx]
        ax.hist(class_probs, bins=20, range=(0, 1), alpha=0.7, color='skyblue')
        ax.set_title(f'Confidence Distribution for {class_names[class_idx]}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confidence histograms saved to: {save_path}")
    plt.show()


def save_classification_report(labels, preds, class_names, save_path="classification_report.csv"):
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(save_path)
    print(f"Classification report saved to: {save_path}")


def save_confusion_matrix_csv(cm, class_names, save_path="confusion_matrix.csv"):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(save_path)
    print(f"Confusion matrix saved to CSV: {save_path}")


def prepare_class_names(dataset):
    class_names = [None] * len(dataset.class_to_idx)
    for k, v in dataset.class_to_idx.items():
        class_names[v] = k
    return class_names


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset_dir = "processed_test"
    dataset = WasteDataset(dataset_dir)
    print(f"Using data from: {dataset_dir}")

    num_classes = len(dataset.class_to_idx)
    class_names = prepare_class_names(dataset)
    print("Class names:", class_names)

    model = load_model(device, num_classes)

    true_labels, predictions, probabilities = get_predictions(dataset, device, model)

    # Confusion matrices (raw and normalized)
    cm = plot_confusion_matrix(true_labels, predictions, class_names, normalized=False, save_path="confusion_matrix.png")
    plot_confusion_matrix(true_labels, predictions, class_names, normalized=True, save_path="confusion_matrix_normalized.png")

    # Classification report
    print(classification_report(true_labels, predictions, target_names=class_names, digits=3))
    save_classification_report(true_labels, predictions, class_names)

    # Save confusion matrix CSV
    save_confusion_matrix_csv(cm, class_names)

    # Per-class metric bar chart
    plot_per_class_metrics(true_labels, predictions, class_names, save_path="per_class_metrics.png")

    # ROC and AUC curves
    plot_roc_auc(true_labels, probabilities, class_names, save_path="roc_curves.png")

    # Confidence histograms for each class
    plot_confidence_histograms(true_labels, probabilities, class_names, save_path="confidence_histograms.png")


if __name__ == "__main__":
    main()
