import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import time

CLASS_NAMES = ['battery','biological','Cardboard','Clothes','glass','metal', 'paper','plastic', 'shoes','trash']

transform = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

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

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)
    return tensor

def load_model(device, num_classes):
    model = ResNetSwinHybrid(num_classes)
    checkpoint = torch.load("best_hybrid_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(CLASS_NAMES)
    model = load_model(device, num_classes)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            class_name = CLASS_NAMES[pred.item()]
            confidence = confidence.item()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"Prediction: {class_name} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Waste Classification - Press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
