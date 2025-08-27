import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ['battery','biological','Cardboard','Clothes','glass','metal', 'paper','plastic', 'shoes','trash']

transform = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)
    return tensor

def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load("model.pth", map_location=device)
    filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('fc.')}
    model.load_state_dict(filtered_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            class_name = CLASS_NAMES[pred.item()]

        cv2.putText(frame, f"Prediction: {class_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Waste Classification - Press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
