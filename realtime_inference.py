import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import numpy as np
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
        # Keep all ResNet layers except the final avgpool and fc
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.swin_features = nn.Sequential(*list(swin.children())[:-1])  # Remove head
        self.classifier = nn.Linear(2048 + 768, num_classes)

        self.gradients = None
        # Register hook for gradients on last conv layer in ResNet
        self.resnet_features[-1].register_full_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple with gradients for outputs
        self.gradients = grad_output[0]

    def forward(self, x):
        resnet_feat_map = self.resnet_features(x)  # Feature map for Grad-CAM [b, 2048, h, w]
        resnet_feat = self.resnet_avgpool(resnet_feat_map).flatten(1)  # [b, 2048]
        swin_feat = self.swin_features(x).squeeze(-1).squeeze(-1)  # [b, 768]
        combined = torch.cat([resnet_feat, swin_feat], dim=1)
        out = self.classifier(combined)
        return out, resnet_feat_map

    def get_gradients(self):
        return self.gradients

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)
    return tensor

def generate_gradcam(model, input_tensor, class_index):
    model.zero_grad()
    output, features = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    if class_index is None:
        class_index = pred_class

    score = output[0, class_index]
    score.backward(retain_graph=True)

    gradients = model.get_gradients()  # [1, 2048, h, w]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # [2048]

    activations = features.detach()[0]  # [2048, h, w]
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.sum(activations, dim=0)
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    heatmap = cv2.resize(heatmap, (input_tensor.size(3), input_tensor.size(2)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

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

        # Forward + Grad-CAM
        outputs, _ = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        class_name = CLASS_NAMES[pred.item()]
        confidence = confidence.item()

        heatmap = generate_gradcam(model, input_tensor, pred.item())

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

# Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

# Blend heatmap with frame
        overlayed = cv2.addWeighted(frame, 0.6, heatmap_resized, 0.4, 0)
        
        # Thresholding heatmap to find salient regions (tune the threshold as needed)
        gray_heatmap = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray_heatmap, 100, 255, cv2.THRESH_BINARY)  # Threshold to binary map

# Find contours around activated areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If contours found, get the largest contour bounding box
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw green rectangle on overlayed frame
        cv2.rectangle(overlayed, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Put prediction text etc.
        cv2.putText(overlayed, f"Prediction: {class_name} ({confidence*100:.1f}%)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlayed, f"FPS: {fps:.1f}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Waste Classification with Grad-CAM - Press q to quit", overlayed)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
