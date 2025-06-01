import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, IMAGE_SIZE, TEST_DIR
from model.cnn import CNNClassifier
import os

def predict_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.inference_mode():
        pred_logits = model(image_tensor)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1).item()

    return class_names[pred_label], pred_probs

if __name__ == "__main__":
    model = CNNClassifier().to(DEVICE)
    model.load_state_dict(torch.load("cnn_model.pth"))
    model.eval()

    image_path = os.path.join(TEST_DIR, "dogs", "dog.4001.jpg")
    class_names = ["cats", "dogs"]
    label, probs = predict_image(image_path, model, class_names)

    print(f"Prediction: {label}, Probabilities: {probs}")
