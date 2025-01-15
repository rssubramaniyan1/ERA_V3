import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load ImageNet class labels
with open('imagenet_classes.txt', 'r') as f:
    idx2label = [line.strip() for line in f.readlines()]

def load_model():
    # Initialize model
    model = models.resnet50(weights=None)
    # Load the checkpoint
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
    # Extract only the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load the model
model = load_model()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    # Preprocess the image
    image = preprocess(Image.fromarray(image))
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Create dictionary for top-1
    top1_result = {
        idx2label[top5_catid[0].item()]: float(top5_prob[0].item())
    }
    
    # Create dictionary for top-5
    top5_result = {
        idx2label[top5_catid[i].item()]: float(top5_prob[i].item())
        for i in range(5)
    }
    
    return top1_result, top5_result

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=[
        gr.Label(num_top_classes=1, label="Top-1 Prediction"),
        gr.Label(num_top_classes=5, label="Top-5 Predictions")
    ],
    title="ImageNet Classification with ResNet50",
    description="Upload an image to classify it into one of 1000 ImageNet categories.",
    examples=[["example1.jpg"], ["example2.jpg"]]
)

iface.launch()
