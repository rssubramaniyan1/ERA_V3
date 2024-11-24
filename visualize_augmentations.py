import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_augmented_images(num_samples=10):
    # Define augmentation transforms
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST dataset
    dataset = datasets.MNIST('data', train=True, download=True)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    
    for i in range(num_samples):
        # Get a random image
        idx = torch.randint(len(dataset), (1,)).item()
        img, label = dataset[idx]
        
        # Original image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original: {label}')
        
        # Augmented image
        aug_img = transform(img)
        # Convert from tensor to numpy for display
        aug_img = aug_img.permute(1, 2, 0).numpy()
        # Denormalize
        aug_img = (aug_img * 0.5 + 0.5).clip(0, 1)
        
        axes[1, i].imshow(aug_img)
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')
    plt.close()

if __name__ == "__main__":
    show_augmented_images() 