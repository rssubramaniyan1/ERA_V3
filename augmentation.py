import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class DataAugmentation:
    def __init__(self):
        """Initialize augmentation with default values"""
        self.mean = None
        self.std = None
        self._calculate_stats = True  # Flag to control stats calculation

    def calculate_stats(self, dataset):
        """Calculate mean and std dev for the dataset"""
        if not self._calculate_stats:
            return

        print("Calculating dataset statistics...")
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=2
        )
        
        mean = 0.
        std = 0.
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
        
        mean /= len(dataset)
        std /= len(dataset)
        
        self.mean = mean.tolist()
        self.std = std.tolist()
        self._calculate_stats = False  # Disable future calculations
        
        print(f"Dataset statistics - Mean: {self.mean}, Std: {self.std}")
        return self.mean, self.std

    def get_transforms(self, train=True):
        """Get transforms for training or testing"""
        if train:
            return transforms.Compose([
                transforms.ToTensor(),
                # Small rotations only - too much rotation can make digits like 6/9 confusing
                transforms.RandomRotation((-8, 8), fill=0),
                
                # # Slight shifts to help with digit position invariance
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),  # 10% translation
                    scale=(0.9, 1.1),      # Slight scaling
                    fill=0
                ),
                
                # Very subtle elastic deformation to simulate natural handwriting variation
                transforms.ElasticTransform(
                    alpha=10.0,
                    sigma=3.0,
                    fill=0
                ),
                
                # Random perspective to simulate different viewing angles
                transforms.RandomPerspective(
                    distortion_scale=0.2,
                    p=0.1,
                    fill=0
                ),
                
                # Gaussian blur to simulate different pen strokes
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 0.5)
                ),
                
                # Random erasing for robustness against occlusion
                transforms.RandomErasing(
                    p=0.1,                # Lower probability
                    scale=(0.02, 0.1),    # Smaller erasing areas
                    ratio=(0.3, 3.3),
                    value=0
                ),
                
               transforms.Normalize(self.mean, self.std) if self.mean is not None else transforms.Lambda(lambda x: x)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std) if self.mean is not None else transforms.Lambda(lambda x: x)
            ])

    def denormalize(self, tensor):
        """Denormalize the tensor with stored mean and std"""
        return tensor * self.std + self.mean

    def apply_transforms(self, image, transform):
        """Apply transforms to image"""
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
        
        # Apply transformation
        transformed = transform(image=image)
        return transformed['image']

    def visualize_augmentations(self, num_samples=10, save_path=None):
        """
        Visualize the augmentations applied to sample images
        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization. If None, saves to default location
        """
        # Get your dataset (assuming MNIST)
        dataset = datasets.MNIST('data', train=True, download=True)
        
        # Create a figure
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Get a random image
            idx = random.randint(0, len(dataset)-1)
            img, _ = dataset[idx]
            
            # Original image
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Augmented image
            aug_img = self.get_transforms(train=True)(img)
            axes[1, i].imshow(aug_img.squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Augmented')
        
        plt.tight_layout()
        
        # Default save path if none provided
        if save_path is None:
            # Get the directory where the current file (augmentation.py) is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Create path to Outputs folder relative to current file
            save_path = os.path.join(current_dir, 'Outputs', 'augmentation_samples.png')
        
        # Ensure the Outputs directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

if __name__ == '__main__':
    augmenter = DataAugmentation()
    augmenter.visualize_augmentations()