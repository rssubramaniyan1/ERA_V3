import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


class DataAugmentation:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def calculate_mean_std(self):
        """Calculate mean and std of training data for normalization"""
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
            
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
        
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        
        for data, _ in tqdm(train_loader, desc='Calculating mean and std'):
            channels_sum += torch.mean(data)
            channels_squared_sum += torch.mean(data**2)
            num_batches += 1
        
        self.mean = channels_sum/num_batches
        self.std = (channels_squared_sum/num_batches - self.mean**2)**0.5
        
        print(f'Dataset Mean: {self.mean:.4f}, Std: {self.std:.4f}')
        return self.mean, self.std

    def get_transforms(self, train=True):
        """Get transforms for training or testing"""
        if self.mean is None or self.std is None:
            self.mean, self.std = self.calculate_mean_std()
    
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
                
                transforms.Normalize((self.mean.item(),), (self.std.item(),))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((self.mean.item(),), (self.std.item(),))
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

    def visualize_augmentations(self, num_samples=10):
        """Visualize original and augmented versions of random samples"""
        # Get transforms
        train_transform = self.get_transforms(train=True)
        
        # Load dataset
        dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        
        # Create figure
        fig = plt.figure(figsize=(20, 4))
        
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Original image
            img, label = dataset[idx]
            img = img.squeeze().numpy()  # Convert to numpy for display
            
            # Display original image
            ax = plt.subplot(2, num_samples, i + 1)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Original\nDigit: {label}', pad=10)
            else:
                ax.set_title(f'Digit: {label}', pad=10)
            
            # Augmented image
            img_pil = transforms.ToPILImage()(torch.tensor(img).unsqueeze(0))  # Create PIL image
            aug_img = train_transform(img_pil)  # Use the PIL image
            aug_img = self.denormalize(aug_img)
            
            ax = plt.subplot(2, num_samples, i + num_samples + 1)
            ax.imshow(aug_img.squeeze(), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Augmented', pad=10)
        
        plt.suptitle('MNIST Augmentation Samples', fontsize=16, y=0.95)
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save the figure
        plt.savefig('outputs/augmentation_samples.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print("Augmentation visualization saved as 'outputs/augmentation_samples.png'")

if __name__ == '__main__':
    augmenter = DataAugmentation()
    augmenter.visualize_augmentations()