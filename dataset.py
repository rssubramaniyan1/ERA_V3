from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import multiprocessing as mp

def custom_collate(batch):
    """Custom collate function to handle the batch creation."""
    try:
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels
    except Exception as e:
        print(f"Error in collate: {e}")
        shapes = [item[0].shape for item in batch]
        print(f"Image shapes in batch: {shapes}")
        raise e

class ImageNetDataset(Dataset):
    def __init__(self, split='train', transform=None):
        print(f"Loading {split} split...")
        self.dataset = load_dataset(
            'imagenet-1k',
            split=split,
            cache_dir='/mnt/volume2/huggingface_cache/'
        )
        self.transform = transform
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = np.array(item['image'])
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[-1] == 4:  # Handle RGBA
            img = img[..., :3]
            
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Ensure img is a tensor with correct shape
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        
        # Ensure correct channel order (C, H, W)
        if img.shape[0] != 3:
            img = img.permute(2, 0, 1)

        # Return class index instead of one-hot
        return img, item['label']

def get_transforms(train=True, resolution=224):
    """Return transform pipeline."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                size=(resolution, resolution),
                scale=(0.08, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2,
                p=0.8
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.02, 0.2),
                hole_width_range=(0.02, 0.2),
                fill=0,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        resize_size = int(resolution/0.875)  # Calculate resize dimensions
        return A.Compose([
            A.Resize(
                height=resize_size,
                width=resize_size
            ),
            A.CenterCrop(
                height=resolution,
                width=resolution
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_data_loaders(batch_size=128, num_workers=None):
    """Get train and validation dataloaders."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print("Initializing datasets...")
    
    # Create datasets
    train_dataset = ImageNetDataset(
        split='train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = ImageNetDataset(
        split='validation',
        transform=get_transforms(train=False)
    )
    
    print(f"\nCreating data loaders...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing data loaders...")
    try:
        train_loader, val_loader = get_data_loaders(batch_size=64)
        
        print("\nTesting first batch from train loader...")
        for images, labels in train_loader:
            print("\nFirst training batch:")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Images dtype: {images.dtype}")
            print(f"Labels dtype: {labels.dtype}")
            print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
            break
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc() 