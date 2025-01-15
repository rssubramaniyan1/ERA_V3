# ResNet50 Training on ImageNet

This project implements training of a ResNet50 model on the ImageNet dataset, achieving 69.30% validation accuracy.

## Project Structure

- `download_imagenet.py`: Downloads and sets up the ImageNet-1k dataset using Hugging Face datasets
- `dataset.py`: Implements data loading and preprocessing for ImageNet
- `Train_Resnet.py`: Main training script with ResNet50 implementation
- `Utils.py`: Utility functions for training, checkpointing, and metrics

## Key Features

### Data Processing
- Custom data loading with efficient preprocessing
- Data augmentation for training
- Normalized inputs using ImageNet statistics
- Efficient data loading with multiple workers

### Training Implementation
- ResNet50 architecture
- Mixed precision training (AMP)
- Gradient accumulation (4x)
- Learning rate scheduling with OneCycleLR
- Label smoothing for better generalization
- Stochastic Weight Averaging (SWA)
- Efficient memory usage with channels_last memory format

### Optimizations
- AdamW optimizer with weight decay
- Gradient clipping
- Early stopping when validation accuracy > 70% or no improvement for 5 epochs
- Checkpoint saving for best models
- CUDA optimizations for faster training

## Training Details

- **Model**: ResNet50
- **Dataset**: ImageNet-1k (1.2M training images, 50K validation images)
- **Best Validation Accuracy**: 69.30%
- **Training Device**: NVIDIA A10G GPU
- **Batch Size**: 256 (effective batch size: 1024 with gradient accumulation)

## Model Deployment

The trained model is deployed as a Hugging Face Space with a Gradio interface, providing:
- Top-1 prediction with confidence score
- Top-5 predictions with confidence scores
- Easy-to-use web interface for image classification

## Requirements

    - datasets>=2.15.0
    - torch
    - torchvision
    - huggingface-hub>=0.19.0
    - python-dotenv
    - psutil
    - tqdm
    - pillow>=9.0.0
    - fsspec>=2023.0.0
    - albumentations
    - load_dotenv

## Data Augmentation Pipeline

### Training Augmentations
- Random Resized Crop (224x224)
- Random Horizontal Flip (50% probability)
- Color Jittering:
  - Brightness adjustment: ±0.2
  - Contrast adjustment: ±0.2
  - Saturation adjustment: ±0.2
  - Hue adjustment: ±0.1
- Random Grayscale conversion (10% probability)
- Normalization with ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Validation Augmentations
- Resize to 256x256
- Center Crop to 224x224
- Normalization with same ImageNet statistics

### Memory Optimizations
- Images stored in channels_last memory format
- Non-blocking tensor transfers to GPU
- Efficient data loading with num_workers=8
- Pin memory enabled for faster data transfer
