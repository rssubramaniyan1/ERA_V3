# MNIST CNN Pipeline

[![ML Pipeline](https://github.com/{username}/{repository}/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/{username}/{repository}/actions/workflows/ml-pipeline.yml)

This project implements a Convolutional Neural Network (CNN) for the MNIST dataset with automated testing and validation through GitHub Actions.

## Project Structure 

├── .github/workflows/
│ └── ml-pipeline.yml
├── models/ # Directory for saved models
├── outputs/ # Directory for visualization outputs
├── train_model.py # Training script with OneCycleLR
├── test_model.py # Testing and model validation
├── model.py # CNN architecture with normalization options
├── normalization.py # Custom normalization layers
├── augmentation.py # Data augmentation pipeline
└── requirements.txt # Project dependencies

## Features
- CNN architecture for MNIST digit classification
- Multiple normalization options (Batch, Layer, Group)
- Extensive data augmentation pipeline
- OneCycleLR scheduler for optimal training
- Automated testing and validation
- Parameter count verification (<20,000 parameters)
- Training accuracy monitoring
- Model versioning with timestamps

## Model Architecture
- Input Layer: 28x28 grayscale images
- 3 Convolutional blocks with:
  - Batch/Layer/Group Normalization
  - Dropout layers
  - ReLU activation
- Transition layers with 1x1 convolutions
- Global Average Pooling
- Output: 10 classes (digits 0-9)

## Data Augmentation
The model uses several advanced augmentation techniques:
- Random Rotation (±8 degrees)
- Random Affine Transforms
  - Translation: ±10%
  - Scale: 90-110%
- Elastic Transform
- Random Perspective
- Gaussian Blur
- Random Erasing
- Normalization using dataset statistics

## Training Details
- Optimizer: Adam
  - Learning Rate: 0.01
  - Weight Decay: 0.0001
- OneCycleLR Scheduler
  - Max LR: 0.01
  - Pct Start: 0.3
  - Three Phase: False
  - Anneal Strategy: Cosine
- Batch Size: 128
- Epochs: 20

## CI/CD Pipeline
The GitHub Actions workflow automatically:
1. Checks model architecture:
   - Parameter count < 20,000
   - Presence of normalization layers
   - Presence of dropout
   - Proper input/output shapes
2. Trains model for 20 epochs
3. Tests model performance
4. Validates accuracy > 99.4%

## Requirements
- Python 3.8+
- PyTorch 2.1.0
- Other dependencies listed in requirements.txt

## Local Setup
1. Clone the repository

bash
git clone <repository-url>
bash
pip install -r requirements.txt
bash
python train_model.py
bash
python test_model.py
bash
python -m augmentation

## Model Checkpoints
Models are saved with timestamps and include:
- Model state dict
- Optimizer state
- Scheduler state
- Training metrics
- Normalization parameters

## Visualization
The augmentation pipeline includes visualization tools to inspect:
- Original vs augmented samples
- Different augmentation effects
- Training progress
- Model performance

## Notes
- The model achieves >99.4% accuracy on the test set
- Training uses full MNIST dataset (50k train, 10k test)
- All hyperparameters are tuned for optimal performance
- The pipeline includes extensive error checking and validation

Output:
checkpoint = torch.load(model_path, map_location=device)
Layer Name                Output Shape              Param Count    
=================================================================
conv1.0.weight            [8, 1, 3, 3]              72             
conv1.2.weight            [8]                       8              
conv1.2.bias              [8]                       8              
conv1.4.weight            [10, 8, 3, 3]             720            
conv1.6.weight            [10]                      10             
conv1.6.bias              [10]                      10             
conv1.8.weight            [16, 10, 3, 3]            1440           
conv1.10.weight           [16]                      16             
conv1.10.bias             [16]                      16             
transition1.0.weight      [8, 16, 1, 1]             128            
conv2.0.weight            [10, 8, 3, 3]             720            
conv2.2.weight            [10]                      10             
conv2.2.bias              [10]                      10             
conv2.4.weight            [16, 10, 3, 3]            1440           
conv2.6.weight            [16]                      16             
conv2.6.bias              [16]                      16             
conv2.8.weight            [24, 16, 3, 3]            3456           
conv2.10.weight           [24]                      24             
conv2.10.bias             [24]                      24             
transition2.0.weight      [8, 24, 1, 1]             192            
conv3.0.weight            [10, 8, 3, 3]             720            
conv3.2.weight            [10]                      10             
conv3.2.bias              [10]                      10             
conv3.4.weight            [16, 10, 3, 3]            1440           
conv3.6.weight            [16]                      16             
conv3.6.bias              [16]                      16             
fc_conv.weight            [10, 16, 1, 1]            160            
=================================================================
Total trainable parameters: 10708

Parameter count: 10708

python train_model.py
Calculating mean and std: 100%|█████████████████████████████████████████████████| 60/60 [00:04<00:00, 13.21it/s]
Dataset Mean: 0.1307, Std: 0.3081
Starting Training Phase...
Train Epoch 1/20 |Loss=0.2980 | Train Acc=74.64% | LR=0.001603: 100%|█████████| 469/469 [01:48<00:00,  4.33it/s]
Train Epoch 2/20 |Loss=0.2114 | Train Acc=93.96% | LR=0.003251: 100%|█████████| 469/469 [01:47<00:00,  4.37it/s]
Train Epoch 3/20 |Loss=0.0947 | Train Acc=95.19% | LR=0.005503: 100%|█████████| 469/469 [01:46<00:00,  4.39it/s]
Train Epoch 4/20 |Loss=0.1920 | Train Acc=95.83% | LR=0.007753: 100%|█████████| 469/469 [01:48<00:00,  4.34it/s]
Train Epoch 5/20 |Loss=0.0838 | Train Acc=96.08% | LR=0.009399: 100%|█████████| 469/469 [01:48<00:00,  4.33it/s]
Train Epoch 6/20 |Loss=0.1105 | Train Acc=96.44% | LR=0.010000: 100%|█████████| 469/469 [01:47<00:00,  4.37it/s]
Train Epoch 7/20 |Loss=0.0555 | Train Acc=96.65% | LR=0.009874: 100%|█████████| 469/469 [01:47<00:00,  4.36it/s]
Train Epoch 8/20 |Loss=0.1030 | Train Acc=96.85% | LR=0.009504: 100%|█████████| 469/469 [01:46<00:00,  4.41it/s]
Train Epoch 9/20 |Loss=0.1221 | Train Acc=97.10% | LR=0.008909: 100%|█████████| 469/469 [01:46<00:00,  4.42it/s]
Train Epoch 10/20 |Loss=0.0987 | Train Acc=97.14% | LR=0.008117: 100%|████████| 469/469 [01:47<00:00,  4.35it/s]
Train Epoch 11/20 |Loss=0.0573 | Train Acc=97.33% | LR=0.007170: 100%|████████| 469/469 [01:45<00:00,  4.43it/s]
Train Epoch 12/20 |Loss=0.1524 | Train Acc=97.32% | LR=0.006114: 100%|████████| 469/469 [01:47<00:00,  4.34it/s]
Train Epoch 13/20 |Loss=0.0673 | Train Acc=97.55% | LR=0.005003: 100%|████████| 469/469 [01:48<00:00,  4.33it/s]
Train Epoch 14/20 |Loss=0.0856 | Train Acc=97.72% | LR=0.003891: 100%|████████| 469/469 [01:48<00:00,  4.31it/s]
Train Epoch 15/20 |Loss=0.0534 | Train Acc=97.98% | LR=0.002836: 100%|████████| 469/469 [01:46<00:00,  4.41it/s]
Train Epoch 16/20 |Loss=0.0081 | Train Acc=97.97% | LR=0.001889: 100%|████████| 469/469 [01:47<00:00,  4.37it/s]
Train Epoch 17/20 |Loss=0.0040 | Train Acc=98.23% | LR=0.001098: 100%|████████| 469/469 [01:48<00:00,  4.31it/s]
Train Epoch 18/20 |Loss=0.0263 | Train Acc=98.32% | LR=0.000504: 100%|████████| 469/469 [01:49<00:00,  4.27it/s]
Train Epoch 19/20 |Loss=0.0263 | Train Acc=98.40% | LR=0.000135: 100%|████████| 469/469 [01:48<00:00,  4.34it/s]
Train Epoch 20/20 |Loss=0.1027 | Train Acc=98.48% | LR=0.000010: 100%|████████| 469/469 [01:48<00:00,  4.34it/s]

Training completed. Best training accuracy: 98.48%

Test Accuracy:

Test Epoch 1/1 | Accuracy=99.43%: 100%|█████████████████████████████████████████| 10/10 [00:01<00:00,  5.50it/s]
Final Test Accuracy: 99.43%