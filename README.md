# MNIST CNN Pipeline

This project implements a Convolutional Neural Network (CNN) for the MNIST dataset with automated testing and validation through GitHub Actions.

## Project Structure 

├── .github/workflows/
│ └── ml-pipeline.yml
├── models/
├── train_model.py
├── test_model.py
├── model.py
└── requirements.txt

## Features
- CNN architecture for MNIST digit classification
- Automated testing and validation pipeline
- Parameter count verification (<25,000 parameters)
- Input shape validation (28x28)
- Training accuracy of >95% target in the 1st epoch
- Model versioning with timestamps

## CI/CD Pipeline
The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Validates model parameters and performance
5. Tests model accuracy

Output:

Using model: models/model_20241124_144442.pth
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
conv1.12.weight           [24, 16, 3, 3]            3456           
conv1.14.weight           [24]                      24             
conv1.14.bias             [24]                      24             
conv2.0.weight            [10, 24, 3, 3]            2160           
conv2.2.weight            [10]                      10             
conv2.2.bias              [10]                      10             
conv2.4.weight            [16, 10, 3, 3]            1440           
conv2.6.weight            [16]                      16             
conv2.6.bias              [16]                      16             
conv2.8.weight            [24, 16, 3, 3]            3456           
conv2.10.weight           [24]                      24             
conv2.10.bias             [24]                      24             
conv3.0.weight            [32, 24, 3, 3]            6912           
conv3.2.weight            [32]                      32             
conv3.2.bias              [32]                      32             
fc_conv.0.weight          [10, 32, 1, 1]            320            
=================================================================
Total trainable parameters: 20256

Parameter count: 20256

## Requirements
- Python 3.8+
- PyTorch
- Other dependencies listed in requirements.txt

## Local Setup
1. Clone the repository