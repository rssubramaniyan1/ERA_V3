import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First block of convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )

        # Max pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second block of convolutional layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third block of convolutional layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)

        )

        # Fully convolutional layer instead of GAP (using 1x1 kernel to output 10 classes)
        self.fc_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Apply the fully connected layer
        x = self.fc_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        # Flatten the output of the convolution layer
        x = x.view(x.size(0), -1)  # Flatten to shape (batch_size, 10)

        return F.log_softmax(x, dim=-1)

    def count_parameters(self):
        print(f"{'Layer Name':<25} {'Output Shape':<25} {'Param Count':<15}")
        print("=" * 65)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                print(f"{name:<25} {str(list(param.shape)):<25} {param_count:<15}")
        print("=" * 65)
        print(f"Total trainable parameters: {total_params}\n")
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 