import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import NormalizationTypes, get_normalization_layer


class Net(nn.Module):
    def __init__(self, norm_type=NormalizationTypes.BATCH):
        super(Net, self).__init__()
        self.norm_type = norm_type

        # First block (28x28 -> 14x14 due to stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 8, 4, [8, 28, 28]),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 10, 4, [10, 28, 28]),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 16, 4, [16, 28, 28]),
            nn.Dropout(0.1)
        )

        # Transition layer: 1x1 convolution + max pooling (28x28 -> 14x14)
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, bias=False),
            nn.MaxPool2d(2, 2)
        )

        # Second block (14x14 -> 7x7 due to max pool)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 10, 4, [10, 14, 14]),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 16, 4, [16, 14, 14]),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            get_normalization_layer(norm_type, 24, 4, [24, 14, 14]),
            nn.Dropout(0.05)
        )

        # Transition layer: 1x1 convolution + max pooling (14x14 -> 7x7)
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, bias=False),
            nn.MaxPool2d(2, 2)
        )

        # Third block (7x7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(),
            get_normalization_layer(norm_type, 10, 4, [10, 7, 7]),
            # nn.Dropout(0.05),
        )

        # self.gap = nn.AdaptiveAvgPool2d(1)

        # self.fc_conv = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, bias=False)

        

        # Calculate the size of flattened features
        self.flatten_size = 10 * 7 * 7  # Channels * Height * Width

        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, 10)
        # self.fc_norm = nn.BatchNorm1d(32)  # Use BatchNorm1d directly
        # self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        # Flatten the output from conv layers for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)  # Log-softmax for classification

        # x = self.gap(x)
        # x = self.fc_conv(x)
        
        # Squeeze out the extra dimensions (H=1, W=1)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        
        return F.log_softmax(x, dim=1)  # Changed dim=-1 to dim=1 for clarity

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

if __name__ == '__main__':
    # Test different normalization types
    for norm_type in [NormalizationTypes.BATCH, NormalizationTypes.LAYER, NormalizationTypes.GROUP]:
        print(f"\nTesting model with {norm_type} normalization:")
        model = Net(norm_type=norm_type)
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        model.count_parameters() 