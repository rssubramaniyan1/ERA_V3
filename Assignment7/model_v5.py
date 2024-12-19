import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import NormalizationTypes, get_normalization_layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # This makes it depthwise
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Net(nn.Module):
    def __init__(self, norm_type=NormalizationTypes.BATCH):
        super(Net, self).__init__()
        self.norm_type = norm_type

        # First block (28x28) - Reduced initial channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 8),
            nn.ReLU(),
             #output size 26
           
             # Reduced dropout
        
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 10),
            nn.ReLU(),
           #output size 24
            

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 16),
            nn.ReLU(),
             #output size 22
            

        )

        # Transition layer (28x28 -> 14x14)
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(in_channels=16, out_channels=20, kernel_size=1, bias=False),
            # nn.ReLU(), #output size 11
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False),
           
        )
        self.dropout1 = nn.Dropout(0.03)
        # Second block (14x14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 10),
            nn.ReLU(),
             #output size 10
            

            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 10),
            nn.ReLU(),
            #output size 8
           

            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 12),
            nn.ReLU(),
            #output size 6
           
            
        )
        self.dropout2 = nn.Dropout(0.03)

        # Transition layer (14x14 -> 7x7)
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(in_channels=12, out_channels=16, kernel_size=1, bias=False),
            # nn.ReLU(), #output size 5
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=1, bias=False),
            
            
        )

        # Third block (7x7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
            get_normalization_layer(norm_type, 8),
            nn.ReLU(), #output size 3
            
    
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),  # Reduces to 4x4
            get_normalization_layer(norm_type, 10),
            nn.ReLU(), #output size 1   
            
            
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3, padding=1, bias=False),
           
        )
        self.fc_conv = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=1, bias=False)
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        
        

    def forward(self, x):
        conv1 = self.conv1(x)
        t1 = self.transition1(conv1)
        conv2 = self.conv2(t1)
        t2 = self.transition2(conv2)
        conv3 = self.conv3(t2 + F.interpolate(t1, size=t2.size()[2:]))
        x = self.fc_conv(conv3)
        x = self.gap(x)
        
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

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

def calculate_rf_and_parameters(input_size=28, layers=None):
    """
    Calculate receptive field and other parameters at each layer
    
    Parameters:
    - input_size: Input image size (default 28 for MNIST)
    - layers: List of dictionaries containing layer parameters
    """
    if layers is None:
        # Default architecture parameters as per your model
        layers = [
            {"name": "conv1", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 1, "out_channels": 8},
            {"name": "conv2", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 8, "out_channels": 10},
            {"name": "conv3", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 10, "out_channels": 16},
            {"name": "maxpool1", "kernel_size": 2, "stride": 2, "padding": 0, "in_channels": 16, "out_channels": 8},
            {"name": "conv4", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 8, "out_channels": 10},
            {"name": "conv5", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 10, "out_channels": 16},
            {"name": "conv6", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 16, "out_channels": 24},
            {"name": "maxpool2", "kernel_size": 2, "stride": 2, "padding": 0, "in_channels": 24, "out_channels": 8},
            {"name": "conv7", "kernel_size": 3, "stride": 1, "padding": 1, "in_channels": 8, "out_channels": 16}
        ]

    # Initialize parameters
    n_in = input_size
    j_in = 1
    r_in = 1
    
    # Print header
    print("\nDetailed Layer Parameters:")
    print(f"{'Layer':<10} {'n_in':<6} {'n_out':<6} {'s':<4} {'p':<4} {'k':<4} {'j_in':<6} {'j_out':<6} {'r_in':<6} {'r_out':<6} {'RF':<6} {'channels':<10}")
    print("-" * 90)

    for layer in layers:
        # Get layer parameters
        k = layer["kernel_size"]
        s = layer.get("stride", 1)
        p = layer.get("padding", 0)
        
        # Calculate output size (n_out)
        n_out = ((n_in + 2*p - k) // s) + 1
        
        # Calculate jump (j_out)
        j_out = j_in * s
        
        # Calculate receptive field (r_out)
        r_out = r_in + (k - 1) * j_in
        
        # Calculate RF for this layer
        rf = r_out

         # Handle channel information
        if layer['in_channels'] is not None and layer['out_channels'] is not None:
            channels = f"{layer['in_channels']}->{layer['out_channels']}"
        else:
            channels = "maxpool/Transition"  # or whatever description you want for non-conv layers
        
        
        # Print all parameters including RF
        print(f"{layer['name']:<10} {n_in:<6} {n_out:<6} {s:<4} {p:<4} {k:<4} {j_in:<6} {j_out:<6} {r_in:<6} {r_out:<6} {rf:<6} {layer['in_channels']}->{layer['out_channels']:<8}")
        
        # Update values for next layer
        n_in = n_out
        j_in = j_out
        r_in = r_out

    print("\nSummary:")
    print(f"Starting Input Size: {input_size}x{input_size}")
    print(f"Final Output Size: {n_out}x{n_out}")
    print(f"Final Receptive Field: {r_out}")
    print(f"Final Jump: {j_out}")

def get_model_layers(model):
    """Extract layer information from the model"""
    layers = []
    
    # Helper function to get layer info
    def get_layer_info(layer, name):
        info = {
            "name": name,
            "in_channels": layer.in_channels if hasattr(layer, 'in_channels') else None,
            "out_channels": layer.out_channels if hasattr(layer, 'out_channels') else None,
            "kernel_size": layer.kernel_size[0] if hasattr(layer, 'kernel_size') else None,
            "stride": layer.stride[0] if hasattr(layer, 'stride') else 1,
            "padding": layer.padding[0] if hasattr(layer, 'padding') else 0
        }
        return info

    # Extract conv1 layers
    for i, layer in enumerate(model.conv1):
        if isinstance(layer, nn.Conv2d):
            layers.append(get_layer_info(layer, f"conv1_{i}"))
            
    # Extract transition1 layers
    for i, layer in enumerate(model.transition1):
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            name = f"trans1_{i}"
            if isinstance(layer, nn.MaxPool2d):
                info = {
                    "name": name,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "in_channels": None,
                    "out_channels": None
                }
                layers.append(info)
            else:
                layers.append(get_layer_info(layer, name))

    # Extract conv2 layers
    for i, layer in enumerate(model.conv2):
        if isinstance(layer, nn.Conv2d):
            layers.append(get_layer_info(layer, f"conv2_{i}"))

    # Extract transition2 layers
    for i, layer in enumerate(model.transition2):
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            name = f"trans2_{i}"
            if isinstance(layer, nn.MaxPool2d):
                info = {
                    "name": name,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "in_channels": None,
                    "out_channels": None
                }
                layers.append(info)
            else:
                layers.append(get_layer_info(layer, name))

    # Extract conv3 layers
    for i, layer in enumerate(model.conv3):
        if isinstance(layer, nn.Conv2d):
            layers.append(get_layer_info(layer, f"conv3_{i}"))

    return layers

# Usage example:
def print_model_architecture(model):
    layers = get_model_layers(model)
    calculate_rf_and_parameters(input_size=28, layers=layers)

