import torch.nn as nn

class NormalizationTypes:
    BATCH = 'batch'
    LAYER = 'layer'
    GROUP = 'group'

def get_normalization_layer(norm_type, num_features, num_groups=4, normalized_shape=None):
    """
    Get the specified normalization layer
    Args:
        norm_type: Type of normalization (batch, layer, or group)
        num_features: Number of input features/channels
        num_groups: Number of groups for group normalization (only used if norm_type is 'group')
        normalized_shape: Shape for layer normalization (only used if norm_type is 'layer')
    Returns:
        Normalization layer
    """
    if norm_type == NormalizationTypes.BATCH:
        return nn.BatchNorm2d(num_features)
    elif norm_type == NormalizationTypes.LAYER:
        return nn.BatchNorm2d(num_features)
    elif norm_type == NormalizationTypes.GROUP:
        return nn.GroupNorm(num_groups, num_features)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

def get_normalization_description():
    """Get description of available normalization types"""
    return """
    Available Normalization Types:
    
    1. Batch Normalization (batch):
       - Normalizes across batch dimension
       - Good for large batch sizes
       - Most commonly used
       - Dependent on batch statistics
    
    2. Layer Normalization (layer):
       - Normalizes across channel dimension
       - Independent of batch size
       - Good for small batches
       - More stable training
    
    3. Group Normalization (group):
       - Normalizes across groups of channels
       - Independent of batch size
       - Good compromise between batch and layer norm
       - Especially useful for small batch sizes
    """

if __name__ == '__main__':
    # Example usage
    print(get_normalization_description())
    
    # Example of creating different normalization layers
    num_features = 64
    
    batch_norm = get_normalization_layer(NormalizationTypes.BATCH, num_features)
    print(f"\nBatch Normalization Layer: {batch_norm}")
    
    layer_norm = get_normalization_layer(NormalizationTypes.LAYER, num_features)
    print(f"Layer Normalization Layer: {layer_norm}")
    
    group_norm = get_normalization_layer(NormalizationTypes.GROUP, num_features, num_groups=4)
    print(f"Group Normalization Layer: {group_norm}") 