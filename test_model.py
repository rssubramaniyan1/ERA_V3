import torch
from torchvision import datasets, transforms
from model import Net
import sys
import os
import glob
import tqdm

def get_latest_model():
    """Get the latest model from the models directory"""
    models = glob.glob('models/*.pth')
    if not models:
        raise FileNotFoundError("No model files found in models directory")
    return max(models, key=os.path.getctime)

def test(model_path=None):
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If no model specified, use the latest one
    if model_path is None:
        model_path = get_latest_model()
    elif not os.path.dirname(model_path):
        model_path = os.path.join('models', model_path)
    
    print(f"Using model: {model_path}")
    
    # Check if data exists
    data_dir = 'data'
    download_required = not os.path.exists(os.path.join(data_dir, 'MNIST'))
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load model and normalization values
    checkpoint = torch.load(model_path, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mean = checkpoint.get('mean', 0.5)  # fallback to 0.5 if not found
    std = checkpoint.get('std', 0.5)    # fallback to 0.5 if not found
    
    # Load test data with same normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=download_required, transform=test_transform),
        batch_size=1000, shuffle=True)
    
    model.eval()
    
    # Test model parameters
    param_count = model.count_parameters()
    print(f"Parameter count: {param_count}")
    assert param_count < 20000, "Model has too many parameters"
    
    # Test input shape
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        output = model(dummy_input)
        assert output.shape[1] == 10, "Model output should have 10 classes"
    except Exception as e:
        print(f"Input shape test failed: {str(e)}")
        sys.exit(1)
    
    # Test accuracy over 1 epochs
    best_accuracy = 0
    for epoch in range(1):
        correct = 0
        total = 0
        
        pbar = tqdm.tqdm(test_loader, desc=f'Test Epoch {epoch + 1}/1')
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Update progress bar with current accuracy
                current_acc = 100. * correct / total
                pbar.set_description(
                    f'Test Epoch {epoch + 1}/1 | Accuracy={current_acc:.2f}%'
                )
        
        accuracy = 100. * correct / total
        print(f'Final Test Accuracy: {accuracy:.2f}%')
        
    
    
    assert accuracy > 99.4, "Model accuracy is below 99.4%"
    
    return True

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: python test_model.py [model_filename]")
        print("Example: python test_model.py model_20230615_143022.pth")
        print("If no model specified, the latest model will be used")
        sys.exit(1)
    
    model_file = sys.argv[1] if len(sys.argv) == 2 else None
    test(model_file) 