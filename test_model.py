import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import glob
import pytest
from train import get_augmented_transforms
import numpy as np
from PIL import Image

def get_latest_model():
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def test_model_architecture():
    model = MNISTModel()
    
    # Test total parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be < 100000"
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%"

def test_augmentation_consistency():
    """Test if augmentations maintain digit recognizability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get transforms
    transforms_list = get_augmented_transforms()
    
    # Load a single image
    dataset = datasets.MNIST('./data', train=False, download=True, transform=None)
    image, label = dataset[0]
    image = Image.fromarray(image.numpy())
    
    predictions = []
    # Test prediction on original and augmented versions
    with torch.no_grad():
        for transform in transforms_list:
            img_tensor = transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
    
    # At least 3 out of 4 predictions should be the same
    most_common = max(set(predictions), key=predictions.count)
    assert predictions.count(most_common) >= 3, \
        "Augmentations are too strong, affecting model predictions"
    assert most_common == label, \
        f"Most common prediction {most_common} doesn't match true label {label}"

def test_prediction_confidence():
    """Test if model makes confident predictions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    confidence_threshold = 0.8  # 80% confidence threshold
    confident_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            probabilities = torch.exp(output)  # Convert log_softmax to probabilities
            max_probs, _ = torch.max(probabilities, dim=1)
            confident_predictions += (max_probs > confidence_threshold).sum().item()
            total_predictions += data.size(0)
    
    confidence_ratio = confident_predictions / total_predictions
    assert confidence_ratio > 0.9, \
        f"Only {confidence_ratio*100:.1f}% of predictions are confident (>80% probability)"

def test_noise_robustness():
    """Test model's robustness to input noise"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    noise_levels = [0.1, 0.2, 0.3]  # Different noise intensities
    min_accuracy = 0.7  # Minimum accuracy required for the noisiest level
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                # Add Gaussian noise
                noise = torch.randn_like(data) * noise_level
                noisy_data = data + noise
                
                output = model(noisy_data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.to(device)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        assert accuracy > min_accuracy, \
            f"Model accuracy with noise level {noise_level} is {accuracy*100:.1f}%, should be >{min_accuracy*100}%"

if __name__ == '__main__':
    pytest.main([__file__]) 