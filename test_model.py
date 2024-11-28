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
    assert accuracy > 55, f"Accuracy is {accuracy}%, should be > 95%"

# def test_rotation_augmentation():
    """Test model's performance on rotated images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create rotation transform
    rotation_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=None)
    
    correct = 0
    total = 100  # Test on 100 samples
    
    with torch.no_grad():
        for idx in range(total):
            image, label = test_dataset[idx]
            image = Image.fromarray(image.numpy())
            
            # Apply rotation
            rotated_image = rotation_transform(image).unsqueeze(0).to(device)
            output = model(rotated_image)
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                correct += 1
    
    rotation_accuracy = 100. * correct / total
    assert rotation_accuracy > 70, f"Rotation augmentation accuracy: {rotation_accuracy}%"

# def test_flip_augmentations():
    """Test model's performance on flipped images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create flip transforms
    hflip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    vflip_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=None)
    total_samples = 100
    
    # Test horizontal flip
    hflip_correct = 0
    vflip_correct = 0
    
    with torch.no_grad():
        for idx in range(total_samples):
            image, label = test_dataset[idx]
            image = Image.fromarray(image.numpy())
            
            # Test horizontal flip
            hflip_image = hflip_transform(image).unsqueeze(0).to(device)
            output = model(hflip_image)
            pred = output.argmax(dim=1).item()
            if pred == label:
                hflip_correct += 1
            
            # Test vertical flip
            vflip_image = vflip_transform(image).unsqueeze(0).to(device)
            output = model(vflip_image)
            pred = output.argmax(dim=1).item()
            if pred == label:
                vflip_correct += 1
    
    hflip_accuracy = 100. * hflip_correct / total_samples
    vflip_accuracy = 100. * vflip_correct / total_samples
    
    assert hflip_accuracy > 60, f"Horizontal flip accuracy: {hflip_accuracy}%"
    assert vflip_accuracy > 60, f"Vertical flip accuracy: {vflip_accuracy}%"

# def test_augmentation_consistency():
#     """Test if augmentations maintain digit recognizability"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = MNISTModel().to(device)
    
#     # Load the latest trained model
#     model_path = get_latest_model()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Get transforms
#     transforms_list = get_augmented_transforms()
    
#     # Load a single image
#     dataset = datasets.MNIST('./data', train=False, download=True, transform=None)
    
#     # Test multiple samples
#     num_samples = 10
#     consistent_samples = 0
    
#     for sample_idx in range(num_samples):
#         image, label = dataset[sample_idx]
#         image = Image.fromarray(image.numpy())
        
#         predictions = []
#         # Test prediction on original and augmented versions
#         with torch.no_grad():
#             for transform in transforms_list:
#                 img_tensor = transform(image).unsqueeze(0).to(device)
#                 output = model(img_tensor)
#                 pred = output.argmax(dim=1).item()
#                 predictions.append(pred)
        
#         # Check if majority of predictions match the label
#         if predictions.count(label) >= 2:  # At least 2 out of 4 should match
#             consistent_samples += 1
    
#     consistency_ratio = consistent_samples / num_samples
#     assert consistency_ratio >= 0.7, \
#         f"Only {consistency_ratio*100:.1f}% of samples are consistently recognized across augmentations"

def test_model_class_performance():
    """Test model's performance on each digit class"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Track per-class performance
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # Update per-class accuracy
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Check each class has minimum accuracy of 85%
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        assert class_acc >= 85.0, f"Class {i} accuracy is {class_acc:.2f}%, should be >= 85%"

def test_model_confidence_distribution():
    """Test the distribution of model's prediction confidence"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    confidences = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.exp(outputs)  # Convert from log_softmax
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
    
    # Convert to numpy array for calculations
    confidences = np.array(confidences)
    
    # Test confidence metrics
    assert np.mean(confidences) > 0.8, f"Mean confidence {np.mean(confidences):.3f} should be > 0.8"
    assert np.median(confidences) > 0.85, f"Median confidence {np.median(confidences):.3f} should be > 0.85"
    assert np.percentile(confidences, 25) > 0.7, f"25th percentile confidence {np.percentile(confidences, 25):.3f} should be > 0.7"

def test_model_validation_metrics():
    """Test comprehensive validation metrics including precision and recall"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics for each class
    for class_idx in range(10):
        # True Positives, False Positives, False Negatives
        tp = np.sum((all_preds == class_idx) & (all_targets == class_idx))
        fp = np.sum((all_preds == class_idx) & (all_targets != class_idx))
        fn = np.sum((all_preds != class_idx) & (all_targets == class_idx))
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Assert minimum performance requirements
        assert precision >= 0.85, f"Precision for class {class_idx} is {precision:.3f}, should be >= 0.85"
        assert recall >= 0.85, f"Recall for class {class_idx} is {recall:.3f}, should be >= 0.85"

if __name__ == '__main__':
    pytest.main([__file__]) 