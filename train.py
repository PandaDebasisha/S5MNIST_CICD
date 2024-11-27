import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from utils import save_augmented_samples
import random
from PIL import Image
import torchvision.transforms.functional as TF

def get_augmented_transforms():
    """Define different augmentation transforms"""
    
    # Basic normalization transform
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Rotation augmentation
    rotation_transform = transforms.Compose([
        transforms.RandomRotation(15),  # Random rotation up to 15 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Affine augmentation
    affine_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Gaussian noise augmentation
    class AddGaussianNoise(object):
        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * 0.1
            return tensor + noise
            
    noise_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return [basic_transform, rotation_transform, affine_transform, noise_transform]

def save_augmentation_examples(dataset):
    """Save examples of augmented images"""
    transforms_list = get_augmented_transforms()
    augmentation_names = ['Original', 'Rotation', 'Affine', 'Noise']
    
    # Get 10 random samples
    indices = random.sample(range(len(dataset)), 10)
    
    for idx in indices:
        # Get the PIL image from dataset
        image, _ = dataset.data[idx], dataset.targets[idx]
        image = Image.fromarray(image.numpy())
        
        # Create original tensor
        original = transforms_list[0](image).unsqueeze(0)
        
        # Apply each transform to the PIL image
        augmented_images = [
            transforms_list[i](image).unsqueeze(0)
            for i in range(1, len(transforms_list))
        ]
        
        # Save the samples
        save_augmented_samples(original, augmented_images, augmentation_names[1:])

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get transforms including augmentations
    transforms_list = get_augmented_transforms()
    
    # Load MNIST dataset with basic transform
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                 transform=None)  # No transform initially
    
    # Save augmentation examples
    save_augmentation_examples(train_dataset)
    
    # Now set the transform for training
    train_dataset.transform = transforms_list[0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Randomly choose an augmentation for this batch
        if random.random() < 0.75:  # 75% chance to apply augmentation
            transform_idx = random.randint(1, len(transforms_list)-1)
            # Convert tensor back to PIL for transforms
            data = torch.stack([
                transforms_list[transform_idx](
                    TF.to_pil_image(img.squeeze())
                ) for img in data
            ])
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 