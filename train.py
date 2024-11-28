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
import numpy as np

def get_augmented_transforms():
    """Define different augmentation transforms"""
    
    # Basic normalization transform
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Rotation augmentation (reduced angle for better accuracy)
    rotation_transform = transforms.Compose([
        transforms.RandomRotation(20),  # Reduced from 30 to 20 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Horizontal flip augmentation
    hflip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Vertical flip augmentation
    vflip_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return [basic_transform, rotation_transform, hflip_transform, vflip_transform]

def save_augmentation_examples(dataset):
    """Save examples of augmented images"""
    transforms_list = get_augmented_transforms()
    augmentation_names = ['Original', 'Rotation', 'X-Flip', 'Y-Flip']
    
    # Get 10 random samples
    indices = random.sample(range(len(dataset)), 10)
    
    for idx in indices:
        # Get the PIL image from dataset
        image = dataset.data[idx].numpy()
        image = Image.fromarray(image.astype(np.uint8))
        
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
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    total_batches = len(train_loader)
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{total_batches} '
                f'({100. * batch_idx / total_batches:.0f}%) '
                f'Loss: {loss.item():.4f}')
            
             # Calculate accuracy for this batch
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            accuracy = 100. * correct / len(target)
            print(f'Batch Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')

    # # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # # Get transforms including augmentations
    # transforms_list = get_augmented_transforms()
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # # Load MNIST dataset with basic transform
    # train_dataset = datasets.MNIST('./data', train=True, download=True,transform=transform
    #                              )  # Using only basic transform
    
    # # Save augmentation examples commented out
    # #save_augmentation_examples(train_dataset)
    
    # # Now set the transform for training
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     batch_size=128,
    #     shuffle=True,
    #     num_workers=2 if device.type == 'cuda' else 0
    # )
    
    # # Initialize model
    # model = MNISTModel().to(device)
    # criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # # Train for 1 epoch
    # model.train()
    # total_batches = len(train_loader)
    
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     # Augmentation code commented out but preserved
    #     # if random.random() < 0.75:
    #     #     transform_idx = random.randint(1, len(transforms_list)-1)
    #     #     data = torch.stack([
    #     #         transforms_list[transform_idx](
    #     #             Image.fromarray(img.squeeze().numpy().astype(np.uint8))
    #     #         ) for img in data
    #     #     ])
        
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()
        
    #     # Forward pass
    #     output = model(data)
    #     loss = criterion(output, target)
        
    #     # Backward pass
    #     loss.backward()
    #     optimizer.step()
        
    #     # Print progress
    #     if batch_idx % 100 == 0:
    #         print(f'Batch {batch_idx}/{total_batches} '
    #               f'({100. * batch_idx / total_batches:.0f}%) '
    #               f'Loss: {loss.item():.4f}')
            
    #         # Calculate accuracy for this batch
    #         pred = output.argmax(dim=1)
    #         correct = pred.eq(target).sum().item()
    #         accuracy = 100. * correct / len(target)
    #         print(f'Batch Accuracy: {accuracy:.2f}%')
    
    # # Save model with timestamp
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # save_path = f'model_mnist_{timestamp}.pth'
    # torch.save(model.state_dict(), save_path)
    # print(f'Model saved as {save_path}')
    
if __name__ == '__main__':
    train() 