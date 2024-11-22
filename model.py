import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv for residual connection if channels change or stride > 1
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # Initial conv layer with stride 2
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        
        # First block (8 channels)
        self.block1_1 = ResidualBlock(8, 8)
        self.block1_2 = ResidualBlock(8, 16, stride=2)
        
        # Second block (16 channels)
        self.block2_1 = ResidualBlock(16, 16)
        self.block2_2 = ResidualBlock(16, 8)
        
        # Channel-wise attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_attention = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.Sigmoid()
        )
        
        # Final layers - adjusted for new spatial dimensions
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Initial conv with stride 2 (28x28 -> 14x14)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First block
        x = self.block1_1(x)
        x = self.block1_2(x)  # stride 2 (14x14 -> 7x7)
        
        # Second block
        x = self.block2_1(x)
        x = self.block2_2(x)
        
        # Apply attention
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_attention(y).view(b, c, 1, 1)
        x = x * y
        
        # Final layers
        x = x.view(-1, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 