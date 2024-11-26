import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
       self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for 2D input

        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)   # Dropout with a probability of 0.5
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.1)   # Dropout with a probability of 0.5
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 16, 3)
        self.conv6 = nn.Conv2d(16, 10, 3)
        self.conv7 = nn.Conv2d(10, 10, 3)
        self.bn3 = nn.BatchNorm2d(10)
        self.fc = nn.Conv2d(10, 10, kernel_size=1)
        # Global Average Pooling
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (1, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))

        x = self.bn1(x)
        x = self.pool2(self.bn2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        x = self.dropout1(x)
        #x = self.bn1(x)
        x = self.bn3(F.relu(self.conv7(self.conv6(F.relu(self.conv5(x))))))
        x = self.dropout2(x)
        #x = F.relu(self.conv8(F.relu(self.conv7(x))))
        x = self.fc(x)
        #x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
