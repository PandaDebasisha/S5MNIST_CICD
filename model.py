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
        self.fc1 = nn.Linear(10*3*3, 10)
        #self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))

        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool2(self.bn2(F.relu(self.conv4(F.relu(self.conv3(x))))))
        x = self.dropout2(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = x.view(-1, 10 * 3 * 3)
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = self.gap(x)
        #x = x.view(-1, 10)
        return F.log_softmax(x)