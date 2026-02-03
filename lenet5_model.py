import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 模型实现
    论文: "Gradient-based learning applied to document recognition" by Yann LeCun et al.
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 第一个卷积层：输入32x32x1，输出28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # 第一个池化层：输出14x14x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层：输入14x14x6，输出10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 第二个池化层：输出5x5x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
    def forward(self, x):
        # 第一个卷积+池化
        x = self.pool1(torch.tanh(self.conv1(x)))
        
        # 第二个卷积+池化
        x = self.pool2(torch.tanh(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 16*5*5)
        
        # 全连接层
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x