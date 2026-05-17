import torch.nn as nn


class Net(nn.Module):
    """
    现代CNN网络结构 - MNIST手写数字分类

    网络层次：
    输入 (1, 28, 28) 
        ↓
    Conv1: 1→32 channels, 3x3 kernel + ReLU
        ↓  
    Conv2: 32→64 channels, 3x3 kernel + ReLU
        ↓
    MaxPool2d: 2x2 池化
        ↓
    Dropout: 25% 丢弃率
        ↓
    Flatten: 展平为 9216 维向量
        ↓
    FC1: 9216→128 + ReLU
        ↓
    Dropout: 50% 丢弃率
        ↓
    FC2: 128→10 (输出10个类别)
        ↓
    LogSoftmax: 输出概率分布

    现代CNN特征:
    ✅ ReLU激活函数(而不是传统的tanh或sigmoid)
    ✅ 最大池化(而不是平均池化)
    ✅ Dropout正则化(防止过拟合)
    ✅ LogSoftmax + NLLLoss(现代分类损失函数)
    """

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # 第一个卷积层：1通道输入 -> 32通道输出，3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二个卷积层：32通道输入 -> 64通道输出，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 第一个Dropout层：25%的神经元被随机丢弃
        self.dropout1 = nn.Dropout(0.25)
        # 第二个Dropout层：50%的神经元被随机丢弃
        self.dropout2 = nn.Dropout(0.5)
        # 第一个全连接层：9216输入特征 -> 128输出特征
        self.fc1 = nn.Linear(9216, 128)
        # 第二个全连接层（输出层）：128输入特征 -> 输出类别
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        # 输入: (batch_size, 1, 28, 28)
        x = self.conv1(x)        # -> (batch_size, 32, 26, 26)
        x = self.relu(x)            # ReLU激活
        x = self.conv2(x)        # -> (batch_size, 64, 24, 24)
        x = self.relu(x)            # ReLU激活
        x = self.max_pool2d(x)   # -> (batch_size, 64, 12, 12)
        x = self.dropout1(x)     # 25% dropout正则化
        x = self.flatten(x)  # -> (batch_size, 9216)
        x = self.fc1(x)          # -> (batch_size, 128)
        x = self.relu(x)            # ReLU激活
        x = self.dropout2(x)     # 50% dropout正则化
        x = self.fc2(x)          # -> (batch_size, 10)
        return x
