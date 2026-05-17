# LeNet-5 MNIST 图像分类项目

这个项目实现了经典的LeNet-5卷积神经网络，用于MNIST手写数字分类任务。同时包含一个现代CNN实现作为对比参考。

## 项目结构

```
├── main.py                # 统一CLI入口（推荐使用）
├── lenet5.py              # LeNet-5模型定义
├── cnn.py                 # 现代CNN模型定义
├── train.py               # 统一训练模块
├── inference.py           # 统一推理模块
├── check_device.py        # MPS设备检测脚本
├── lenet5_mnist.pth       # 训练好的LeNet-5模型文件
├── training_history.png   # 训练历史图表
├── requirements.txt       # Python依赖项
├── CLAUDE.md              # Claude Code配置
└── data/                  # MNIST数据集目录
    └── MNIST/
        ├── raw/
        └── processed/
```

## 模型架构

### LeNet-5（经典网络）

LeNet-5是Yann LeCun在1998年提出的经典卷积神经网络，包含：

1. **第一层**：卷积层 (1→6 channels, 5x5 kernel) + 平均池化
2. **第二层**：卷积层 (6→16 channels, 5x5 kernel) + 平均池化
3. **第三层**：全连接层 (400→120)
4. **第四层**：全连接层 (120→84)
5. **输出层**：全连接层 (84→10)

激活函数使用tanh函数（原论文设计），输入要求 32×32 灰度图。

### 现代CNN（对比参考）

`cnn.py` 中的 `Net` 类实现了现代CNN设计：
- 两个卷积层 (1→32→64 channels, 3x3 kernel)
- ReLU激活 + MaxPool池化 + Dropout正则化
- 两个全连接层 (9216→128→10)
- 原始logits输出 + CrossEntropyLoss损失函数（与LeNet-5统一）
- Adadelta优化器 + StepLR学习率调度

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据集

MNIST 数据集会在首次训练时自动下载到 `./data/` 目录，无需手动操作。

```bash
python main.py train
# 首次运行输出：Downloading http://yann.lecun.com/exdb/mnist/...
# 下载完成后 data/ 目录结构：
#   data/MNIST/raw/
#     ├── train-images-idx3-ubyte.gz    # 训练图像 (60,000张)
#     ├── train-labels-idx1-ubyte.gz    # 训练标签
#     ├── t10k-images-idx3-ubyte.gz     # 测试图像 (10,000张)
#     └── t10k-labels-idx1-ubyte.gz     # 测试标签
```

如果下载过慢，可手动从 [MNIST 官网](http://yann.lecun.com/exdb/mnist/) 下载上述四个 `.gz` 文件放入 `data/MNIST/raw/` 即可。

### 统一CLI（推荐）

```bash
# 训练
python main.py train                      # 训练LeNet-5（默认）
python main.py train -m cnn               # 训练现代CNN

# 评估
python main.py evaluate                   # 评估LeNet-5
python main.py evaluate -m cnn            # 评估CNN

# 演示
python main.py demo --samples 5           # LeNet-5演示
python main.py demo -m cnn --samples 10   # CNN演示

# 预测自定义图像
python main.py predict my_digit.png
python main.py predict -m cnn my_digit.png
```

### 直接运行脚本

```bash
python train.py       # 训练LeNet-5（独立运行）
python inference.py   # 交互式推理（独立运行）
```


## 特性

### 设备支持
所有脚本自动检测并使用可用的最佳计算设备：
1. Apple Silicon MPS (优先)
2. NVIDIA CUDA
3. CPU (后备)

### 数据预处理
- LeNet-5：将28×28调整为32×32；CNN：保持28×28
- 标准化：均值0.1307，标准差0.3081
- 自动支持RGB到灰度转换

### 模型保存格式
模型以统一的PyTorch字典格式保存：
```python
{
    'model_state_dict': model.state_dict(),
    'model_architecture': 'LeNet5'  # 或 'CNN'
}
```
加载时需要确保对应的模型类可导入（checkpoint只保存权重，不保存模型定义）。

## 预期性能

在MNIST测试集上：
- 准确率：98%+
- 训练时间：约10-15分钟（Apple Silicon Mac）
- 模型大小：约240KB

## 自定义图像预测

```bash
python main.py predict my_digit.png
# 或
python main.py predict -m cnn my_digit.png
```

也可交互式运行：`python inference.py` 选择选项2。

**注意**：为获得最佳效果，图像应该：
- 包含清晰的手写数字
- 数字居中且占据大部分图像区域
- 背景为白色或浅色，数字为黑色或深色

## 常见问题

### libjpeg警告

如果看到关于libjpeg的警告消息，通常不会影响MNIST项目的功能。如需解决，可以重新安装torchvision：

```bash
pip uninstall torchvision
pip install torchvision
```

### DataLoader崩溃或卡死

如果在macOS上训练时程序崩溃或卡死，将 `train.py` 中的 `num_workers=2` 改为 `num_workers=0` 即可。

## 依赖包

核心依赖：
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- Matplotlib >= 3.5.0
- NumPy >= 1.21.0
- Pillow >= 8.0.0

## 参考文献

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
