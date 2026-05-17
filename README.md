# LeNet-5 MNIST 图像分类项目

这个项目实现了经典的LeNet-5卷积神经网络，用于MNIST手写数字分类任务。同时包含一个现代CNN实现作为对比参考。

## 项目结构

```
├── lenet5_model.py        # LeNet-5模型定义
├── lenet5_train.py        # LeNet-5模型训练脚本
├── lenet5_inference.py    # LeNet-5模型推理脚本
├── cnn.py                 # 现代CNN模型（对比参考）
├── test.py                # MPS设备检测脚本
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
- LogSoftmax输出 + NLLLoss损失函数
- Adadelta优化器 + StepLR学习率调度

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练LeNet-5模型

```bash
python lenet5_train.py
```

训练脚本将会：
- 自动下载MNIST数据集到`./data`目录
- 训练LeNet-5模型10个epoch
- 保存训练好的模型到`lenet5_mnist.pth`
- 生成训练历史图表`training_history.png`

### LeNet-5推理预测

```bash
python lenet5_inference.py
```

推理脚本提供交互式菜单：
1. 使用MNIST测试集随机样本进行演示
2. 预测自定义图像文件
3. 评估模型在整个测试集上的性能
4. 退出

### 训练现代CNN

```bash
python cnn.py
```

可选参数：
- `--batch-size`：训练批次大小（默认64）
- `--epochs`：训练轮数（默认4）
- `--lr`：学习率（默认1.0）
- `--use_gpu`：启用MPS加速（默认使用CPU）
- `--save-model`：保存训练好的模型

## 特性

### 设备支持
代码自动检测并使用可用的最佳计算设备（LeNet-5脚本）：
1. Apple Silicon MPS (优先)
2. NVIDIA CUDA
3. CPU (后备)

注意：`cnn.py` 默认使用CPU，需要通过 `--use_gpu` 参数手动启用MPS。

### 数据预处理
- 将MNIST 28×28图像调整为32×32（符合LeNet-5标准输入）
- 标准化：均值0.1307，标准差0.3081
- 自动支持RGB到灰度转换

### 模型保存格式
模型以PyTorch字典格式保存：
```python
{
    'model_state_dict': model.state_dict(),
    'model_architecture': 'LeNet5'
}
```
加载时需要确保 `LeNet5` 类可导入（checkpoint只保存权重，不保存模型定义）。

## 预期性能

在MNIST测试集上：
- 准确率：98%+
- 训练时间：约10-15分钟（Apple Silicon Mac）
- 模型大小：约240KB

## 自定义图像预测

1. 准备图像文件（支持常见格式：PNG、JPG等）
2. 运行 `python lenet5_inference.py` 选择选项2
3. 输入图像路径

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

如果在macOS上训练时程序崩溃或卡死，将 `lenet5_train.py` 中的 `num_workers=2` 改为 `num_workers=0` 即可。

## 依赖包

核心依赖：
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- Matplotlib >= 3.5.0
- NumPy >= 1.21.0
- Pillow >= 8.0.0

## 参考文献

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
