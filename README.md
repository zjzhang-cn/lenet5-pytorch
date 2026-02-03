# LeNet-5 MNIST 图像分类项目

这个项目实现了经典的LeNet-5卷积神经网络，用于MNIST手写数字分类任务。

## 项目结构

```
├── lenet5_model.py        # LeNet-5模型定义
├── lenet5_train.py        # 模型训练脚本
├── lenet5_inference.py    # 模型推理脚本
├── lenet5_mnist.pth       # 训练好的模型文件
├── training_history.png   # 训练历史图表
├── suppress_warnings.py   # 警告抑制工具
├── test.py               # 原始测试文件
├── README.md             # 项目说明
├── PROJECT_SUMMARY.md    # 项目完成总结
└── data/                 # MNIST数据集目录
    ├── MNIST/
    │   ├── raw/
    │   └── processed/
    └── ...
```

## 模型架构

LeNet-5是Yann LeCun在1998年提出的经典卷积神经网络，包含：

1. **第一层**：卷积层 (1→6 channels, 5x5 kernel) + 平均池化
2. **第二层**：卷积层 (6→16 channels, 5x5 kernel) + 平均池化  
3. **第三层**：全连接层 (400→120)
4. **第四层**：全连接层 (120→84)
5. **输出层**：全连接层 (84→10)

激活函数使用tanh函数（原论文设计）。

## 使用方法

### 1. 训练模型

```bash
python lenet5_train.py
```

训练脚本将会：
- 自动下载MNIST数据集到`./data`目录
- 训练LeNet-5模型10个epoch
- 保存训练好的模型到`lenet5_mnist.pth`
- 生成训练历史图表`training_history.png`

### 2. 推理预测

```bash
python lenet5_inference.py
```

推理脚本提供多种功能：
- 使用MNIST测试集随机样本进行演示
- 预测自定义图像文件
- 评估模型在整个测试集上的性能

## 特性

### 训练特性
- 支持MPS (Apple Silicon)、CUDA和CPU设备
- 自动数据下载和预处理
- 实时训练进度监控
- 可视化训练历史曲线
- 模型性能评估

### 推理特性
- 多种推理模式（单图、批量、演示）
- 图像预处理自动化
- 预测结果可视化
- 置信度分析
- 按类别性能分析

## 数据预处理

- 将MNIST 28×28图像调整为32×32（符合LeNet-5标准输入）
- 标准化：均值0.1307，标准差0.3081
- 自动支持RGB到灰度转换

## 预期性能

在MNIST测试集上，该实现预期达到：
- 准确率：98%+
- 训练时间：约10-15分钟（M1 Mac）
- 模型大小：约240KB

## 自定义图像预测

要预测自定义手写数字图像：

1. 准备图像文件（支持常见格式：PNG、JPG等）
2. 运行推理脚本选择选项2
3. 输入图像路径

**注意**：为获得最佳效果，图像应该：
- 包含清晰的手写数字
- 数字居中且占据大部分图像区域
- 背景为白色或浅色
- 数字为黑色或深色

## 常见问题

### libjpeg警告
如果你看到关于libjpeg的警告消息：
```
UserWarning: Failed to load image Python extension: 'dlopen(...libjpeg.9.dylib'...
```

这个警告不会影响MNIST项目的功能。我们的脚本已经自动抑制了这些警告。如果你想手动解决这个问题，可以：

1. **简单解决方案**（推荐）：脚本已经自动处理，忽略即可
2. **完整解决方案**：重新安装torchvision
   ```bash
   conda activate ml-gpu
   conda uninstall torchvision
   conda install torchvision -c pytorch
   ```

### 环境要求
- macOS (Apple Silicon优先，Intel也支持)
- Python 3.8+
- Conda环境管理器

## 依赖包

- PyTorch
- TorchVision  
- Matplotlib
- NumPy
- Pillow (PIL)

## 技术说明

### 设备支持
代码自动检测并使用可用的最佳计算设备：
1. Apple Silicon MPS (优先)
2. NVIDIA CUDA
3. CPU (后备)

### 模型保存格式
模型以PyTorch状态字典格式保存，包含：
- 模型权重
- 架构信息

### 图像处理流程
1. 调整大小到32×32
2. 转换为Tensor
3. 标准化
4. 添加批次维度（推理时）

## 扩展建议

- 尝试不同的优化器（SGD、RMSprop等）
- 调整学习率和批次大小
- 添加数据增强技术
- 实验不同的激活函数（ReLU、LeakyReLU等）
- 添加Dropout层防止过拟合
- 尝试更深的网络结构

## 参考文献

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.