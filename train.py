import warnings
import os

# 在导入torchvision之前设置环境变量和警告过滤
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import time

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from lenet5_model import LeNet5

def get_device():
    """获取可用的设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_data(batch_size=64, input_size=32):
    """
    加载和预处理MNIST数据集
    LeNet-5原论文中使用32x32的图像，所以我们需要将28x28的MNIST图像填充到32x32
    """
    # 数据预处理：将图像大小调整为input_size（LeNet-5标准输入为32x32，CNN使用28x28）
    transform_list = [transforms.ToTensor()]
    if input_size != 28:
        transform_list.insert(0, transforms.Resize((input_size, input_size)))
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)
    
    # 下载和加载训练数据
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载和加载测试数据
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, num_epochs=10, learning_rate=0.001,
                 optimizer_name='adam', scheduler_gamma=0.7, log_interval=None):
    """训练模型"""

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"开始训练，使用设备: {device}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            log_steps = log_interval if log_interval else 200
            if batch_idx % log_steps == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)

        # 测试阶段
        test_accuracy = evaluate_model(model, test_loader, device)

        # 学习率调度器
        if scheduler:
            scheduler.step()
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'测试准确率: {test_accuracy:.2f}%, 耗时: {epoch_time:.2f}秒')
        print("-" * 50)
    
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def save_model(model, filepath, architecture_name='LeNet5'):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': architecture_name
    }, filepath)
    print(f"模型已保存到: {filepath}")

def plot_training_history(train_losses, train_accuracies, test_accuracies):
    """绘制训练历史"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 绘制损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率差异
    plt.subplot(1, 3, 3)
    accuracy_diff = [train_acc - test_acc for train_acc, test_acc in zip(train_accuracies, test_accuracies)]
    plt.plot(epochs, accuracy_diff, 'm-', label='Train-Test Gap')
    plt.title('Overfitting Monitor')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载MNIST数据集...")
    train_loader, test_loader = load_data(batch_size=64)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    model = LeNet5(num_classes=10)
    model.to(device)
    
    # 打印模型结构
    print("\nLeNet-5 模型结构:")
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    print("\n开始训练...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=10,
        learning_rate=0.001
    )
    
    # 最终测试
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"\n最终测试准确率: {final_accuracy:.2f}%")
    
    # 保存模型
    model_path = 'lenet5_mnist.pth'
    save_model(model, model_path)
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accuracies, test_accuracies)
    
    print("训练完成！")

if __name__ == "__main__":
    main()