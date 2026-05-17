import warnings
import os

# 在导入torchvision之前设置环境变量和警告过滤
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from lenet5 import LeNet5

def get_device():
    """获取可用的设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_trained_model(model_path, device, model_class=LeNet5):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 创建模型实例
    model = model_class(num_classes=10)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 设置为评估模式
    model.eval()
    model.to(device)

    print(f"成功加载模型: {model_path}")
    return model

def preprocess_image(image_path_or_array, input_size=32):
    """
    预处理图像以适合模型
    输入可以是图像路径或numpy数组
    """
    # 数据预处理管道
    transform_list = [transforms.ToTensor()]
    if input_size != 28:
        transform_list.insert(0, transforms.Resize((input_size, input_size)))
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)
    
    if isinstance(image_path_or_array, str):
        # 如果输入是文件路径
        if not os.path.exists(image_path_or_array):
            raise FileNotFoundError(f"图像文件未找到: {image_path_or_array}")
        
        # 打开图像并转换为灰度
        image = Image.open(image_path_or_array).convert('L')
    
    elif isinstance(image_path_or_array, np.ndarray):
        # 如果输入是numpy数组
        if image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 3:
            # 如果是RGB图像，转换为灰度
            image_path_or_array = np.dot(image_path_or_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # 确保数据类型正确并归一化到0-255
        if image_path_or_array.max() <= 1.0:
            image_path_or_array = (image_path_or_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_path_or_array.astype(np.uint8))
    
    else:
        raise ValueError("输入必须是图像路径(字符串)或numpy数组")
    
    # 应用预处理
    processed_image = transform(image)
    
    return processed_image, image

def predict_single_image(model, image_tensor, device, class_names=None):
    """
    对单个图像进行预测
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]  # MNIST的类别是0-9
    
    # 添加批次维度
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # 移动到设备
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]

def predict_batch(model, image_tensors, device, class_names=None):
    """
    批量预测
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    image_tensors = image_tensors.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensors)
        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
    
    return predicted_classes.cpu().numpy(), confidences.cpu().numpy(), probabilities.cpu().numpy()

def visualize_prediction(image, predicted_class, confidence, probabilities, class_names=None):
    """
    可视化预测结果
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示原始图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'预测类别: {class_names[predicted_class]}\n置信度: {confidence:.4f}')
    ax1.axis('off')
    
    # 显示概率分布
    ax2.bar(range(len(probabilities)), probabilities, alpha=0.7)
    ax2.set_xlabel('类别')
    ax2.set_ylabel('概率')
    ax2.set_title('类别概率分布')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    
    # 高亮预测类别
    ax2.bar(predicted_class, probabilities[predicted_class], color='red', alpha=0.8)
    
    plt.tight_layout()
    plt.show()

def demo_with_mnist_test():
    """
    使用MNIST测试集进行演示
    """
    import torchvision
    
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = 'lenet5_mnist.pth'
    model = load_trained_model(model_path, device)
    
    # 加载MNIST测试集
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=False,  # 假设数据已经下载
        transform=transform
    )
    
    # 随机选择几个样本进行预测
    indices = np.random.choice(len(test_dataset), 5, replace=False)
    
    print("\\n随机选择的测试样本预测结果:")
    print("-" * 60)
    
    for i, idx in enumerate(indices):
        image_tensor, true_label = test_dataset[idx]
        
        # 预测
        predicted_class, confidence, probabilities = predict_single_image(
            model, image_tensor, device
        )
        
        # 转换为可显示的图像
        original_image = image_tensor.squeeze().cpu().numpy()
        
        print(f"样本 {i+1}:")
        print(f"  真实标签: {true_label}")
        print(f"  预测标签: {predicted_class}")
        print(f"  预测置信度: {confidence:.4f}")
        print(f"  预测{'正确' if predicted_class == true_label else '错误'}")
        print()
        
        # 可视化（可选）
        # visualize_prediction(original_image, predicted_class, confidence, probabilities.cpu().numpy())

def predict_custom_image(image_path):
    """
    预测自定义图像
    """
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = 'lenet5_mnist.pth'
    model = load_trained_model(model_path, device)
    
    try:
        # 预处理图像
        processed_image, original_image = preprocess_image(image_path)
        
        # 预测
        predicted_class, confidence, probabilities = predict_single_image(
            model, processed_image, device
        )
        
        print(f"\\n预测结果:")
        print(f"图像路径: {image_path}")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence:.4f}")
        
        # 显示top-3预测
        top3_indices = torch.argsort(probabilities, descending=True)[:3]
        print("\\nTop-3 预测:")
        for i, idx in enumerate(top3_indices):
            print(f"  {i+1}. 类别 {idx}: {probabilities[idx]:.4f}")
        
        # 可视化
        visualize_prediction(
            original_image, 
            predicted_class, 
            confidence, 
            probabilities.cpu().numpy()
        )
        
    except Exception as e:
        print(f"预测失败: {e}")

def evaluate_model_performance():
    """
    评估模型在整个测试集上的性能
    """
    import torchvision
    from torch.utils.data import DataLoader
    
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = 'lenet5_mnist.pth'
    model = load_trained_model(model_path, device)
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 评估
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    print("评估模型性能...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 按类别统计
            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == target[i]).item()
                class_total[label] += 1
    
    # 打印结果
    overall_accuracy = 100 * correct / total
    print(f"\\n总体准确率: {overall_accuracy:.2f}%")
    print(f"正确预测: {correct}/{total}")
    
    print("\\n各类别准确率:")
    print("-" * 30)
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"数字 {i}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"数字 {i}: 无样本")

def main():
    """主函数"""
    print("LeNet-5 MNIST 推理脚本")
    print("=" * 40)
    
    # 检查模型文件是否存在
    model_path = 'lenet5_mnist.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        print("请先运行 train_lenet5.py 训练模型")
        return
    
    while True:
        print("\\n选择操作:")
        print("1. 使用MNIST测试集演示预测")
        print("2. 预测自定义图像")
        print("3. 评估模型性能")
        print("4. 退出")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            demo_with_mnist_test()
        
        elif choice == '2':
            image_path = input("请输入图像路径: ").strip()
            if image_path:
                predict_custom_image(image_path)
            else:
                print("路径不能为空")
        
        elif choice == '3':
            evaluate_model_performance()
        
        elif choice == '4':
            print("再见!")
            break
        
        else:
            print("无效选择，请重新选择")

if __name__ == "__main__":
    main()