import argparse
import sys
import torch

from lenet5_model import LeNet5
from cnn import Net


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


MODEL_CONFIGS = {
    'lenet5': {
        'class': LeNet5,
        'input_size': 32,
        'save_name': 'lenet5_mnist.pth',
        'optimizer': 'adam',
        'lr': 0.001,
        'architecture': 'LeNet5',
    },
    'cnn': {
        'class': Net,
        'input_size': 28,
        'save_name': 'mnist_cnn.pth',
        'optimizer': 'adadelta',
        'lr': 1.0,
        'architecture': 'CNN',
    },
}


def cmd_train(args):
    from train import load_data, train_model, evaluate_model, save_model, plot_training_history

    cfg = MODEL_CONFIGS[args.model]
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    print("加载MNIST数据集...")
    train_loader, test_loader = load_data(batch_size=args.batch_size, input_size=cfg['input_size'])
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    model = cfg['class'](num_classes=10).to(device)
    print(f"\n{cfg['architecture']} 模型结构:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    print("\n开始训练...")
    train_kwargs = {
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'device': device,
        'num_epochs': args.epochs,
        'learning_rate': args.lr if args.lr is not None else cfg['lr'],
        'optimizer_name': cfg['optimizer'],
        'scheduler_gamma': args.gamma,
    }
    if args.log_interval is not None:
        train_kwargs['log_interval'] = args.log_interval
    train_losses, train_accuracies, test_accuracies = train_model(**train_kwargs)

    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"\n最终测试准确率: {final_accuracy:.2f}%")

    if not args.no_save:
        model_path = args.output or cfg['save_name']
        save_model(model, model_path, architecture_name=cfg['architecture'])

    if not args.no_plot:
        plot_training_history(train_losses, train_accuracies, test_accuracies)

    print("训练完成！")


def cmd_evaluate(args):
    from inference import load_trained_model
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import numpy as np

    cfg = MODEL_CONFIGS[args.model]
    device = get_device()
    print(f"使用设备: {device}")

    model_path = args.model_path or cfg['save_name']
    model = load_trained_model(model_path, device, model_class=cfg['class'])

    transform_list = [transforms.ToTensor()]
    if cfg['input_size'] != 28:
        transform_list.insert(0, transforms.Resize((cfg['input_size'], cfg['input_size'])))
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)

    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == target[i]).item()
                class_total[label] += 1

    overall_accuracy = 100 * correct / total
    print(f"\n总体准确率: {overall_accuracy:.2f}%")
    print(f"正确预测: {correct}/{total}")
    print("\n各类别准确率:")
    print("-" * 30)
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"数字 {i}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"数字 {i}: 无样本")


def cmd_demo(args):
    import os
    from inference import load_trained_model, predict_single_image
    from torchvision import datasets, transforms
    import numpy as np

    cfg = MODEL_CONFIGS[args.model]
    device = get_device()
    model_path = args.model_path or cfg['save_name']

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        sys.exit(1)

    model = load_trained_model(model_path, device, model_class=cfg['class'])

    transform_list = [transforms.ToTensor()]
    if cfg['input_size'] != 28:
        transform_list.insert(0, transforms.Resize((cfg['input_size'], cfg['input_size'])))
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)

    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    indices = np.random.choice(len(test_dataset), args.samples, replace=False)

    print(f"\n随机选择 {args.samples} 个测试样本预测结果:")
    print("-" * 60)

    for i, idx in enumerate(indices):
        image_tensor, true_label = test_dataset[idx]
        predicted_class, confidence, _ = predict_single_image(model, image_tensor, device)
        print(f"样本 {i+1}: 真实: {true_label}, 预测: {predicted_class}, "
              f"置信度: {confidence:.4f}  {'✓' if predicted_class == true_label else '✗'}")


def cmd_predict(args):
    import os
    from inference import load_trained_model, preprocess_image, predict_single_image, visualize_prediction

    cfg = MODEL_CONFIGS[args.model]
    device = get_device()
    print(f"使用设备: {device}")

    model_path = args.model_path or cfg['save_name']
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        sys.exit(1)

    model = load_trained_model(model_path, device, model_class=cfg['class'])

    try:
        processed_image, original_image = preprocess_image(args.image_path, input_size=cfg['input_size'])
        predicted_class, confidence, probabilities = predict_single_image(model, processed_image, device)

        print(f"\n预测结果:")
        print(f"图像路径: {args.image_path}")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence:.4f}")

        top3_indices = torch.argsort(probabilities, descending=True)[:3]
        print("\nTop-3 预测:")
        for i, idx in enumerate(top3_indices):
            print(f"  {i+1}. 类别 {idx}: {probabilities[idx]:.4f}")

        visualize_prediction(original_image, predicted_class, confidence, probabilities.cpu().numpy())

    except Exception as e:
        print(f"预测失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='LeNet-5 / CNN MNIST 统一入口')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # train
    p_train = subparsers.add_parser('train', help='训练模型')
    p_train.add_argument('--model', '-m', choices=['lenet5', 'cnn'], default='lenet5',
                         help='选择模型架构 (默认: lenet5)')
    p_train.add_argument('--epochs', type=int, default=10, help='训练轮数 (默认: 10, CNN: 4)')
    p_train.add_argument('--lr', type=float, default=None, help='学习率 (默认: lenet5=0.001, cnn=1.0)')
    p_train.add_argument('--gamma', type=float, default=0.7, help='学习率衰减 (仅 Adadelta 使用, 默认: 0.7)')
    p_train.add_argument('--batch-size', type=int, default=64, help='批次大小 (默认: 64)')
    p_train.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    p_train.add_argument('--output', type=str, default=None, help='模型保存路径 (默认按模型类型决定)')
    p_train.add_argument('--log-interval', type=int, default=None, help='日志输出间隔')
    p_train.add_argument('--no-save', action='store_true', help='不保存模型')
    p_train.add_argument('--no-plot', action='store_true', help='不生成训练历史图')
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser('evaluate', help='评估模型在测试集上的性能')
    p_eval.add_argument('--model', '-m', choices=['lenet5', 'cnn'], default='lenet5',
                        help='选择模型架构 (默认: lenet5)')
    p_eval.add_argument('--model-path', type=str, default=None, help='模型文件路径 (默认按模型类型决定)')
    p_eval.add_argument('--batch-size', type=int, default=64, help='批次大小 (默认: 64)')
    p_eval.set_defaults(func=cmd_evaluate)

    # demo
    p_demo = subparsers.add_parser('demo', help='MNIST 随机样本演示预测')
    p_demo.add_argument('--model', '-m', choices=['lenet5', 'cnn'], default='lenet5',
                        help='选择模型架构 (默认: lenet5)')
    p_demo.add_argument('--model-path', type=str, default=None, help='模型文件路径 (默认按模型类型决定)')
    p_demo.add_argument('--samples', type=int, default=5, help='预测样本数 (默认: 5)')
    p_demo.set_defaults(func=cmd_demo)

    # predict
    p_pred = subparsers.add_parser('predict', help='预测自定义图像')
    p_pred.add_argument('--model', '-m', choices=['lenet5', 'cnn'], default='lenet5',
                        help='选择模型架构 (默认: lenet5)')
    p_pred.add_argument('--model-path', type=str, default=None, help='模型文件路径 (默认按模型类型决定)')
    p_pred.add_argument('image_path', help='图像文件路径')
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
