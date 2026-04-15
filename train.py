import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Import thư viện vẽ biểu đồ
from tqdm import tqdm
from mobilenet import MobileNetV1

def parse_args():
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Huấn luyện MobileNetV1 từ đầu trên tập CIFAR-10")
    
    # Khai báo các tham số CLI
    parser.add_argument('--epochs', type=int, default=5, help='Số lượng epoch để huấn luyện (mặc định: 5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Kích thước batch (mặc định: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (mặc định: 0.001)')
    parser.add_argument('--weight-file', type=str, default='mobilenet_v1_best.pth', help='Tên file để lưu trọng số mô hình')
    parser.add_argument('--plot-file', type=str, default='training_curves.png', help='Tên file ảnh xuất biểu đồ')
    
    return parser.parse_args()

def main():
    # 1. Lấy tham số từ CLI
    args = parse_args()
    print("=== THÔNG SỐ HUẤN LUYỆN ===")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Learning Rate: {args.lr}")
    print(f"- Output Weight: {args.weight_file}")
    print(f"- Output Plot: {args.plot_file}")
    print("===========================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}\n")

    # 2. Chuẩn bị dữ liệu
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Khởi tạo Model, Loss, Optimizer
    model = MobileNetV1(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Biến để theo dõi quá trình
    history_train_loss, history_train_acc = [], []
    history_test_loss, history_test_acc = [], []
    best_test_acc = 0.0 # Biến để lưu lại model tốt nhất

    # 4. Vòng lặp huấn luyện
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=running_loss/len(trainloader), acc=100.*correct/total)

        history_train_loss.append(running_loss / len(trainloader))
        history_train_acc.append(100. * correct / total)

        # --- Đánh giá trên tập Test ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
        epoch_test_loss = test_loss / len(testloader)
        epoch_test_acc = 100. * test_correct / test_total
        history_test_loss.append(epoch_test_loss)
        history_test_acc.append(epoch_test_acc)
                
        print(f"==> Kết thúc Epoch {epoch+1} | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

        # 5. Lưu trọng số nếu độ chính xác trên tập test tăng lên
        if epoch_test_acc > best_test_acc:
            print(f"    [*] Tìm thấy model tốt hơn ({best_test_acc:.2f}% -> {epoch_test_acc:.2f}%). Đang lưu trọng số vào {args.weight_file}...")
            torch.save(model.state_dict(), args.weight_file)
            best_test_acc = epoch_test_acc

    print("\nQuá trình huấn luyện hoàn tất!")

    # 6. Vẽ và xuất biểu đồ
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs+1), history_train_loss, label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, args.epochs+1), history_test_loss, label='Test/Val Loss', marker='s', color='red')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(range(1, args.epochs+1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs+1), history_train_acc, label='Train Accuracy', marker='o', color='blue')
    plt.plot(range(1, args.epochs+1), history_test_acc, label='Test/Val Accuracy', marker='s', color='green')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(1, args.epochs+1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ thành công: '{args.plot_file}'")

if __name__ == "__main__":
    main()