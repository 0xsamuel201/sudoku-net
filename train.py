import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Import thư viện vẽ biểu đồ
from tqdm import tqdm
from mobilenet import MobileNetV1

# 1. Cấu hình thiết bị (GPU nếu có, nếu không thì dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# 2. Chuẩn bị dữ liệu (Dataset & DataLoader)
# Áp dụng các phép biến đổi: Phóng to lên 224x224, chuyển thành Tensor và chuẩn hóa
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Mean & Std của CIFAR-10
])

# Tải tập huấn luyện (Train set)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Tải tập kiểm tra (Test set)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# 3. Khởi tạo Mô hình, Hàm mất mát và Bộ tối ưu
# CIFAR-10 có 10 class
model = MobileNetV1(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss() # Phù hợp cho bài toán phân loại nhiều lớp
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam thường hội tụ nhanh hơn SGD

# 4. Vòng lặp huấn luyện (Training Loop)
epochs = 3

# Khởi tạo các mảng để lưu trữ lịch sử huấn luyện
history_train_loss, history_train_acc = [], []
history_test_loss, history_test_acc = [], []

for epoch in range(epochs):
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
        
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=running_loss/len(trainloader), acc=100.*correct/total)

    # Lưu lại metrics của tập Train cho epoch hiện tại
    epoch_train_loss = running_loss / len(trainloader)
    epoch_train_acc = 100. * correct / total
    history_train_loss.append(epoch_train_loss)
    history_train_acc.append(epoch_train_acc)

    # --- ĐÁNH GIÁ TRÊN TẬP TEST ---
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
            
    # Lưu lại metrics của tập Test cho epoch hiện tại
    epoch_test_loss = test_loss / len(testloader)
    epoch_test_acc = 100. * test_correct / test_total
    history_test_loss.append(epoch_test_loss)
    history_test_acc.append(epoch_test_acc)
            
    print(f"==> Kết thúc Epoch {epoch+1} | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

print("Quá trình huấn luyện hoàn tất. Bắt đầu xuất biểu đồ...")

# ==========================================
# 5. VẼ VÀ XUẤT BIỂU ĐỒ (VISUALIZATION)
# ==========================================
# Cài đặt kích thước khung hình
plt.figure(figsize=(14, 5))

# Biểu đồ 1: Loss
plt.subplot(1, 2, 1) # (1 hàng, 2 cột, vị trí số 1)
plt.plot(range(1, epochs+1), history_train_loss, label='Train Loss', marker='o', color='blue')
plt.plot(range(1, epochs+1), history_test_loss, label='Test/Val Loss', marker='s', color='red')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Biểu đồ 2: Accuracy
plt.subplot(1, 2, 2) # (1 hàng, 2 cột, vị trí số 2)
plt.plot(range(1, epochs+1), history_train_acc, label='Train Accuracy', marker='o', color='blue')
plt.plot(range(1, epochs+1), history_test_acc, label='Test/Val Accuracy', marker='s', color='green')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Căn chỉnh layout
plt.tight_layout()

# XUẤT RA ẢNH: tham số dpi=300 giúp ảnh sắc nét (chuẩn in ấn/presentation)
plt.savefig('./output/mobilenet_training_curves.png', dpi=300, bbox_inches='tight')

plt.show()

print("Đã lưu biểu đồ thành công: './output/mobilenet_training_curves.png'")