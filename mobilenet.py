# implement a digit classifier using simple neural network based on MobileNet architecture
# MobileNetV1 sử dụng Depthwise Separable Convolution để giảm số lượng tham số và tăng tốc độ tính toán, phù hợp cho các tác vụ phân loại ảnh như nhận diện chữ số.
# Cấu trúc mạng bao gồm một conv layer đầu tiên để trích xuất đặc trưng cơ bản, sau đó là nhiều block Depthwise Separable Convolution để trích xuất đặc trưng sâu hơn, cuối cùng là một lớp phân loại để dự đoán nhãn của ảnh.
# train model on MNIST dataset and evaluate its performance

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 1. Depthwise Convolution (groups = in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Pointwise Convolution (kernel_size = 1)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # Hàm helper cho Convolution layer đầu tiên
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # Cấu trúc mạng chính
        self.features = nn.Sequential(
            conv_bn(3, 32, 2), # Đầu vào là ảnh RGB (3 channels)
            
            # Khởi tạo các block Depthwise Separable Conv
            # (in_channels, out_channels, stride)
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            
            # 5 blocks liên tiếp có cấu trúc giống nhau
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),
            
            # Global Average Pooling (đưa feature map về kích thước 1x1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Lớp phân loại cuối cùng
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten tensor trước khi đưa vào lớp Linear
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    # Khởi tạo mô hình cho tập dữ liệu có 1000 class (như ImageNet)
    model = MobileNetV1(num_classes=1000)
    
    # In ra số lượng tham số (sẽ rơi vào khoảng ~3.2 triệu cho MobileNet V1 1.0)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số của mô hình: {total_params:,}")
    
    # Tạo một dummy input: (batch_size, channels, height, width)
    x = torch.randn(1, 3, 224, 224)
    
    # Chạy forward pass
    output = model(x)
    
    print(f"Kích thước đầu vào: {x.shape}")
    print(f"Kích thước đầu ra: {output.shape}") 
    # Kỳ vọng đầu ra là: torch.Size([1, 1000])