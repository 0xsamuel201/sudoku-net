import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MiniMobileNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(MiniMobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # super lite version of MobileNetV1, optimized for small input images (like 28x28)
        self.features = nn.Sequential(
            conv_bn(in_channels, 32, stride=2),             # 28x28 -> 14x14
            
            DepthwiseSeparableConv(32, 64, stride=1),       # 14x14 -> 14x14
            DepthwiseSeparableConv(64, 128, stride=2),      # 14x14 -> 7x7
            DepthwiseSeparableConv(128, 128, stride=1),     # 7x7 -> 7x7
            DepthwiseSeparableConv(128, 256, stride=2),     # 7x7 -> 4x4
            
            # AdaptiveAvgPool2d(1) will pool any spatial size (like 4x4 or 3x3) down to 1x1
            nn.AdaptiveAvgPool2d((1, 1))                    
        )
        
        # the final linear layer now only takes 256 features instead of 1024
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # test the original MNIST size: 1 channel, 28x28 image
    model = MiniMobileNet(num_classes=10, in_channels=1)
    
    # calculate total params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")