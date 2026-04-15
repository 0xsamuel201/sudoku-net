import argparse
import torch
import onnx

from mobilenet import MiniMobileNet

def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch MiniMobileNet model to ONNX format")
    parser.add_argument('--weight-file', type=str, required=True, help='path to .pth file (ví dụ: mobilenet_v1_best.pth)')
    parser.add_argument('--output-file', type=str, default='mobilenet_v1.onnx', help='path to output ONNX file')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes (default: 10 for CIFAR-10)')
    parser.add_argument('--img-size', type=int, default=224, help='input image size (default: 224)')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[*] Loading weights from: {args.weight_file}")
    
    # 1. init model & weights
    # init model with the same architecture as during training, but we will load the trained weights from the .pth file
    model = MiniMobileNet(num_classes=args.num_classes)
    
    # use map_location='cpu' to ensure compatibility even if the model was trained on GPU
    state_dict = torch.load(args.weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # MUST set model to eval mode before exporting to ONNX
    # if not, layers like Dropout or BatchNorm will behave incorrectly during export
    model.eval()

    # 2. create dummy input for ONNX export
    # size: (Batch_Size, Channels, Height, Width). Batch size usually set to 1 when exporting.
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, requires_grad=True)

    print(f"[*] Starting export to ONNX format...")
    
    # 3. Export model to ONNX
    torch.onnx.export(
        model,                       
        dummy_input,                 
        args.output_file,            
        export_params=True,          
        opset_version=11,            
        do_constant_folding=True,    
        input_names=['input'],       
        output_names=['output'],     
        
        # setting Dynamic Axes allows the exported ONNX model to accept variable batch sizes during inference.
        # helps the ONNX file later to accept variable batch sizes instead of being fixed to 1
        dynamic_axes={
            'input': {0: 'batch_size'},    
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[+] Successfully exported file: {args.output_file}")

    # 4. verify the exported ONNX model
    try:
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print("[+] Valid ONNX model! Graph structure has no errors.")
    except Exception as e:
        print(f"[-] Warning: ONNX verification failed: {e}")

if __name__ == "__main__":
    main()