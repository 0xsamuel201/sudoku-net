import argparse
import torch
import onnx

# Import the lightweight model designed for MNIST
from mobilenet import MiniMobileNet 

def parse_args():
    """
    Parse command line arguments for ONNX export configuration.
    """
    parser = argparse.ArgumentParser(description="Export PyTorch MiniMobileNet to ONNX format")
    
    # File paths
    parser.add_argument('--weight-file', type=str, default='./output/sudoku-net.pth', 
                        help='Path to the trained PyTorch weights file (.pth)')
    parser.add_argument('--output-file', type=str, default='./output/sudoku-net.onnx', 
                        help='Output filename for the ONNX model')
    
    # Model configuration (Must match training setup)
    parser.add_argument('--num-classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of input channels (1 for Grayscale)')
    parser.add_argument('--img-size', type=int, default=28, help='Input image size (28 for MNIST)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[*] Loading weights from: {args.weight_file}")
    
    # ---------------------------------------------------------
    # 1. Initialize Model & Load Weights
    # ---------------------------------------------------------
    model = MiniMobileNet(num_classes=args.num_classes, in_channels=args.in_channels)
    
    try:
        # Load weights, mapping to CPU to ensure it works on machines without GPU
        state_dict = torch.load(args.weight_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("[+] Weights loaded successfully.")
    except Exception as e:
        print(f"[-] Error loading weights: {e}")
        return

    # CRITICAL: Set model to evaluation mode before exporting.
    # This disables Dropout and fixes BatchNorm running statistics.
    model.eval()

    # ---------------------------------------------------------
    # 2. Create Dummy Input
    # ---------------------------------------------------------
    # Shape: (Batch_Size, Channels, Height, Width) -> (1, 1, 28, 28) for MNIST
    dummy_input = torch.randn(1, args.in_channels, args.img_size, args.img_size, requires_grad=True)

    print(f"[*] Exporting model to ONNX format...")
    
    # ---------------------------------------------------------
    # 3. Export to ONNX
    # ---------------------------------------------------------
    torch.onnx.export(
        model,                       # The PyTorch model
        dummy_input,                 # Model input (or a tuple for multiple inputs)
        args.output_file,            # Where to save the model
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=18,            # The ONNX version to export the model to
        do_constant_folding=True,    # Optimize by pre-calculating constant operations
        input_names=['input'],       # The model's input names
        output_names=['output'],     # The model's output names
        
        # Dynamic axes allow the ONNX model to accept variable batch sizes later
        dynamic_axes={
            'input': {0: 'batch_size'},    
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[+] Model successfully exported to: {args.output_file}")

    # ---------------------------------------------------------
    # 4. Verify the ONNX Model
    # ---------------------------------------------------------
    try:
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model is valid! Graph structure is correct.")
    except Exception as e:
        print(f"[-] Warning: ONNX model validation failed: {e}")

if __name__ == "__main__":
    main()