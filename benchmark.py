import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import time
from tqdm import tqdm

from mobilenet import MiniMobileNet

def main():
    # ---------------------------------------------------------
    # 1. config for benchmarking
    # ---------------------------------------------------------
    weight_path = './weights/sudoku-net.pth'
    onnx_path = './weights/sudoku-net.onnx'
    batch_size = 64 # use 1 for measuring latency per image, or larger batch size for measuring throughput (images per second)
    
    print("=== prepare MNIST data ===")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    total_images = len(testset)
    print(f"[+] Downloaded {total_images} test images.\n")

    # ---------------------------------------------------------
    # 2. load models
    # ---------------------------------------------------------
    print("=== load models ===")
    
    # Load PyTorch Model
    pt_model = MiniMobileNet(num_classes=10, in_channels=1)
    pt_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    pt_model.eval() # must set to eval() mode
    print("[+] PyTorch Model loaded.")

    # Load ONNX Model
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print("[+] ONNX Runtime Session loaded.\n")

    # ---------------------------------------------------------
    # 3. benchmark PyTorch
    # ---------------------------------------------------------
    print("=== BENCHMARK: PYTORCH ===")
    pt_correct = 0
    
    # PyTorch test loop
    start_time_pt = time.perf_counter()
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="PyTorch Inference"):
            outputs = pt_model(images)
            _, predicted = torch.max(outputs.data, 1)
            pt_correct += (predicted == labels).sum().item()
            
    end_time_pt = time.perf_counter()
    pt_total_time = end_time_pt - start_time_pt
    pt_accuracy = 100.0 * pt_correct / total_images

    # ---------------------------------------------------------
    # 4. benchmark ONNX Runtime
    # ---------------------------------------------------------
    print("\n=== BENCHMARK: ONNX RUNTIME ===")
    ort_correct = 0
    
    # ONNX test loop
    start_time_ort = time.perf_counter()
    for images, labels in tqdm(testloader, desc="ONNX Inference"):
        # convert input from Tensor to Numpy Array
        np_images = images.numpy()
        np_labels = labels.numpy()
        
        # run inference
        ort_outputs = ort_session.run(None, {input_name: np_images})[0]
        
        # find class with highest score
        predicted = np.argmax(ort_outputs, axis=1)
        ort_correct += np.sum(predicted == np_labels)
        
    end_time_ort = time.perf_counter()
    ort_total_time = end_time_ort - start_time_ort
    ort_accuracy = 100.0 * ort_correct / total_images

    # ---------------------------------------------------------
    # 5. summarize results
    # ---------------------------------------------------------
    print("\n==================================================")
    print("                summary of results             ")
    print("==================================================")
    
    print(f"[ACCURACY]")
    print(f"- PyTorch Accuracy : {pt_accuracy:.2f} % ({pt_correct}/{total_images})")
    print(f"- ONNX Accuracy    : {ort_accuracy:.2f} % ({ort_correct}/{total_images})")
    
    acc_diff = abs(pt_accuracy - ort_accuracy)
    if acc_diff < 0.01:
        print("  => ONNX maintains the same accuracy as PyTorch! No significant difference.")
    else:
        print(f"  => there is a difference of ({acc_diff:.4f}%) due to floating-point precision errors.")

    print(f"\n[SPEED / THROUGHPUT]")
    print(f"- PyTorch Total Time : {pt_total_time:.2f} seconds   |  FPS: {total_images / pt_total_time:.0f} img/s")
    print(f"- ONNX Total Time    : {ort_total_time:.2f} seconds  |  FPS: {total_images / ort_total_time:.0f} img/s")
    
    speedup = pt_total_time / ort_total_time
    print(f"  => CONCLUSION: ONNX Runtime processes images {speedup:.2f} times faster than PyTorch!")
    print("==================================================")

if __name__ == "__main__":
    main()