import torch
import onnxruntime as ort
import numpy as np
import time

from mobilenet import MiniMobileNet

def benchmark_pytorch(model, input_tensor, num_runs=1000):
    """PyTorch benchmark on CPU"""
    print("[*] Varming up PyTorch...")
    # Warm-up
    for _ in range(50):
        _ = model(input_tensor)
        
    print(f"[*] Benchmarking PyTorch ({num_runs} runs)...")
    start_time = time.perf_counter()
    with torch.no_grad(): # turn off gradient calculation for inference
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000 # convert to milliseconds
    return avg_latency

def benchmark_onnx(ort_session, input_np, num_runs=1000):
    """ONNX Runtime benchmark on CPU"""
    print("[*] Varming up ONNX Runtime...")
    # get the name of the first input of the ONNX model (usually 'input')
    input_name = ort_session.get_inputs()[0].name
    
    # Warm-up
    for _ in range(50):
        _ = ort_session.run(None, {input_name: input_np})
        
    print(f"[*] Benchmarking ONNX Runtime ({num_runs} runs)...")
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = ort_session.run(None, {input_name: input_np})
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000 # convert to milliseconds
    return avg_latency

def main():
    # ---------------------------------------------------------
    # 1. SETUP & LOAD MODELS
    # ---------------------------------------------------------
    weight_path = './weights/sudoku-net.pth'
    onnx_path = './weights/sudoku-net.onnx'
    
    # create dummy input gray image of shape (1, 1, 28, 28)
    dummy_input_pt = torch.randn(1, 1, 28, 28)
    dummy_input_np = dummy_input_pt.numpy() # ONNX using numpy array as input instead of PyTorch tensor

    print("=== LOADING MODELS ===")
    # load PyTorch
    pt_model = MiniMobileNet(num_classes=10, in_channels=1)
    pt_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    pt_model.eval() # must be in eval() mode
    print("[+] PyTorch model loaded.")

    # load the ONNX model using ONNX Runtime (Running on CPU)
    # if using GPU, use ['CUDAExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("[+] ONNX model loaded.")

    # ---------------------------------------------------------
    # 2. CORRECTNESS CHECK
    # ---------------------------------------------------------
    print("\n=== CORRECTNESS CHECK ===")
    # get output from PyTorch
    with torch.no_grad():
        pt_output = pt_model(dummy_input_pt).numpy()
        
    # get output from ONNX
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: dummy_input_np})[0]

    # compare the maximum difference between the two output matrices
    max_diff = np.max(np.abs(pt_output - ort_output))
    print(f"Maximum difference between PyTorch and ONNX: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("[+] EXCELLENT! ONNX produces identical results to PyTorch.")
    else:
        print("[-] WARNING: Significant error found during ONNX export!")

    # ---------------------------------------------------------
    # 3. SPEED BENCHMARK
    # ---------------------------------------------------------
    print("\n=== BENCHMARK (CPU) ===")
    num_tests = 2000 # run for 2000 iterations to get a stable average latency
    
    pt_latency = benchmark_pytorch(pt_model, dummy_input_pt, num_runs=num_tests)
    ort_latency = benchmark_onnx(ort_session, dummy_input_np, num_runs=num_tests)

    print("\n=== FINAL RESULTS ===")
    print(f"- PyTorch inference time (1 image):    {pt_latency:.4f} ms  (~{1000/pt_latency:.0f} FPS)")
    print(f"- ONNX Runtime inference time (1 image): {ort_latency:.4f} ms  (~{1000/ort_latency:.0f} FPS)")
    
    speedup = pt_latency / ort_latency
    print(f"\n=> EVALUATION: ONNX Runtime is {speedup:.2f}x faster than PyTorch!")

if __name__ == "__main__":
    main()