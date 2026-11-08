import time
import torch

def main():
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    
    if not cuda_avail:
        print("CUDA is NOT available. Exiting.")
        return
        
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_vram_gb = device_props.total_memory / (1024 ** 3)
    print(f"Device Name: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"CUDA Device Capability: {device_props.major}.{device_props.minor}")
    
    # Test BF16 matrix multiplication
    print("\nTesting BF16 Matrix Multiplication on GPU...")
    device = torch.device("cuda:0")
    
    # Warmup
    size = 4096
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Timed run
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_ms = ((end - start) / iters) * 1000
    # 2 * N^3 FLOPs for NxN matmul
    tflops = (2 * (size ** 3) / ((end - start) / iters)) / 1e12
    print(f"BF16 {size}x{size} MatMul Average Latency: {avg_ms:.2f} ms")
    print(f"Estimated Throughput: {tflops:.2f} TFLOPS")
    print("Verification completed successfully!")

if __name__ == "__main__":
    main()
