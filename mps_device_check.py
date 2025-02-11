import torch
import time
import psutil  # For process monitoring

def check_mps_utilization(iterations=100, warmup=10):
    """
    Checks if MPS is being utilized by running a matrix multiplication 
    and monitoring GPU processes.

    Args:
        iterations: Number of iterations for the matrix multiplication.
        warmup: Number of initial iterations to discard for timing (to account for initialization overhead).
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type != 'mps':
        print("MPS is not available. Skipping GPU utilization check.")
        return

    # Create random matrices on the MPS device
    matrix_size = 2048  # Adjust as needed
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    # Warmup iterations
    for _ in range(warmup):
        C = torch.matmul(A, B)

    # Time the matrix multiplication
    start_time = time.time()
    for _ in range(iterations):
        C = torch.matmul(A, B)  # The actual computation
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_per_iteration = elapsed_time / iterations
    gflops = (2 * matrix_size**3 * iterations) / (elapsed_time * 1e9)  # Calculate GFLOPS
    print(f"Average time per iteration: {avg_time_per_iteration:.6f} seconds")
    print(f"GFLOPS: {gflops:.2f}")



    # Check GPU processes (crude but often effective)
    gpu_processes_count_start = count_gpu_processes()

    # Run a short additional computation to see if processes change
    C = torch.matmul(A, B)  # A small computation

    gpu_processes_count_end = count_gpu_processes()

    print(f"GPU processes at start: {gpu_processes_count_start}")
    print(f"GPU processes at end: {gpu_processes_count_end}")

    if gpu_processes_count_end > gpu_processes_count_start:
        print("GPU processes increased, suggesting GPU usage.")
    elif gpu_processes_count_end == gpu_processes_count_start and gpu_processes_count_end > 0:
        print("GPU processes found, but count did not change. Likely using GPU.")
    else:
        print("No GPU processes found. GPU is likely not being used.")

def print_process(process):
    # Accessing process information
    name = process.name()
    exe = process.exe()
    cwd = process.cwd()
    status = process.status()
    cpu_times = process.cpu_times()
    memory_info = process.memory_info()
    threads = process.threads()
    print(f"Name: {name}")
    print(f"Executable: {exe}")
    print(f"Current working directory: {cwd}")
    print(f"Status: {status}")
    print(f"CPU times: {cpu_times}")
    print(f"Memory info: {memory_info}")
    print(f"Threads: {threads}")

def count_gpu_processes():
    """Counts processes that might be using the GPU (crude check)."""
    count = 0
    N = len(psutil.pids())
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            name = proc.name().lower()  # Access name directly
            cmdline = ' '.join(proc.cmdline()).lower()  # Access cmdline directly
            if 'python' in name and ('mps' in cmdline or 'metal' in cmdline):
                if count==0:
                    print_process(proc)
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Handle process errors
    return count


if __name__ == "__main__":
    check_mps_utilization()