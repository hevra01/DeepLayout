import torch
import torch.version

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using device: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")

print(torch.__version__)  # This will print the installed PyTorch version
