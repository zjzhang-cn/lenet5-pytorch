import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Success! MPS device found on {torch.backends.mps.is_built()}")
    
    # Test with a simple operation
    x = torch.ones(1, device=device)
    print(f"Tensor created on: {x.device}")
else:
    print("MPS device not found. Check your macOS version.")