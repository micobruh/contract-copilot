import torch
from pathlib import Path


def determine_device():
    # Prefer CUDA when available, but fall back to CPU if the installed
    # PyTorch/CUDA stack cannot actually execute kernels on this GPU.
    if not torch.cuda.is_available():
        return "cpu"

    try:
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1
        return "cuda"
    except Exception as exc:
        print(f"CUDA unavailable at runtime, falling back to CPU: {exc}")
        return "cpu"


def determine_dtype(device):
    if device != "cuda":
        return torch.float32

    major, _minor = torch.cuda.get_device_capability(0)

    # bfloat16 support is reliable on newer GPUs only.
    if major >= 8:
        return torch.bfloat16

    # Older CUDA GPUs are more likely to work with float16 than bfloat16.
    return torch.float16


def determine_model_path(model_name, local_model_map, models_root):
    if model_name not in local_model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = Path(f"./{models_root}/{local_model_map[model_name]}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model folder not found: {model_path.resolve()}"
        )
    
    model_path_str = str(model_path)
    return model_path_str
