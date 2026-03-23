import torch
from pathlib import Path

def determine_device():
    # Force CPU if current PyTorch build cannot actually use your GPU
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
        device = "cuda"
    else:
        device = "cpu"
    return device    


def determine_dtype(device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32    
    return dtype        


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