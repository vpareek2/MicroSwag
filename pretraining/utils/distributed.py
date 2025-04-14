import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Set up distributed training environment and return configuration."""
    # Check if this is a DDP run
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        # DDP setup (distributed)
        assert torch.cuda.is_available(), "CUDA required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # Single-process setup
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # Autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        if master_process:
            print(f"Using device: {device}")

    # Determine device type
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Set reproducible seeds
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    return {
        "ddp": ddp,
        "ddp_rank": ddp_rank,
        "ddp_local_rank": ddp_local_rank,
        "ddp_world_size": ddp_world_size,
        "device": device,
        "master_process": master_process,
        "device_type": device_type
    }

def cleanup_distributed(ddp):
    """Clean up distributed training resources"""
    if ddp:
        destroy_process_group()

def wrap_model_for_distributed(model, device, ddp, ddp_local_rank, use_compile=True):
    """Prepare a model for distributed training by applying compile and DDP wrappers."""
    # Get the original raw model instance before any wrapping
    # This is important to get the correct class name before torch.compile might wrap it
    raw_model_instance = model

    model.to(device)

    if use_compile:
        # Apply compile *before* DDP if possible
        # Note: DDP(torch.compile(model)) is generally recommended over torch.compile(DDP(model))
        model = torch.compile(model)

    if ddp:
        # Decide whether to use find_unused_parameters based on model type
        model_class_name = type(raw_model_instance).__name__

        # Enable the flag only for RWKV and DeepSeekMoE
        if model_class_name in ['RWKV', 'DeepSeekMoE']:
            find_unused = True
        else:
            find_unused = False

        master_process = int(os.environ.get('RANK', 0)) == 0
        if master_process:
            print(f"DDP Info: Setting find_unused_parameters={find_unused} for model type {model_class_name}")

        # Pass the conditional flag to DDP
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=find_unused)

    # Get the final unwrapped model (after potential DDP and compile)
    final_raw_model = unwrap_model(model)

    return model, final_raw_model

def unwrap_model(model):
    """Unwrap a model from DDP and torch.compile wrappers."""
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

def all_reduce_mean(tensor, ddp):
    """Perform all_reduce operation if in DDP mode."""
    if ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor

def all_reduce_sum(tensor, ddp, device):
    """Convert to tensor if needed, then perform all_reduce sum operation if in DDP mode."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float, device=device)
    if ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() if tensor.numel() == 1 else tensor
