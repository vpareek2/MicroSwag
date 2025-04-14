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
    # Move model to device
    model.to(device)

    # Apply torch.compile if requested
    if use_compile:
        model = torch.compile(model)

    # Wrap with DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank],  find_unused_parameters=True)

    # Get raw model for checkpointing
    raw_model = unwrap_model(model)

    return model, raw_model

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
