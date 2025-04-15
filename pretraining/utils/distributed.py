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
    raw_model_instance = model
    model_class_name = type(raw_model_instance).__name__ # Get class name early

    # Determine master process for logging
    master_process = int(os.environ.get('RANK', 0)) == 0

    # Move model to device first
    model.to(device)

    # --- Conditional Compilation ---
    # Decide whether to apply torch.compile based on global flag AND model type
    apply_compile = use_compile # Start with the global flag from config
    if model_class_name in ['Gemma3', 'Mistral']:
        apply_compile = False # Explicitly disable for Gemma3 and Mistral
        if use_compile and master_process: # Log only if compile was globally enabled but disabled here
             print(f"Compile Info: Skipping torch.compile for model type {model_class_name} due to known issues/slowness.")

    # Apply compile if decided
    if apply_compile:
        if master_process: print(f"Compile Info: Applying torch.compile for model type {model_class_name}...")
        model = torch.compile(model)
    else:
        # Log skipping only if it wasn't already logged above
        if not (model_class_name in ['Gemma3', 'Mistral'] and use_compile) and master_process:
             print(f"Compile Info: Skipping torch.compile for model type {model_class_name} (use_compile=False).")


    # --- Conditional DDP Wrapping ---
    if ddp:
        # Decide whether to use find_unused_parameters based on model type
        if model_class_name in ['RWKV', 'DeepSeekMoE']:
            find_unused = True
        else:
            find_unused = False

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
