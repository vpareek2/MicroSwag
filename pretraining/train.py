import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from utils.dataloader import DataLoader
from utils import distributed
from utils import optimization
from utils import evaluation

import models.gpt2
import models.gemma3
import models.llama3
import models.phi4
import models.mistral
import models.deepseekv3
import models.rwkv

# ===================================================
# parse_args, save_checkpoint, validate functions
# are assumed to be defined above here as before
# ===================================================
# Example validate function signature (ensure it's correct):
# def validate(model, val_loader, dist_config, device, device_type, model_type_str, steps=20):
#     ... (implementation with conditional unpacking) ...

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train language models on HellaSwag benchmark')
    parser.add_argument('--model', type=str, default='gpt2', # Default doesn't matter much now
                        help='Model architecture to train (gpt2, llama, phi4, mistral, gemma3)')
    return parser.parse_args()

def save_checkpoint(config, raw_model, optimizer, step, val_loss, train_loader, log_dir):
    """Save a checkpoint of the model and training state"""
    # Create a checkpoint of the train loader
    train_loader_checkpoint = train_loader.get_loader_checkpoint()

    # Get model type and create model-specific subdirectory
    model_type = config.model.model_type
    model_log_dir = os.path.join(log_dir, model_type) # e.g., log/llama/
    os.makedirs(model_log_dir, exist_ok=True) # Create subdir if needed

    # Construct checkpoint path
    checkpoint_path = os.path.join(model_log_dir, f"model_{step:05d}.pt")

    # Create checkpoint
    # Ensure raw_model.config exists or pass config.model_specific
    model_config_to_save = raw_model.config if hasattr(raw_model, 'config') else config.model_specific
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': model_config_to_save, # Save the specific model config
        'step': step,
        'val_loss': val_loss,
        'train_loader': train_loader_checkpoint,
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def validate(model, val_loader, dist_config, device, device_type, model_type_str, steps=20):
    """Run validation on the model"""
    model.eval()
    val_loader.reset()

    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Use the passed-in model_type_str here
                outputs = model(x, y)
                if model_type_str == "deepseek":
                    logits, loss, _ = outputs
                else:
                    logits, loss = outputs
            val_loss_accum += loss.detach()

        # Ensure steps is not zero before division
        if steps > 0:
            val_loss_accum /= steps
        else:
            # Handle case where steps=0 if necessary, maybe return 0 or NaN
            val_loss_accum = torch.tensor(float('nan'), device=val_loss_accum.device)


    # Average loss across all processes if DDP
    val_loss_accum = distributed.all_reduce_mean(val_loss_accum, dist_config["ddp"])

    return val_loss_accum.item()


# ===================================================
# === Main train function starts here ===
# ===================================================
def train():
    """Main training function"""
    # Parse command line args
    args = parse_args()

    # Load configuration
    config = Config()
    config.model.model_type = args.model # Set the type based on args
    # Manually update the specific configs after setting the model_type from args
    config.model_specific = config.model.get_model_specific_config()
    config.model_training = config.model.get_model_training_config()

    # RWKV specific setups, check the CORRECT config.model_type
    if config.model.model_type == "rwkv": # Use the updated type
        print("Setting environment variables required for RWKV...")
        try:
            # Retrieve the specific RWKVConfig instance from the updated attribute
            model_config_rwkv = config.model_specific # Get the already created RWKVConfig
            if not hasattr(model_config_rwkv, 'head_size'):
                raise AttributeError("RWKVConfig missing 'head_size' attribute.")

            os.environ['RWKV_HEAD_SIZE'] = str(model_config_rwkv.head_size)
            os.environ['RWKV_MY_TESTING'] = 'x070' # Required for the v7 kernel logic
            os.environ['RWKV_JIT_ON'] = '0' # Disable TorchScript JIT to avoid issues

            print(f"  Set RWKV_HEAD_SIZE={os.environ['RWKV_HEAD_SIZE']}")
            print(f"  Set RWKV_MY_TESTING={os.environ['RWKV_MY_TESTING']}")
            print(f"  Set RWKV_JIT_ON={os.environ['RWKV_JIT_ON']}")
        except Exception as e:
                print(f"ERROR setting RWKV environment variables: {e}")
                print("Ensure RWKVConfig is correctly defined in config.py")
                raise

    # Setup distributed training
    dist_config = distributed.setup_distributed()
    device = dist_config["device"]
    device_type = dist_config["device_type"]
    master_process = dist_config["master_process"]

    # Set up precision
    if master_process:
        print(f"Setting float32 matmul precision to {config.system.float32_matmul_precision}")
    torch.set_float32_matmul_precision(config.system.float32_matmul_precision)

    # Set up logging
    log_dir = config.system.log_dir
    os.makedirs(log_dir, exist_ok=True) # Ensures log dir exists
    log_file = os.path.join(log_dir, "log.txt")

    # Set up data loaders
    if master_process:
        print("Setting up data loaders...")

    train_loader = DataLoader(
        B=config.model_training.micro_batch_size,
        T=config.model_training.sequence_length,
        process_rank=dist_config["ddp_rank"],
        num_processes=dist_config["ddp_world_size"],
        split="train",
        data_root=config.data.data_root,
        master_process=master_process
    )

    val_loader = DataLoader(
        B=config.model_training.micro_batch_size,
        T=config.model_training.sequence_length,
        process_rank=dist_config["ddp_rank"],
        num_processes=dist_config["ddp_world_size"],
        split="val",
        data_root=config.data.data_root,
        master_process=master_process
    )

    # Configure gradient accumulation
    total_batch_size = config.model_training.total_batch_size
    B = config.model_training.micro_batch_size
    T = config.model_training.sequence_length
    ddp_world_size = dist_config["ddp_world_size"]

    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Resume or start training
    current_step = 0

    # Check the resume_training flag from the system config
    if config.system.resume_training:
        if master_process:
            print("Attempting to resume training from checkpoint...")

        # Try to find checkpoint files in the model-specific subdirectory
        model_type_str = args.model
        model_log_dir = os.path.join(log_dir, model_type_str)
        checkpoint_files = []
        if os.path.isdir(model_log_dir): # Check if model subdir exists before listing
             checkpoint_files = [f for f in os.listdir(model_log_dir)
                               if f.startswith("model_") and f.endswith(".pt")]

        if len(checkpoint_files) > 0:
            # --- RESUME LOGIC ---
            checkpoint_files = sorted(checkpoint_files)
            last_checkpoint = checkpoint_files[-1]
            # Load from the model-specific subdirectory
            checkpoint_path = os.path.join(model_log_dir, last_checkpoint)

            if master_process:
                print(f"Loading checkpoint from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model config FROM CHECKPOINT
            checkpoint_model_config = checkpoint['config']

            # Create model based on CHECKPOINT config
            # Note: We use the config stored in the checkpoint here, not the current script's config
            if args.model == "gpt2":
                model = models.gpt2.GPT(checkpoint_model_config)
            elif args.model == "llama":
                model = models.llama3.LLaMA(checkpoint_model_config)
            elif args.model == "phi4":
                model = models.phi4.Phi4(checkpoint_model_config)
            elif args.model == "gemma3":
                model = models.gemma3.Gemma3(checkpoint_model_config)
            elif args.model == "mistral":
                model = models.mistral.Mistral(checkpoint_model_config)
            elif args.model == "rwkv":
                model = models.rwkv.RWKV(checkpoint_model_config)
            elif args.model == "deepseek":
                # Assuming DeepSeekMoE __init__ takes the specific config object
                model = models.deepseekv3.DeepSeekMoE(checkpoint_model_config)
            else:
                raise ValueError(f"Unsupported model type in checkpoint: {args.model}") # Or check config type

            # Wrap model for DDP AFTER loading config and creating model instance
            model, raw_model = distributed.wrap_model_for_distributed(
                model,
                device,
                dist_config["ddp"],
                dist_config["ddp_local_rank"],
                use_compile=config.system.use_compile # Use current system config for compile setting
            )

            # Load model state_dict
            print(f"Loading model state_dict for {args.model} from checkpoint...")
            # Need to handle potential mismatches if DDP state is saved differently
            state_dict = checkpoint['model']
            # Adjust keys if necessary (e.g., remove 'module.' prefix if saved from DDP)
            if dist_config["ddp"] and not list(state_dict.keys())[0].startswith('module.'):
                 # If running DDP now but checkpoint wasn't saved from DDP model
                 pass # DDP wrapper will handle this
            elif not dist_config["ddp"] and list(state_dict.keys())[0].startswith('module.'):
                 # If not running DDP now but checkpoint was saved from DDP model
                 state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

            raw_model.load_state_dict(state_dict)
            print("Model state loaded.")

            # Parameter count (using raw_model)
            if master_process:
                total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                print(f"--- Resumed Model '{args.model}' ---")
                print(f"--- Trainable Parameters: {total_params:,} ---")

            # Create optimizer and load its state
            optimizer = optimization.create_optimizer(
                raw_model,
                config, # Pass the main config (contains training params)
                device_type,
                master_process=master_process
            )
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Load step and set learning rate scheduler state
            current_step = checkpoint['step'] + 1
            # LR scheduler state might need explicit loading if it has state

            # Set training data loader state
            if 'train_loader' in checkpoint:
                 train_loader.set(checkpoint['train_loader'])
            else:
                 if master_process: print("Warning: train_loader state not found in checkpoint.")


            if master_process:
                print(f"Resuming training from step {current_step} with validation loss {checkpoint['val_loss']:.4f}")

        else: # --- STARTING FRESH (because resume=True but no checkpoints found for this model) ---
            if master_process:
                print(f"No checkpoints found for model '{model_type_str}', starting fresh training run")

            # ====> DEBUG PRINT BLOCK <====
            if master_process:
                print(f"DEBUG: Creating model '{args.model}' using config type: {type(config.model_specific)}")
                if hasattr(config.model_specific, 'n_layer'): print(f"DEBUG: n_layer={config.model_specific.n_layer}")
                if hasattr(config.model_specific, 'n_embd'): print(f"DEBUG: n_embd={config.model_specific.n_embd}")
                if hasattr(config.model_specific, 'vocab_size'): print(f"DEBUG: vocab_size={config.model_specific.vocab_size}")
            # =============================

            # Create a new model using the CURRENT script's config
            if args.model == "gpt2":
                model = models.gpt2.create_gpt_from_config(config)
            elif args.model == "llama":
                model = models.llama3.create_llama_from_config(config)
            elif args.model == "phi4":
                # Ensure Phi4Config and create_phi4_from_config exist if used
                model = models.phi4.create_phi4_from_config(config)
            elif args.model == "gemma3":
                model = models.gemma3.create_gemma3_from_config(config)
            elif args.model == "mistral":
                model = models.mistral.create_mistral_from_config(config)
            elif args.model == "rwkv":
                model = models.rwkv.create_rwkv_from_config(config)
            elif args.model == "deepseek":
                model = models.deepseekv3.create_deepseek_from_config(config)
            else:
                raise ValueError(f"Unsupported model type: {args.model}")

            # Prepare model for distributed training
            model, raw_model = distributed.wrap_model_for_distributed(
                model,
                device,
                dist_config["ddp"],
                dist_config["ddp_local_rank"],
                use_compile=config.system.use_compile
            )

            # Parameter count (using raw_model)
            if master_process:
                 total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                 print(f"--- Fresh Model '{args.model}' ---")
                 print(f"--- Trainable Parameters: {total_params:,} ---")

            # Create optimizer
            optimizer = optimization.create_optimizer(
                raw_model,
                config, # Pass the main config
                device_type,
                master_process=master_process
            )

            # Clear the log file only when starting completely fresh
            if master_process:
                try: # Avoid error if log file doesn't exist yet
                    with open(log_file, "w") as f:
                         pass
                except FileNotFoundError:
                    pass

    else: # --- STARTING FRESH (because resume=False) ---
        if master_process:
            print("Starting fresh training run (resume_training=False)")

        # ====> DUPLICATE DEBUG PRINT BLOCK (if needed, otherwise remove) <====
        # You could put the debug prints here if resume_training is False
        # if master_process:
        #     print(f"DEBUG (resume=False): Creating model '{args.model}' using config type: {type(config.model_specific)}")
        #     # ... etc ...
        # ===================================================================

        # Create a new model using the CURRENT script's config
        if args.model == "gpt2":
            model = models.gpt2.create_gpt_from_config(config)
        elif args.model == "llama":
            model = models.llama3.create_llama_from_config(config)
        elif args.model == "phi4":
            model = models.phi4.create_phi4_from_config(config)
        elif args.model == "gemma3":
            model = models.gemma3.create_gemma3_from_config(config)
        elif args.model == "mistral":
            model = models.mistral.create_mistral_from_config(config)
        elif args.model == "rwkv":
            model = models.rwkv.create_rwkv_from_config(config)
        elif args.model == "deepseek":
            model = models.deepseekv3.create_deepseek_from_config(config)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        # Prepare model for distributed training
        model, raw_model = distributed.wrap_model_for_distributed(
            model,
            device,
            dist_config["ddp"],
            dist_config["ddp_local_rank"],
            use_compile=config.system.use_compile
        )

        # Parameter count (using raw_model) - Also needed here
        if master_process:
             total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
             print(f"--- Fresh Model '{args.model}' ---")
             print(f"--- Trainable Parameters: {total_params:,} ---")

        # Create optimizer
        optimizer = optimization.create_optimizer(
            raw_model,
            config, # Pass the main config
            device_type,
            master_process=master_process
        )

        # Clear the log file
        if master_process:
             try: # Avoid error if log file doesn't exist yet
                 with open(log_file, "w") as f:
                     pass
             except FileNotFoundError:
                 pass

    # Get LR scheduler
    get_lr = optimization.get_lr_scheduler(config, optimizer)

    # Main training loop
    max_steps = config.model_training.max_steps

    for step in range(current_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Evaluate validation loss if needed
        if step % config.model_training.eval_interval == 0 or last_step:
            # Pass args.model to validate
            val_loss = validate(
                model,
                val_loader,
                dist_config,
                device,
                device_type,
                args.model, # Pass model type string
                steps=config.data.val_loss_steps
            )

            if master_process:
                print(f"validation loss: {val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss:.4f}\n")

                # Save checkpoint if needed
                # Ensure raw_model is defined before saving
                if step > 0 and (step % config.model_training.checkpoint_interval == 0 or last_step):
                    if 'raw_model' in locals() and raw_model is not None:
                        save_checkpoint(
                            config, # Pass main config (needed for training params maybe?)
                            raw_model,
                            optimizer,
                            step,
                            val_loss,
                            train_loader,
                            log_dir
                        )
                    else:
                        if master_process:
                            print("Warning: raw_model not found or None, skipping checkpoint save.")

        # Run HellaSwag evaluation
        if step % config.model_training.eval_interval == 0 or last_step:
            results = evaluation.evaluate_hellaswag(
                model,
                device,
                device_type,
                dist_config["ddp_rank"],
                dist_config["ddp_world_size"],
                distributed
            )

            if master_process:
                print(f"HellaSwag accuracy: {results['correct']}/{results['total']}={results['accuracy']:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {results['accuracy']:.4f}\n")

        # Forward and backward passes with gradient accumulation
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        aux_loss_accum = 0.0 # For MoE training

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if dist_config["ddp"]:
                # Only synchronize gradients on the last micro-step
                # Check if model is DDP-wrapped before accessing module
                ddp_model = model.module if isinstance(model, DDP) else model
                # Check if the underlying model requires sync control (might not be needed for all)
                if hasattr(ddp_model, 'require_backward_gradient_sync'):
                     ddp_model.require_backward_gradient_sync = (micro_step == grad_accum_steps - 1)
                elif isinstance(model, DDP): # Default DDP behavior
                     model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)


            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Call model first
                outputs = model(x, y)

                # Conditional unpacking and loss handling
                if args.model == "deepseek":
                    logits, combined_loss, _ = outputs # Unpack 3 for deepseek
                    # Calculate aux loss explicitly for logging
                    # Add check for non-empty logits/targets
                    if logits.numel() > 0 and y.numel() > 0:
                         main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
                         # Ensure combined_loss is valid before subtracting
                         if torch.is_tensor(combined_loss) and combined_loss.numel() > 0:
                              aux_loss = combined_loss - main_loss
                         else: # Handle potential invalid combined_loss
                              aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
                              main_loss = combined_loss # Fallback? Or handle error
                    else: # Handle empty batch case
                        main_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16, requires_grad=True) # Ensure it requires grad if needed later
                        aux_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
                        combined_loss = main_loss + aux_loss

                else:
                    logits, combined_loss = outputs # Unpack 2 for others
                    # No separate aux loss for other models
                    main_loss = combined_loss
                    aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)

            # Scale the combined loss for gradient accumulation
            # Ensure combined_loss is valid before division/backward
            if torch.is_tensor(combined_loss) and combined_loss.requires_grad:
                 loss_to_backward = combined_loss / grad_accum_steps

                 # Accumulate detached losses for logging
                 loss_accum += main_loss.detach()
                 aux_loss_accum += aux_loss.detach() # Accumulate aux loss

                 # Backward pass
                 loss_to_backward.backward()
            else:
                 # Skip accumulation/backward if loss is invalid (e.g., from empty batch)
                 if master_process: print(f"Warning: Skipping micro_step {micro_step} due to invalid loss.")


        # Average loss across processes if DDP
        loss_accum = distributed.all_reduce_mean(loss_accum, dist_config["ddp"])
        aux_loss_accum = distributed.all_reduce_mean(aux_loss_accum, dist_config["ddp"])

        # Clip gradients
        # Ensure model has parameters before clipping
        model_params = list(model.parameters())
        if model_params:
             norm = torch.nn.utils.clip_grad_norm_(model_params, config.model_training.grad_clip)
        else:
             norm = torch.tensor(0.0) # Or handle appropriately

        # Update learning rate
        lr = get_lr(step)

        # Optimizer step
        optimizer.step()

        # Wait for GPU to finish
        if device_type == "cuda":
            torch.cuda.synchronize()

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt if dt > 0 else 0 # Avoid division by zero

        # GPU Memory Logging
        gpu_mem_gb_max_reserved = 0.0
        if device_type == "cuda":
            # Get PEAK reserved memory in GB for the current device
            gpu_mem_bytes = torch.cuda.max_memory_reserved(device=device)
            gpu_mem_gb_max_reserved = gpu_mem_bytes / (1024**3) # Convert bytes to GB
            # Optional: Reset peak stats for next interval if desired
            # torch.cuda.reset_peak_memory_stats(device=device)

        if master_process:
            log_str = f"step: {step:5d} | loss: {loss_accum.item():.6f}"
            if args.model == "deepseek":
                    log_str += f" | aux: {aux_loss_accum.item():.6f}"
            # Use the max_reserved value in the log string
            log_str += f" | lr: {lr:.4e} | norm: {norm.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | gpu_mem: {gpu_mem_gb_max_reserved:.2f}GB"
            print(log_str)

            # Write to log file (optional: add memory here too)
            with open(log_file, "a") as f:
                log_entry = f"{step} train {loss_accum.item():.6f}"
                if args.model == "deepseek":
                        log_entry += f" aux {aux_loss_accum.item():.6f}"
                f.write(log_entry + "\n")

    # Clean up distributed training
    distributed.cleanup_distributed(dist_config["ddp"])

    if master_process:
        print("Training complete!")

# ===================================================
# === Main execution block ===
# ===================================================
if __name__ == "__main__":
    train()
