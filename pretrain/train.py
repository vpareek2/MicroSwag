import os
import time
import argparse
import torch
import torch.nn.functional as F

from config import Config
from utils.dataloader import DataLoader
from utils import distributed
from utils import optimization
from utils import evaluation
import models.gpt2

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train language models on HellaSwag benchmark')
    parser.add_argument('--model', type=str, default='gpt2', help='Model architecture to train')
    return parser.parse_args()

def save_checkpoint(config, model, optimizer, step, val_loss, train_loader, log_dir):
    """Save a checkpoint of the model and training state"""
    # Create a checkpoint of the train loader
    train_loader_checkpoint = train_loader.get_loader_checkpoint()

    # Construct checkpoint path
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")

    # Create checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': model.config,
        'step': step,
        'val_loss': val_loss,
        'train_loader': train_loader_checkpoint,
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def validate(model, val_loader, dist_config, device, device_type, steps=20):
    """Run validation on the model"""
    model.eval()
    val_loader.reset()

    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += loss.detach()

        val_loss_accum /= steps

    # Average loss across all processes if DDP
    val_loss_accum = distributed.all_reduce_mean(val_loss_accum, dist_config["ddp"])

    return val_loss_accum.item()

def train():
    """Main training function"""
    # Parse command line args
    args = parse_args()

    # Load configuration
    config = Config()
    config.model.model_type = args.model

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
    os.makedirs(log_dir, exist_ok=True)
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

    if config.system.resume_training:
        if master_process:
            print("Attempting to resume training from checkpoint...")

        # Try to find checkpoint files
        checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]

        if len(checkpoint_files) > 0:
            # Load latest checkpoint
            checkpoint_files = sorted(checkpoint_files)
            last_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(log_dir, last_checkpoint)

            if master_process:
                print(f"Loading checkpoint from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Get model-specific config from the architecture-specific module
            if args.model == "gpt2":
                model = models.gpt2.GPT(checkpoint['config'])
            else:
                raise ValueError(f"Unsupported model type: {args.model}")

            # Move model to device
            model, raw_model = distributed.wrap_model_for_distributed(
                model,
                device,
                dist_config["ddp"],
                dist_config["ddp_local_rank"],
                use_compile=config.system.use_compile
            )

            # Load model state
            raw_model.load_state_dict(checkpoint['model'])

            # Create optimizer and load its state
            optimizer = optimization.create_optimizer(
                raw_model,
                config,
                device_type,
                master_process=master_process
            )
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Load step and set learning rate
            current_step = checkpoint['step'] + 1

            # Set training data state
            train_loader.set(checkpoint['train_loader'])

            if master_process:
                print(f"Resuming training from step {current_step} with a validation loss of {checkpoint['val_loss']:.4f}")
        else:
            if master_process:
                print("No checkpoints found, starting fresh training run")

            # Create a new model
            if args.model == "gpt2":
                model = models.gpt2.GPT(config.model_specific)
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

            # Create optimizer
            optimizer = optimization.create_optimizer(
                raw_model,
                config,
                device_type,
                master_process=master_process
            )

            # Clear the log file
            if master_process:
                with open(log_file, "w") as f:
                    pass
    else:
        # Always start fresh
        if master_process:
            print("Starting fresh training run")

        # Create a new model
        if args.model == "gpt2":
            model = models.gpt2.create_gpt_from_config(config)
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

        # Create optimizer
        optimizer = optimization.create_optimizer(
            raw_model,
            config,
            device_type,
            master_process=master_process
        )

        # Clear the log file
        if master_process:
            with open(log_file, "w") as f:
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
            val_loss = validate(
                model,
                val_loader,
                dist_config,
                device,
                device_type,
                steps=config.data.val_loss_steps
            )

            if master_process:
                print(f"validation loss: {val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss:.4f}\n")

                # Save checkpoint if needed
                if step > 0 and (step % config.model_training.checkpoint_interval == 0 or last_step):
                    save_checkpoint(
                        config,
                        raw_model,
                        optimizer,
                        step,
                        val_loss,
                        train_loader,
                        log_dir
                    )

        # Run HellaSwag evaluation if needed
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

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if dist_config["ddp"]:
                # Only synchronize gradients on the last micro-step
                model.require_backward_gradient_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        # Average loss across processes if DDP
        loss_accum = distributed.all_reduce_mean(loss_accum, dist_config["ddp"])

        # Clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.model_training.grad_clip)

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
        tokens_per_sec = tokens_processed / dt

        if master_process: # <--- IMPORTANT: Only prints on rank 0
            print(f"step: {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # Clean up distributed training
    distributed.cleanup_distributed(dist_config["ddp"])

    if master_process:
        print("Training complete!")

if __name__ == "__main__":
    train()
