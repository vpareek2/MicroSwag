import math
import torch

def get_lr_scheduler(config, optimizer):
    """Create a learning rate scheduler function based on configuration."""
    # Extract parameters from config
    max_lr = config.model_training.learning_rate
    min_lr = max_lr * config.model_training.min_lr_ratio
    warmup_steps = config.model_training.warmup_steps
    max_steps = config.model_training.max_steps

    def lr_lambda(step):
        # 1) linear warmup for warmup_steps steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        # 2) if step > max_steps, return min learning rate
        if step > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    def get_lr(step):
        """Get learning rate for current step and update optimizer"""
        lr = lr_lambda(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    return get_lr

def create_optimizer(model, config, device_type, master_process=True):
    """Create an optimizer for the model based on configuration."""
    return model.configure_optimizers(
        weight_decay=config.model_training.weight_decay,
        learning_rate=config.model_training.learning_rate,
        betas=config.model_training.betas,
        device_type=device_type,
        master_process=master_process
    )
