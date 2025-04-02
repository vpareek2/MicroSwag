from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any

# -------------------------------------------------------------------
# BASE CONFIGS
# -------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base configuration class for model architectures"""
    model_type: str = "gpt2"  # Identifier for the model type

    def get_model_specific_config(self):
        """Return the model-specific configuration based on model_type"""
        if self.model_type == "gpt2":
            return GPT2Config()
        elif self.model_type == "llama":
            return LLaMAConfig()
        elif self.model_type == "phi4":
            return Phi4Config()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_model_training_config(self):
        """Return the model-specific training configuration"""
        if self.model_type == "gpt2":
            return GPT2TrainingConfig()
        elif self.model_type == "llama":
            return LLaMATrainingConfig()
        elif self.model_type == "phi4":
            return Phi4TrainingConfig()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

@dataclass
class BaseTrainingConfig:
    """Base training configuration shared across models"""
    # Batch sizes
    total_batch_size: int = 524_288  # 2**19 ~0.5M tokens
    micro_batch_size: int = 16  # micro-batch size up to 64 on a100
    sequence_length: int = 1024  # sequence length for training

    # Common optimization settings
    grad_clip: float = 1.0  # gradient clipping value

    # Schedule settings
    eval_interval: int = 250  # interval between evaluations
    checkpoint_interval: int = 2500  # interval between checkpoints

# -------------------------------------------------------------------
# MODEL-SPECIFIC CONFIGS
# -------------------------------------------------------------------

@dataclass
class GPT2Config:
    """GPT-2 model architecture configuration"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # number of tokens (adjusted for more factors of 2)
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension

@dataclass
class GPT2TrainingConfig(BaseTrainingConfig):
    """GPT-2 specific training configuration as per GPT-3 paper"""
    # GPT-2 specific optimization
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 715  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW

@dataclass
class LLaMAConfig:
    """LLaMA model architecture configuration"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # number of tokens (same as GPT-2 for compatibility with tokenizer)
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key/value heads for grouped query attention
    n_embd: int = 768  # embedding dimension
    ffn_dim_multiplier: float = 1.0  # multiplier for feed-forward layer dimension
    multiple_of: int = 256  # multiple of for hidden dimension rounding
    norm_eps: float = 1e-5  # epsilon for normalization
    rope_theta: float = 10000.0  # base for rotary positional embeddings
    use_scaled_rope: bool = False  # whether to use scaled rotary positional embeddings

@dataclass
class LLaMATrainingConfig(BaseTrainingConfig):
    """LLaMA specific training configuration"""
    # LLaMA specific optimization
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 715  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW

@dataclass
class Phi4Config:
    """Phi-4 model architecture configuration"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # number of tokens (same as GPT-2 for compatibility with tokenizer)
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key/value heads for grouped query attention
    n_embd: int = 768  # embedding dimension
    ffn_dim_multiplier: float = 1.0  # multiplier for feed-forward layer dimension
    norm_eps: float = 1e-5  # epsilon for normalization
    rope_theta: float = 250000.0  # base for rotary positional embeddings (much higher than LLaMA)
    use_scaled_rope: bool = True  # whether to use scaled rotary positional embeddings
    qk_layernorm: bool = True  # whether to apply normalization to queries and keys

@dataclass
class Phi4TrainingConfig(BaseTrainingConfig):
    """Phi-4 specific training configuration"""
    # Keeping similar training params to the other models for fair comparison, will change hyperparams
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 715  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW
# -------------------------------------------------------------------
# CONFIGS
# -------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    data_root: str = "edu_fineweb10B"  # root directory for data
    val_loss_steps: int = 20  # number of steps for validation loss calculation

@dataclass
class SystemConfig:
    """System and hardware configuration"""
    seed: int = 1337  # random seed
    use_compile: bool = True  # whether to use torch.compile
    resume_training: bool = True  # whether to resume training
    log_dir: str = "log"  # directory for logs and checkpoints
    float32_matmul_precision: str = "high"  # precision for float32 matmul

@dataclass
class Config:
    """Main configuration class that combines all sub-configurations"""
    model: ModelConfig = field(default_factory=ModelConfig)
    model_specific: Any = field(init=False)  # Will be set in post_init
    model_training: Any = field(init=False)  # Will be set in post_init
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Initialize model-specific configs after initialization"""
        self.model_specific = self.model.get_model_specific_config()
        self.model_training = self.model.get_model_training_config()

# Default configuration
default_config = Config()
