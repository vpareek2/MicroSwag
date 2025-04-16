"""Model Configuration File"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any

@dataclass
class ModelConfig:
    """Base configuration class for model architectures"""
    model_type: str = "gpt2"

    def get_model_specific_config(self):
        """Return the model-specific configuration based on model_type"""
        if self.model_type == "gpt2":
            return GPT2Config()
        elif self.model_type == "llama":
            return LLaMAConfig()
        elif self.model_type == "mistral":
            return MistralConfig()
        elif self.model_type == "gemma3":
            return Gemma3Config()
        elif self.model_type == "deepseek":
            return DeepSeekMoEConfig()
        elif self.model_type == "rwkv":
            return RWKVConfig()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_model_training_config(self):
        """Return the model-specific training configuration"""
        if self.model_type == "gpt2":
            return GPT2TrainingConfig()
        elif self.model_type == "llama":
            return LLaMATrainingConfig()
        elif self.model_type == "mistral":
            return MistralTrainingConfig()
        elif self.model_type == "gemma3":
            return Gemma3TrainingConfig()
        elif self.model_type == "deepseek":
            return DeepSeekMoETrainingConfig()
        elif self.model_type == "rwkv":
            return RWKVTrainingConfig()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

@dataclass
class BaseTrainingConfig:
    """Base training configuration shared across models"""
    # Batch sizes
    total_batch_size: int = 524_288  # 2**19 ~0.5M tokens
    micro_batch_size: int = 64  # micro-batch size
    sequence_length: int = 1024  # sequence length for training

    # Common optimization settings
    grad_clip: float = 1.0  # gradient clipping value

    # Schedule settings
    eval_interval: int = 500  # interval between evaluations
    checkpoint_interval: int = 2500  # interval between checkpoints

# -------------------------------------------------------------------
# MODEL-SPECIFIC CONFIGS
# -------------------------------------------------------------------

# GPT-2 Configs will be the base training hyperparameters for most models, this is meant to test architectures so while these parameters are most likely
# not optimal, they are going to be the baseline. I will tune parameters later once I find the best architecture.
@dataclass
class GPT2Config:
    """GPT-2 model architecture configuration, 124,475,904 parameters"""
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

# In config.py

@dataclass
class LLaMAConfig:
    """LLaMA model architecture configuration, Params: 124,467,904"""
    model_type: str = "llama"
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 16
    n_embd: int = 704
    n_head: int = 8
    n_kv_head: int = 2
    ffn_dim_multiplier: float = 1.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    def __post_init__(self):
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

@dataclass
class LLaMATrainingConfig(BaseTrainingConfig):
    """LLaMA specific training configuration"""
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

# @dataclass
# class Phi4Config:
#     """Phi-4 model architecture configuration"""
#     block_size: int = 1024  # max sequence length
#     vocab_size: int = 50304  # number of tokens (same as GPT-2 for compatibility with tokenizer)
#     n_layer: int = 12  # number of transformer layers
#     n_head: int = 12  # number of attention heads
#     n_kv_head: int = 4  # number of key/value heads for grouped query attention
#     n_embd: int = 768  # embedding dimension
#     ffn_dim_multiplier: float = 1.0  # multiplier for feed-forward layer dimension
#     norm_eps: float = 1e-5  # epsilon for normalization
#     rope_theta: float = 250000.0  # base for rotary positional embeddings (much higher than LLaMA)
#     use_scaled_rope: bool = True  # whether to use scaled rotary positional embeddings
#     qk_layernorm: bool = True  # whether to apply normalization to queries and keys

# @dataclass
# class Phi4TrainingConfig(BaseTrainingConfig):
#     """Phi-4 specific training configuration"""
#     # Keeping similar training params to the other models for fair comparison, will change hyperparams
#     weight_decay: float = 0.1  # weight decay for optimizer
#     learning_rate: float = 6e-4  # base learning rate
#     min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
#     warmup_steps: int = 715  # number of warmup steps
#     max_steps: int = 19073  # maximum number of training steps
#     betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
#     eps: float = 1e-8  # epsilon parameter for AdamW

@dataclass
class Gemma3Config:
    """Gemma 3 model architecture configuration, Params: 124,393,600"""
    model_type: str = "gemma3"
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 15
    n_embd: int = 640
    n_head: int = 8
    n_kv_head: int = 4
    norm_eps: float = 1e-5
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    use_qk_norm: bool = True
    rope_local_theta: float = 10000.0
    rope_global_theta: float = 1000000.0
    tie_word_embeddings=True

    def __post_init__(self):
         if self.n_head % self.n_kv_head != 0:
             raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
         if self.n_embd % self.n_head != 0:
             raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

class Gemma3TrainingConfig(BaseTrainingConfig):
    """Gemma 3 specific training configuration"""
    # Similar to other models
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

@dataclass
class MistralConfig:
    """Mistral model architecture configuration, Params: 124,374,400"""
    model_type: str = "mistral"
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 15
    n_embd: int = 640
    n_head: int = 8
    n_kv_head: int = 4
    ffn_dim_multiplier: float = 1.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 1024

    def __post_init__(self):
         if self.n_head % self.n_kv_head != 0:
             raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
         if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

@dataclass
class MistralTrainingConfig(BaseTrainingConfig):
    """Mistral specific training configuration"""
    # Using similar training parameters as the other models for fair comparison
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 715  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW

@dataclass
class DeepSeekMoEConfig:
    """DeepSeekMoE model architecture configuration, Params: [RECALCULATE]"""
    model_type: str = "deepseek" # Identifier

    # --- Core Transformer Args (Tuned for ~124.5M Params) ---
    n_layer: int = 16
    n_embd: int = 640
    n_head: int = 8
    block_size: int = 1024      # Project Constraint
    vocab_size: int = 50304     # Project Constraint

    # --- Attention (MLA) Specific Args (Tuned) ---
    n_kv_head: int = 4          # GQA 2:1 Ratio
    q_lora_rank: int = 256      # Tuned
    kv_lora_rank: int = 64       # Tuned
    v_head_dim: int = 80
    qk_rope_head_dim: int = 40
    qk_nope_head_dim: int = 40
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # --- MoE Specific Args (Aligned closer to paper, scaled down) ---
    n_routed_experts: int = 8   # Tuned (Keep this, paper used 256)
    moe_intermediate_size: int = 288 # Tuned (Keep this, paper used 2048)
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1   # CHANGED from 2 (Matches paper)

    # --- Router Specific Args (Tuned) ---
    n_group: int = 2            # Tuned (Must divide n_routed_experts=8)
    routed_scaling_factor: float = 1.0 # Default
    topk_group: int = 1         # Default
    norm_topk_prob: bool = False # Default

    # --- Aux Loss Coefficients (Aligned closer to paper) ---
    z_loss_coef: float = 0.001 # Keep this for stability
    routing_balance_coef: float = 0.0001 # CHANGED from 0.01 (Matches paper's alpha)

    # --- Normalization & RoPE Args ---
    norm_eps: float = 1e-6      # DeepSeek default
    rope_theta: float = 1000000.0 # DeepSeek default

    # --- Other Architectural Args ---
    hidden_act: str = "silu"    # Standard activation
    initializer_range: float = 0.006 # CHANGED from 0.02 (Matches paper)
    tie_word_embeddings: bool = True
    use_cache: bool = False

    # --- Derived Attributes ---
    qk_head_dim: int = field(init=False)
    num_key_value_heads: int = field(init=False)
    num_key_value_groups: int = field(init=False)
    intermediate_size: Optional[int] = None # For potential dense layers
    first_k_dense_replace: int = 0 # Assume all layers are MoE

    def __post_init__(self):
        """Calculate derived parameters and perform checks."""
        # Calculate full QK head dimension
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        # Set KV heads (already required)
        self.num_key_value_heads = self.n_kv_head

        # Calculate intermediate size for potential dense layers (using 4*D heuristic)
        # Only relevant if first_k_dense_replace > 0
        if self.intermediate_size is None:
            self.intermediate_size = int(4 * self.n_embd)

        # --- Validation Checks ---
        if self.n_embd % self.n_head != 0:
             raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % self.num_key_value_heads != 0:
             raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.num_key_value_heads})")
        if self.n_routed_experts > 0 and self.n_group > 0 and self.n_routed_experts % self.n_group != 0:
             raise ValueError(f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})")
        # Check head dimension consistency (QK)
        if self.qk_head_dim != (self.n_embd // self.n_head):
             # This might be flexible in MLA, but good to be aware
              print(f"Note: Calculated qk_head_dim ({self.qk_head_dim}) differs from n_embd / n_head ({self.n_embd // self.n_head}).")
        # Check V head dim consistency
        if self.v_head_dim != (self.n_embd // self.n_head):
             print(f"Note: v_head_dim ({self.v_head_dim}) differs from n_embd / n_head ({self.n_embd // self.n_head}).")

        self.num_key_value_groups = self.n_head // self.num_key_value_heads

@dataclass
class DeepSeekMoETrainingConfig(BaseTrainingConfig):
    """DeepSeekMoE specific training configuration"""
    # Similar parameters to other models for fair comparison
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 750  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW

    # MoE-specific training parameters
    routing_balance_coef: float = 0.0001  # coefficient for expert balancing loss
    z_loss_coef: float = 0.001  # coefficient for z-loss to stabilize gating


@dataclass
class RWKVConfig:
    """RWKV model architecture configuration, Params: 125,367,552"""
    model_type: str = "rwkv"
    vocab_size: int = 50304
    n_layer: int = 16
    n_embd: int = 576
    ctx_len: int = 1024
    head_size: int = 64
    grad_cp: int = 0
    n_head: int = field(init=False)

    def __post_init__(self):
         if self.n_embd % self.head_size != 0:
             raise ValueError(f"n_embd ({self.n_embd}) must be divisible by head_size ({self.head_size})")
         self.n_head = self.n_embd // self.head_size

@dataclass
class RWKVTrainingConfig(BaseTrainingConfig):
    """RWKV specific training configuration. """
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073

# -------------------------------------------------------------------
# CONFIGS
# -------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    data_root: str = "utils/edu_fineweb10B"  # root directory for data
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
