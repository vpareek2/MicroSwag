from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any

# -------------------------------------------------------------------
# BASE CONFIGS
# -------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Base configuration class for model architectures"""
    model_type: str

    def get_model_specific_config(self):
        """Return the model-specific configuration based on model_type"""
        if self.model_type == "gpt2":
            return GPT2Config()
        elif self.model_type == "llama":
            return LLaMAConfig()
        elif self.model_type == "phi4":
            return Phi4Config()
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
        elif self.model_type == "phi4":
            return Phi4TrainingConfig()
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

# GPT-2 Configs will be the base training hyperparameters for most models, this is meant to test architectures so while these parameters are most likely
# not optimal, they are going to be the baseline. I will tune parameters later once I find the best architecture.
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
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    ffn_dim_multiplier: float = 1.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = False

    # Ensure n_head is divisible by n_kv_head in post_init or check during tuning
    def __post_init__(self):
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")

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

@dataclass
class Gemma3Config:
    """Gemma 3 model architecture configuration"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 262144  # number of tokens (Gemma 3 uses a larger vocabulary)
    n_layer: int = 34  # number of transformer layers for 4B model
    n_head: int = 8  # number of attention heads for 4B model
    n_kv_head: int = 4  # number of key/value heads for grouped query attention
    n_embd: int = 2560  # embedding dimension for 4B model
    norm_eps: float = 1e-5  # epsilon for normalization
    sliding_window: int = 1024  # size of sliding window
    sliding_window_pattern: int = 6  # 5 local layers then 1 global layer
    use_qk_norm: bool = True  # use QK normalization
    rope_local_theta: float = 10000.0  # base for rotary positional embeddings in local layers
    rope_global_theta: float = 1000000.0  # base for rotary positional embeddings in global layers
    rope_scaling: float = 8.0  # scaling factor for RoPE when using extended context

@dataclass
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
    """Mistral model architecture configuration"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304  # number of tokens (same as GPT-2 for compatibility with tokenizer)
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key/value heads for grouped query attention
    n_embd: int = 768  # embedding dimension
    norm_eps: float = 1e-5  # epsilon for normalization
    sliding_window: int = 4096  # size of sliding window attention
    rope_theta: float = 10000.0  # base for rotary positional embeddings
    multiple_of: int = 256  # multiple of for hidden dimension rounding

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
    """DeepSeekMoE model architecture configuration"""
    # --- Core Transformer Args (using names consistent with your other configs) ---
    block_size: int = 4096                   # Renamed from max_position_embeddings
    vocab_size: int = 102400                 # Example value, adjust
    n_layer: int = 30                        # Renamed from num_hidden_layers
    n_head: int = 32                         # Renamed from num_attention_heads
    n_embd: int = 4096                       # Renamed from hidden_size

    # --- Attention (MLA) Specific Args ---
    n_kv_head: Optional[int] = None          # Set to n_head if None in post_init (for MHA/GQA)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    attention_bias: bool = False             # Bias in attention projections
    attention_dropout: float = 0.0

    # --- MoE Specific Args ---
    moe_intermediate_size: int = 1024       # FFN dim inside each expert
    n_routed_experts: int = 60              # Total number of experts
    num_experts_per_tok: int = 2            # Active experts per token
    n_shared_experts: int = 2               # Number of shared experts

    # --- Router Specific Args ---
    routed_scaling_factor: float = 1.0
    n_group: int = 4
    topk_group: int = 1
    norm_topk_prob: bool = False            # Normalize top-k expert weights?

    # --- Aux Loss Coefficients --- ADD THIS SECTION ---
    z_loss_coef: float = 0.001              # <-- ADD THIS LINE
    routing_balance_coef: float = 0.01      # <-- ADD THIS LINE

    # --- Normalization & RoPE Args ---
    norm_eps: float = 1e-6                  # Renamed from rms_norm_eps
    rope_theta: float = 1000000.0           # Base for RoPE
    # rope_scaling: Optional[Dict[str, Any]] = None # Optional: Add later if needed
    # rope_interleave: bool = False         # Optional: Add later if needed

    # --- Other Architectural Args ---
    hidden_act: str = "silu"                # Activation function
    initializer_range: float = 0.02         # Weight initialization std dev
    tie_word_embeddings: bool = False       # Tie input/output embeddings?

    # --- Derived Attributes (calculated after initialization) ---
    qk_head_dim: int = field(init=False)    # Full QK dim: rope + nope
    num_key_value_heads: int = field(init=False) # Actual number of K/V heads used
    num_key_value_groups: int = field(init=False) # Ratio for GQA repeats
    intermediate_size: Optional[int] = None # ADDED: For dense MLP layers
    first_k_dense_replace: int = 0         # ADDED: Control MoE vs Dense layers

    def __post_init__(self):
        """Calculate derived parameters."""
        # Calculate full QK head dimension
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        # Set num_key_value_heads if not provided (defaults to MHA)
        if self.n_kv_head is None:
            self.num_key_value_heads = self.n_head
        else:
             self.num_key_value_heads = self.n_kv_head

        if self.intermediate_size is None:
            # Common heuristic, adjust multiplier if needed
            ffn_dim_multiplier = 4 # Example
            self.intermediate_size = int(ffn_dim_multiplier * self.n_embd)

        # Validate head dimensions and calculate groups for GQA
        if self.n_head % self.num_key_value_heads != 0:
            raise ValueError(
                f"`n_head` ({self.n_head}) must be divisible by "
                f"`num_key_value_heads` ({self.num_key_value_heads})"
            )
        self.num_key_value_groups = self.n_head // self.num_key_value_heads


@dataclass
class DeepSeekMoETrainingConfig(BaseTrainingConfig):
    """DeepSeekMoE specific training configuration"""
    # Similar parameters to other models for fair comparison
    weight_decay: float = 0.1  # weight decay for optimizer
    learning_rate: float = 6e-4  # base learning rate
    min_lr_ratio: float = 0.1  # minimum learning rate as ratio of max lr
    warmup_steps: int = 715  # number of warmup steps
    max_steps: int = 19073  # maximum number of training steps
    betas: Tuple[float, float] = (0.9, 0.95)  # beta parameters for AdamW
    eps: float = 1e-8  # epsilon parameter for AdamW

    # MoE-specific training parameters
    routing_balance_coef: float = 0.01  # coefficient for expert balancing loss
    z_loss_coef: float = 0.001  # coefficient for z-loss to stabilize gating
    expert_dropout: float = 0.1  # dropout rate applied to experts

@dataclass
class RWKVConfig:
    """RWKV model architecture configuration (NanoTitan Adaptation)"""
    model_type: str = "rwkv"    # Identifier for the model type
    vocab_size: int = 50304    # Standard vocab size for the project
    n_layer: int = 12          # Placeholder - Tune later for ~124M params
    n_embd: int = 768          # Placeholder - Tune later for ~124M params
    ctx_len: int = 1024        # Matches BaseTrainingConfig.sequence_length
    head_size: int = 64        # Required by the specific RWKV v7 CUDA kernel used
    grad_cp: int = 0           # Gradient checkpointing flag (0=False, 1=True)

@dataclass
class RWKVTrainingConfig(BaseTrainingConfig):
    """
    RWKV specific training configuration.
    NOTE: For the NanoTitan project's fair comparison goal, these parameters
    are intentionally set to match the standard settings (e.g., GPT2TrainingConfig)
    rather than potentially different defaults from RWKV's original scripts.
    """
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    betas: Tuple[float, float] = (0.9, 0.95) # AdamW betas
    eps: float = 1e-8                       # AdamW epsilon
    # Cosine decay schedule parameters
    min_lr_ratio: float = 0.1   # lr_final = learning_rate * min_lr_ratio
    warmup_steps: int = 715     # Number of linear warmup steps
    max_steps: int = 19073      # Total number of training steps for decay
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
