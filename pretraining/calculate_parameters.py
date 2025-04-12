import math
from dataclasses import dataclass, field
from typing import Optional, Tuple # Added Tuple

# --- Fixed Project Constraints ---
TARGET_VOCAB_SIZE = 50304
TARGET_BLOCK_SIZE = 1024
TARGET_PARAMS = 124_500_000
PARAM_TOLERANCE = 4_000_000 # +/- ~3% tolerance

# --- DeepSeek Architectural Defaults ---
DS_NORM_EPS = 1e-6
DS_ATTENTION_BIAS = False
DS_HIDDEN_ACT = "silu"
DS_TIE_EMBEDDINGS = True
DS_NUM_EXPERTS_PER_TOK = 2 # Default k value
DS_ROUTED_SCALING_FACTOR = 1.0
DS_TOPK_GROUP = 1
DS_NORM_TOPK_PROB = False
DS_ATTN_DROPOUT = 0.0
DS_FIRST_K_DENSE = 0

# --- Configuration Dataclass for Calculation ---
# In find_deepseek_params.py

# --- Configuration Dataclass for Calculation ---
@dataclass
class DeepSeekMoEConfigCalc:
    # --- Fields WITHOUT Defaults (Must be provided by search loop) ---
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    n_group: int

    # --- Fields WITH Defaults ---
    vocab_size: int = TARGET_VOCAB_SIZE
    block_size: int = TARGET_BLOCK_SIZE
    attention_bias: bool = DS_ATTENTION_BIAS
    num_experts_per_tok: int = DS_NUM_EXPERTS_PER_TOK
    routed_scaling_factor: float = DS_ROUTED_SCALING_FACTOR
    topk_group: int = DS_TOPK_GROUP
    norm_topk_prob: bool = DS_NORM_TOPK_PROB
    norm_eps: float = DS_NORM_EPS
    rope_theta: float = 1000000.0
    hidden_act: str = DS_HIDDEN_ACT
    tie_word_embeddings: bool = DS_TIE_EMBEDDINGS
    first_k_dense_replace: int = DS_FIRST_K_DENSE
    # attention_dropout: float = DS_ATTN_DROPOUT # Not needed for param count

    # --- Derived Fields ---
    qk_head_dim: int = field(init=False)
    num_key_value_heads: int = field(init=False)

    def __post_init__(self):
        # Calculate full QK head dimension
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        # Set KV heads (already required)
        self.num_key_value_heads = self.n_kv_head

        # --- VALIDATION CHECKS ---
        if self.n_embd % self.n_head != 0:
             raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % self.num_key_value_heads != 0:
             raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.num_key_value_heads})")
        if self.n_routed_experts > 0 and self.n_group > 0 and self.n_routed_experts % self.n_group != 0:
             raise ValueError(f"n_routed_experts ({self.n_routed_experts}) must be divisible by n_group ({self.n_group})")
        # Check head dimension consistency
        # if self.n_head * self.v_head_dim != self.n_embd: # This check might be too strict for MLA
        #      print(f"Warning: H*VHD ({self.n_head * self.v_head_dim}) != D ({self.n_embd}). Check o_proj.")

# --- Parameter Calculation Function ---
def calculate_deepseek_moe_params(config: DeepSeekMoEConfigCalc, verbose: bool = False) -> Tuple[int, int]: # Return Tuple
    """Calculates total AND active trainable parameters for DeepSeekMoE."""

    total_params = 0
    params_per_layer_list = []
    total_routed_expert_params_all_layers = 0
    params_mlp_expert = 0 # Define outside loop

    # --- Shared Components ---
    params_wte = config.vocab_size * config.n_embd
    total_params += params_wte
    if verbose: print(f"  Embeddings (wte/tied head): {params_wte:,}")
    params_ln_f = config.n_embd
    total_params += params_ln_f
    if verbose: print(f"  Final Norm (ln_f): {params_ln_f:,}")

    # --- Calculate Params for ONE Routed Expert MLP ---
    if config.n_routed_experts > 0:
        params_mlp_expert = ( (config.n_embd * config.moe_intermediate_size) * 2 + # gate + up
                              (config.moe_intermediate_size * config.n_embd)       # down
                            ) # Assuming no bias

    # --- Calculate Params for ONE Shared Expert MLP ---
    params_mlp_shared = 0
    if config.n_shared_experts > 0:
        shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        params_mlp_shared = ( (config.n_embd * shared_intermediate_size) * 2 + # gate + up
                              (shared_intermediate_size * config.n_embd)       # down
                            ) # Assuming no bias

    # --- Calculate Params for ONE Router Gate ---
    params_router_gate = 0
    if config.n_routed_experts > 0:
         params_router_gate = config.n_routed_experts * config.n_embd # No bias

    # --- Per-Layer Components ---
    for layer_idx in range(config.n_layer):
        params_this_layer = 0
        is_moe_layer = layer_idx >= config.first_k_dense_replace

        # 1. Norms
        params_norms_per_layer = 2 * config.n_embd
        params_this_layer += params_norms_per_layer

        # 2. Attention
        params_attn_q_a = config.n_embd * config.q_lora_rank + (config.q_lora_rank if config.attention_bias else 0)
        params_attn_q_a_norm = config.q_lora_rank
        params_attn_q_b = config.q_lora_rank * (config.n_head * config.qk_head_dim)
        params_attn_kv_a = config.n_embd * (config.kv_lora_rank + config.qk_rope_head_dim) + \
                           ( (config.kv_lora_rank + config.qk_rope_head_dim) if config.attention_bias else 0)
        params_attn_kv_a_norm = config.kv_lora_rank
        params_attn_kv_b = config.kv_lora_rank * (config.num_key_value_heads * (config.qk_nope_head_dim + config.v_head_dim))
        params_attn_o = (config.n_head * config.v_head_dim) * config.n_embd + (config.n_embd if config.attention_bias else 0)
        params_attn_block = (params_attn_q_a + params_attn_q_a_norm + params_attn_q_b +
                             params_attn_kv_a + params_attn_kv_a_norm + params_attn_kv_b +
                             params_attn_o)
        params_this_layer += params_attn_block

        # 3. MoE / Dense Block
        if is_moe_layer:
            params_moe_block_this_layer = params_router_gate + \
                                          (config.n_routed_experts * params_mlp_expert) + \
                                          params_mlp_shared
            params_this_layer += params_moe_block_this_layer
            # Accumulate total expert params for active calculation later
            total_routed_expert_params_all_layers += config.n_routed_experts * params_mlp_expert
        else:
            # Dense MLP (use fallback calculation if needed)
            dense_intermediate_size = getattr(config, 'intermediate_size', int(4 * config.n_embd))
            params_mlp_dense = ( (config.n_embd * dense_intermediate_size) * 2 + # gate + up
                                 (dense_intermediate_size * config.n_embd)       # down
                               )
            params_this_layer += params_mlp_dense

        params_per_layer_list.append(params_this_layer)

    # Sum total parameters
    total_params += sum(params_per_layer_list)

    # --- Calculate Active Parameters ---
    # Start with total params and subtract the inactive experts
    # Inactive = (Total Routed Experts - Active Routed Experts per token) * Params per Expert * Num MoE Layers
    num_moe_layers = config.n_layer - config.first_k_dense_replace
    params_inactive_experts = num_moe_layers * (config.n_routed_experts - config.num_experts_per_tok) * params_mlp_expert
    total_active_params = total_params - params_inactive_experts

    if verbose:
        print(f"  Total Params Calculated: {total_params:,}")
        print(f"  Total Routed Expert Params (all layers): {total_routed_expert_params_all_layers:,}")
        print(f"  Inactive Expert Params (all layers): {params_inactive_experts:,}")
        print(f"  Active Params Calculated: {total_active_params:,}")

    return total_params, total_active_params # Return both

# --- Search Script ---

print(f"Target TOTAL parameters: {TARGET_PARAMS:,} (+/- {PARAM_TOLERANCE:,})")
print(f"Fixed vocab_size: {TARGET_VOCAB_SIZE}, block_size: {TARGET_BLOCK_SIZE}")
print(f"Assuming tied_word_embeddings = {DS_TIE_EMBEDDINGS}")
print("-" * 40)

results = []

# --- Define Search Ranges (Adjust as needed) ---
n_layer_range = [10, 12, 14, 16]
n_embd_range = [640, 768]
n_head_range = [8, 12]
# Fixed head dims relative to n_embd/n_head
# v_head_dim = n_embd // n_head
# qk_rope_head_dim = v_head_dim // 2
# qk_nope_head_dim = v_head_dim - qk_rope_head_dim
q_lora_rank_range = [128, 256]
kv_lora_rank_range = [64, 128]
n_routed_experts_range = [8, 16]
moe_intermediate_range = [256, 512]
n_shared_experts_range = [0, 2]
n_group_range = [2, 4]

# --- Nested Loops ---
print("Starting search...")
count = 0
found_count = 0

for n_layer in n_layer_range:
    for n_embd in n_embd_range:
     for n_head in n_head_range:
      if n_embd % n_head != 0: continue
      v_head_dim = n_embd // n_head
      qk_rope_head_dim = v_head_dim // 2
      qk_nope_head_dim = v_head_dim - qk_rope_head_dim

      for n_kv_head_divisor in [2, 4]: # GQA ratios
        n_kv_head = n_head // n_kv_head_divisor
        if n_head % n_kv_head != 0: continue

        for q_lora_rank in q_lora_rank_range:
         for kv_lora_rank in kv_lora_rank_range:
          for n_routed_experts in n_routed_experts_range:
           for moe_intermediate_size in moe_intermediate_range:
            for n_shared_experts in n_shared_experts_range:
             for n_group in n_group_range:
              if n_routed_experts % n_group != 0: continue

              count += 1
              if count % 100 == 0: print(f"Checked {count} configs...")

              try:
                  config = DeepSeekMoEConfigCalc(
                      n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head,
                      q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
                      qk_rope_head_dim=qk_rope_head_dim, qk_nope_head_dim=qk_nope_head_dim,
                      v_head_dim=v_head_dim,
                      moe_intermediate_size=moe_intermediate_size,
                      n_routed_experts=n_routed_experts, n_shared_experts=n_shared_experts,
                      n_group=n_group
                  )
                  # Calculate both total and active params
                  total_params, active_params = calculate_deepseek_moe_params(config) # Unpack tuple

                  # Filter based on TOTAL parameters
                  if abs(total_params - TARGET_PARAMS) <= PARAM_TOLERANCE:
                      results.append({
                          "total_params": total_params,
                          "active_params": active_params, # Store active params
                          "config": config,
                          "diff": total_params - TARGET_PARAMS
                      })
                      found_count += 1
                      print(f"  ---> Found {found_count}: Total={total_params:,} Active={active_params:,} (Diff={total_params - TARGET_PARAMS:,})")
                      # Print config details... (shortened for brevity)
                      print(f"       Config: L={n_layer}, D={n_embd}, H={n_head}, KVH={n_kv_head}, NExp={n_routed_experts}, MoeD={moe_intermediate_size} ...")

              except ValueError as e: pass

# --- Report Results ---
print("\n" + "=" * 40)
print(f"Search Complete. Checked {count} configurations.")
print("DeepSeek MoE Configurations within tolerance (based on TOTAL params):")
print("=" * 40)
if not results:
    print(f"No DeepSeek MoE configurations found within {PARAM_TOLERANCE:,} of {TARGET_PARAMS:,}.")
    print("Consider adjusting search ranges.")
else:
    results.sort(key=lambda x: abs(x['diff'])) # Sort by diff in TOTAL params
    for result in results:
        cfg = result['config']
        print(f"Total Params: {result['total_params']:,} (Diff: {result['diff']:,})")
        print(f"Active Params: {result['active_params']:,}") # Print active params
        print(f"  Config: L={cfg.n_layer}, D={cfg.n_embd}, H={cfg.n_head}, KVH={cfg.n_kv_head}, "
              f"QLR={cfg.q_lora_rank}, KVLR={cfg.kv_lora_rank}, RoHD={cfg.qk_rope_head_dim}, NoHD={cfg.qk_nope_head_dim}, VHD={cfg.v_head_dim}, "
              f"NExp={cfg.n_routed_experts}, MoeD={cfg.moe_intermediate_size}, NShared={cfg.n_shared_experts}, NGrp={cfg.n_group}")
        print("-" * 20)

    print(f"\nClosest DeepSeek MoE configuration (by TOTAL params):")
    best_result = results[0]
    cfg = best_result['config']
    print(f"Total Params: {best_result['total_params']:,} (Diff: {best_result['diff']:,})")
    print(f"Active Params: {best_result['active_params']:,}") # Print active params for best total
    print(f"  Config: L={cfg.n_layer}, D={cfg.n_embd}, H={cfg.n_head}, KVH={cfg.n_kv_head}, "
          f"QLR={cfg.q_lora_rank}, KVLR={cfg.kv_lora_rank}, RoHD={cfg.qk_rope_head_dim}, NoHD={cfg.qk_nope_head_dim}, VHD={cfg.v_head_dim}, "
          f"NExp={cfg.n_routed_experts}, MoeD={cfg.moe_intermediate_size}, NShared={cfg.n_shared_experts}, NGrp={cfg.n_group}")
    print("=" * 40)
