# calculate_params.py
import math
from config import DeepSeekMoEConfig # Assuming config.py is in the same directory

def calculate_deepseek_params(config: DeepSeekMoEConfig):
    """
    Calculates the estimated total and active parameters for the DeepSeekMoE model
    based on its configuration. Does not instantiate the model.
    """

    # --- Ensure derived attributes are calculated (happens on init) ---
    # config.__post_init__() # Not strictly needed if config is instantiated fresh

    # --- Component Parameter Calculations ---

    # 1. Embeddings (Tied) + Final LM Head (weight is shared)
    params_embedding = config.vocab_size * config.n_embd

    # 2. Attention Block (MLA) - Per Layer
    params_q_a_proj = config.n_embd * config.q_lora_rank
    params_q_b_proj = config.q_lora_rank * (config.n_head * config.qk_head_dim)
    params_kv_a_proj = config.n_embd * (config.kv_lora_rank + config.qk_rope_head_dim)
    params_kv_b_proj = config.kv_lora_rank * (config.num_key_value_heads * (config.qk_nope_head_dim + config.v_head_dim))
    params_o_proj = (config.n_head * config.v_head_dim) * config.n_embd
    params_attn_bias = 0
    if config.attention_bias:
        # Biases for q_a, kv_a, o_proj (assuming no bias on b_proj based on deepseekv3.py)
        params_attn_bias = config.q_lora_rank + (config.kv_lora_rank + config.qk_rope_head_dim) + config.n_embd
    params_attention_per_layer = (
        params_q_a_proj + params_q_b_proj + params_kv_a_proj +
        params_kv_b_proj + params_o_proj + params_attn_bias
    )

    # 3. Normalization Layers - Per Layer
    # RMSNorm weights only (no bias)
    params_input_ln = config.n_embd
    params_post_attn_ln = config.n_embd
    params_q_a_ln = config.q_lora_rank
    params_kv_a_ln = config.kv_lora_rank
    params_norm_per_layer = (
        params_input_ln + params_post_attn_ln + params_q_a_ln + params_kv_a_ln
    )

    # 4. MoE Block - Per Layer
    params_router_gate = config.n_routed_experts * config.n_embd

    # Params for ONE routed expert MLP (gate_proj, up_proj, down_proj weights)
    params_one_routed_expert = (
        (config.n_embd * config.moe_intermediate_size) + # gate_proj
        (config.n_embd * config.moe_intermediate_size) + # up_proj
        (config.moe_intermediate_size * config.n_embd)   # down_proj
    )
    params_all_routed_experts_per_layer = config.n_routed_experts * params_one_routed_expert

    # Params for Shared Expert MLP (if exists)
    params_shared_expert_per_layer = 0
    if config.n_shared_experts > 0:
        shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        params_shared_expert_per_layer = (
            (config.n_embd * shared_intermediate_size) + # gate_proj
            (config.n_embd * shared_intermediate_size) + # up_proj
            (shared_intermediate_size * config.n_embd)   # down_proj
        )

    params_moe_block_per_layer = (
        params_router_gate + params_all_routed_experts_per_layer + params_shared_expert_per_layer
    )

    # 5. Final Normalization Layer
    params_final_norm = config.n_embd

    # --- Total Parameters ---
    # Note: MoE block replaces a dense MLP, so we add params_moe_block_per_layer
    total_params = (
        params_embedding + # Includes LM head due to tying
        config.n_layer * (
            params_attention_per_layer +
            params_norm_per_layer +
            params_moe_block_per_layer # Use MoE block params here
        ) +
        params_final_norm
    )

    # --- Active Parameters ---
    # Start with non-routed parameters
    active_params_base = (
        params_embedding +
        config.n_layer * (
            params_attention_per_layer +
            params_norm_per_layer +
            params_router_gate + # Router is always active
            params_shared_expert_per_layer # Shared expert is always active
        ) +
        params_final_norm
    )

    # Add the parameters from the activated routed experts
    active_params_routed = config.n_layer * (params_one_routed_expert * config.num_experts_per_tok)
    total_active_params = active_params_base + active_params_routed


    return int(total_params), int(total_active_params)

if __name__ == "__main__":
    print("Calculating parameters for DeepSeekMoEConfig...")

    # --- Configuration to test ---
    # Option 1: Use the default from config.py
    # config = DeepSeekMoEConfig()

    # Option 2: Define explicitly or modify defaults here if needed
    # (e.g., to test the paper's k=8 vs your k=2)
    config = DeepSeekMoEConfig(
        # --- Set parameters exactly as in your config.py ---
        n_layer = 16,
        n_embd = 640,
        n_head = 8,
        block_size = 1024,
        vocab_size = 50304,
        n_kv_head = 4,
        q_lora_rank = 256,
        kv_lora_rank = 64,
        v_head_dim = 80,
        qk_rope_head_dim = 40,
        qk_nope_head_dim = 40,
        attention_bias = False,
        attention_dropout = 0.0,
        n_routed_experts = 8,
        moe_intermediate_size = 288,
        num_experts_per_tok = 2,
        n_shared_experts = 1,    # <<< Use 1 based on paper/previous discussion
        n_group = 2,
        routed_scaling_factor = 1.0,
        topk_group = 1,
        norm_topk_prob = False,
        z_loss_coef = 0.001,
        routing_balance_coef = 0.0001, # <<< Use paper's alpha
        norm_eps = 1e-6,
        rope_theta = 1000000.0,
        hidden_act = "silu",
        initializer_range = 0.006, # <<< Use paper's init range
        tie_word_embeddings = True,
        use_cache = False,
        first_k_dense_replace = 0
        # Add any other relevant fields if your config changes
    )

    # --- Perform Calculation ---
    total_params, active_params = calculate_deepseek_params(config)

    # --- Print Results ---
    print("\n--- Configuration Used ---")
    for field_name, field_value in config.__dict__.items():
         # Exclude derived fields shown in __post_init__ print for brevity
         if field_name not in ['qk_head_dim', 'num_key_value_heads', 'num_key_value_groups', 'intermediate_size']:
              print(f"  {field_name}: {field_value}")

    print("\n--- Calculated Parameters ---")
    print(f"  Total Parameters: {total_params:,} (~{total_params/1e6:.2f} M)")
    print(f"  Active Parameters: {active_params:,} (~{active_params/1e6:.2f} M)")
    print(f"  (Based on num_experts_per_tok = {config.num_experts_per_tok})")
