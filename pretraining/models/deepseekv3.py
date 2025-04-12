"""DeepSeekV3 as per https://arxiv.org/pdf/2412.19437"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Any
from config import DeepSeekMoEConfig
import inspect
import math

class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Ensure computation in float32 for stability
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast back to original type after norm and scaling
        output = self._norm(x).type_as(x)
        return output * self.weight

# --- Rename MLP to DeepseekV3MLP for clarity ---
class DeepseekV3MLP(nn.Module): # Renamed from MLP
    def __init__(self, config: DeepSeekMoEConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        # Use config.n_embd as it's defined in your DeepSeekMoEConfig now
        self.hidden_size = config.n_embd if hidden_size is None else hidden_size
        # Default to config.moe_intermediate_size if used for experts, or config.intermediate_size for dense
        default_intermediate = config.moe_intermediate_size if intermediate_size is None else intermediate_size
        self.intermediate_size = default_intermediate

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Use hidden_act from config
        if config.hidden_act == "silu":
            self.act_fn = F.silu
        # Add other activations if needed (e.g., elif config.hidden_act == "gelu": self.act_fn = F.gelu)
        else:
            # Default or raise error
            self.act_fn = F.silu # Defaulting to silu if not specified or matched
            print(f"Warning: Activation function '{config.hidden_act}' not explicitly handled, defaulting to SiLU.")


    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# --- Rename RoPE to DeepseekV3RotaryEmbedding ---
class DeepseekV3RotaryEmbedding(nn.Module): # Renamed from RoPE
    def __init__(self, config: DeepSeekMoEConfig, device=None):
        super().__init__()
        self.config = config
        self.dim = config.qk_rope_head_dim # Dimension RoPE is applied to
        # Use config.block_size instead of max_position_embeddings
        self.max_seq_len_cached = config.block_size
        self.base = config.rope_theta

        # Calculate inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # TODO: Implement scaling logic from HF if needed (e.g., YaRN mscale)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, head_dim] or similar, used for device/dtype
        # position_ids: [bs, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [bs, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [bs, 1, seq_len]

        # Force float32 for precision
        # Use x.device.type directly
        device_type = x.device.type
        if device_type == 'mps': # MPS doesn't support autocast float32 well
             device_type = 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2) # [bs, seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # [bs, seq_len, dim]
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Re-paste helper functions for completeness
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# RoPE application function (mirrors HF BUT handles None inputs)
def apply_rotary_pos_emb(q: Optional[torch.Tensor],
                         k: Optional[torch.Tensor],
                         cos: torch.Tensor,
                         sin: torch.Tensor,
                         position_ids=None, # Unused, kept for compatibility maybe
                         unsqueeze_dim=1
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Applies RoPE to q and/or k tensors."""
    # cos/sin: [B, S, D] or [1, S, D] after slicing
    # Unsqueeze for broadcasting: [B, 1, S, D] or [1, 1, S, D]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = None
    if q is not None:
        # Ensure sequence lengths match before applying RoPE
        # Slicing of cos/sin should handle this before calling the function
        if q.shape[2] != cos.shape[2]:
             raise ValueError(f"Sequence length mismatch for RoPE Q: q.shape[2]={q.shape[2]}, cos.shape[2]={cos.shape[2]}")
        q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = None
    if k is not None:
        if k.shape[2] != cos.shape[2]:
             raise ValueError(f"Sequence length mismatch for RoPE K: k.shape[2]={k.shape[2]}, cos.shape[2]={cos.shape[2]}")
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class DeepseekV3Attention(nn.Module):
    """ DeepSeek V3 Multi-Head Latent Attention (MLA) module with KV Caching."""
    def __init__(self, config: DeepSeekMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.n_embd

        # Ensure derived config attributes are calculated
        if not hasattr(config, 'num_key_value_heads'): config.__post_init__()

        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim_q = config.qk_head_dim
        self.head_dim_v = config.v_head_dim
        self.rope_head_dim = config.qk_rope_head_dim # Explicitly store RoPE dim

        self.attention_dropout = config.attention_dropout

        # === Q Projections ===
        self.q_a_proj = nn.Linear(config.n_embd, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim_q, bias=False)

        # === KV Projections ===
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_key_value_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )

        # === Output Projection ===
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim_v,
            config.n_embd,
            bias=config.attention_bias,
        )

        self.scaling = self.head_dim_q ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) -> [1, MaxSeqLen, rope_dim]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,  # Absolute positions [SeqLen]
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the DeepseekV3Attention."""

        batch_size, seq_length, _ = hidden_states.shape
        if cache_position is None:
            raise ValueError("cache_position must be provided for attention calculation and caching.")

        # --- Q Pathway ---
        q_a = self.q_a_proj(hidden_states)
        q_a_norm = self.q_a_layernorm(q_a)
        q_states = self.q_b_proj(q_a_norm)
        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.head_dim_q).transpose(1, 2)  # [B, H, S, D_qk]
        q_pass, q_rot = torch.split(q_states, [self.config.qk_nope_head_dim, self.rope_head_dim], dim=-1)

        # --- KV Pathway ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_lora_rank + rope_dim]
        k_pass_a, k_rot_a = torch.split(compressed_kv, [self.config.kv_lora_rank, self.rope_head_dim], dim=-1)  # k_rot_a: [B, S, rope_dim]
        k_pass_a_norm = self.kv_a_layernorm(k_pass_a)
        kv_pass_b = self.kv_b_proj(k_pass_a_norm)
        kv_pass_b = kv_pass_b.view(batch_size, seq_length, self.num_key_value_heads,
                                     self.config.qk_nope_head_dim + self.config.v_head_dim).transpose(1, 2)  # [B, H_kv, S, D_nope+D_v]
        k_pass, value_states_current = torch.split(kv_pass_b, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)  # k_pass: [B, H_kv, S, D_nope]

        # --- Apply RoPE ---
        cos, sin = position_embeddings  # [1, MaxSeqLen, rope_dim]
        # Slice cos/sin based on the absolute positions of the current tokens
        cos = cos.squeeze(0)[cache_position].unsqueeze(0)  # [1, S, rope_dim]
        sin = sin.squeeze(0)[cache_position].unsqueeze(0)  # [1, S, rope_dim]

        # Apply to Q
        q_rot, _ = apply_rotary_pos_emb(q=q_rot, k=None, cos=cos, sin=sin)

        # Apply to K's RoPE part (k_rot_a)
        # Reshape k_rot_a to [B, 1, S, D_rope] for RoPE application standard
        k_rot_a_reshaped = k_rot_a.view(batch_size, seq_length, 1, self.rope_head_dim).transpose(1, 2)  # [B, 1, S, D_rope]
        _, k_rot = apply_rotary_pos_emb(q=None, k=k_rot_a_reshaped, cos=cos, sin=sin)  # k_rot is now [B, 1, S, D_rope] with RoPE applied

        # Expand K RoPE part to match number of KV heads
        k_rot = k_rot.expand(batch_size, self.num_key_value_heads, seq_length, self.rope_head_dim)  # [B, H_kv, S, D_rope]

        # --- Combine K parts (current step only) ---
        key_states_current = torch.cat((k_pass, k_rot), dim=-1)  # [B, H_kv, S, D_qk]

        # Repeat key_states_current and value_states_current to match query heads if necessary
        if self.num_heads > self.num_key_value_heads:
            num_key_value_groups = self.num_heads // self.num_key_value_heads
            key_states = key_states_current.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states_current.repeat_interleave(num_key_value_groups, dim=1)
        else:
            key_states = key_states_current
            value_states = value_states_current

        # --- KV Caching ---
        if past_key_value is not None:
            cached_key, cached_value = past_key_value
            # If caching, you might need to adjust cached_key and cached_value similarly.
            key_states = torch.cat([cached_key, key_states], dim=2)
            value_states = torch.cat([cached_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # --- Attention Calculation ---
        # Combine Q parts AFTER RoPE
        query_states = torch.cat((q_pass, q_rot), dim=-1)  # [B, H, S, D_qk]

        is_causal_sdpa = attention_mask is None and query_states.shape[2] > 1

        attn_output = F.scaled_dot_product_attention(
            query=query_states,   # [B, n_head, S, d_q]
            key=key_states,       # [B, n_head, S, d_q] after repeating
            value=value_states,   # [B, n_head, S, d_v] after repeating
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal_sdpa
        )

        # --- Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * self.head_dim_v)
        attn_output = self.o_proj(attn_output)

        # --- Return ---
        attn_weights = None
        if output_attentions:
            print("Warning: output_attentions=True is not efficiently supported.")

        return attn_output, attn_weights, present_key_value


# --- Helper function for manual attention calculation if needed ---
# You would need this if you implement the output_attentions fallback
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Expands K/V heads for Grouped Query Attention.
    Input: [batch, num_key_value_heads, seqlen, head_dim]
    Output: [batch, num_attention_heads, seqlen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class DeepseekV3TopkRouter(nn.Module):
    """
    Expert router using Top-k gating with optional grouping.

    Mimics the structure from Hugging Face's implementation.
    """
    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        # Get routing parameters from config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        # Validate grouping parameters
        if self.n_routed_experts % self.n_group != 0:
                raise ValueError(f"`n_routed_experts` ({self.n_routed_experts}) must be divisible by `n_group` ({self.n_group})")
        self.experts_per_group = self.n_routed_experts // self.n_group

        # Routing weights (Linear layer)
        # Input: hidden_size (n_embd), Output: n_routed_experts
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.n_embd)))
        # Initialize weights (using config's initializer_range might be good)
        # Example initialization (match HF's _init_weights if possible)
        std = config.initializer_range
        self.weight.data.normal_(mean=0.0, std=std)

        # HF uses a buffer for 'e_score_correction_bias', often zero. Let's include it.
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))


    @torch.no_grad() # This complex indexing logic shouldn't require gradients
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Calculates top-k expert indices using the grouping strategy.
        Args:
            scores: Sigmoid scores after routing projection. Shape [num_tokens, n_routed_experts]
        Returns:
            topk_indices: Indices of the selected experts. Shape [num_tokens, top_k]
        """
        # Ensure scores are flat [num_tokens, n_experts]
        scores_flat = scores.view(-1, self.n_routed_experts)
        num_tokens = scores_flat.shape[0]

        # Add correction bias (usually zero)
        scores_for_choice = scores_flat + self.e_score_correction_bias.unsqueeze(0)

        # --- Grouping Logic ---
        # Reshape for groups: [num_tokens, n_group, experts_per_group]
        scores_grouped = scores_for_choice.view(num_tokens, self.n_group, self.experts_per_group)

        # Get top-2 scores within each group (HF implementation uses top-2 sum for group selection)
        # Summing top-2 scores per group: [num_tokens, n_group]
        group_scores_sum_top2 = scores_grouped.topk(k=2, dim=-1)[0].sum(dim=-1)

        # Select top `topk_group` groups based on these summed scores
        # Shape: [num_tokens, topk_group]
        top_group_indices = torch.topk(group_scores_sum_top2, k=self.topk_group, dim=-1, sorted=False)[1]

        # Create a mask for selected groups
        # Shape: [num_tokens, n_group]
        group_mask = torch.zeros_like(group_scores_sum_top2, dtype=torch.bool) # Use bool for efficiency
        # Scatter True values at the indices of the selected top groups
        group_mask.scatter_(1, top_group_indices, True)

        # Expand group mask back to expert dimension
        # Shape: [num_tokens, n_group, experts_per_group] -> [num_tokens, n_routed_experts]
        expert_mask_from_groups = group_mask.unsqueeze(-1).expand(num_tokens, self.n_group, self.experts_per_group)
        expert_mask_from_groups = expert_mask_from_groups.reshape(num_tokens, self.n_routed_experts)

        # --- Final Top-k Selection ---
        # Mask out scores of experts belonging to non-selected groups
        # Use a very small number (or -inf for float) instead of 0.0 for masked_fill
        # This prevents selecting masked experts if their original score was 0.0
        # Since scores are sigmoids (0 to 1), 0.0 is a valid minimum. -1.0 works.
        scores_masked = scores_for_choice.masked_fill(~expert_mask_from_groups, -1.0) # Use a value clearly outside [0,1]

        # Select final top-k experts from the masked scores
        # Shape: [num_tokens, top_k]
        topk_indices = torch.topk(scores_masked, k=self.top_k, dim=-1, sorted=False)[1]

        return topk_indices

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routes input hidden states to experts.

        Args:
            hidden_states: Input tensor. Shape [batch_size, seq_len, hidden_size]

        Returns:
            topk_indices: Indices of selected experts. Shape [*, top_k] (flattened batch/seq dims)
            topk_weights: Weights for selected experts. Shape [*, top_k]
            router_logits: Raw logits before sigmoid. Shape [*, n_routed_experts]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Flatten input tokens: [batch_size * seq_len, hidden_size]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # Project to get routing logits - ensure calculation in float32 for stability
        # Shape: [num_tokens, n_routed_experts]
        router_logits = F.linear(hidden_states_flat.float(), self.weight.float())

        # Calculate scores using sigmoid
        # Shape: [num_tokens, n_routed_experts]
        scores = router_logits.sigmoid()

        # Get top-k indices using the grouping logic
        # Shape: [num_tokens, top_k]
        topk_indices = self.get_topk_indices(scores)

        # Gather the scores (weights) corresponding to the selected experts
        # Shape: [num_tokens, top_k]
        topk_weights = scores.gather(dim=1, index=topk_indices)

        # Optional: Normalize the weights of the selected experts
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20 # Add epsilon for stability
            topk_weights = topk_weights / denominator

        # Apply scaling factor
        topk_weights = topk_weights * self.routed_scaling_factor

        # Return flat indices/weights and original logits (casted back to input dtype)
        return topk_indices, topk_weights.to(hidden_states.dtype), router_logits.to(hidden_states.dtype)

class DeepseekV3MoE(nn.Module):
    """
    Mixture of Experts layer for DeepSeek V3.

    Combines a router, routed experts, and optional shared experts.
    Calculates auxiliary losses during training.
    """
    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # --- Router ---
        self.gate = DeepseekV3TopkRouter(config)

        # --- Routed Experts ---
        self.experts = nn.ModuleList(
            [DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
             for _ in range(self.n_routed_experts)]
        )

        # --- Shared Experts ---
        self.n_shared_experts = config.n_shared_experts
        if self.n_shared_experts > 0:
            # The intermediate size for shared experts is scaled
            shared_intermediate_size = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config, intermediate_size=shared_intermediate_size)
        else:
            self.shared_experts = None

        # --- Aux Loss Coefficients ---
        self.z_loss_coef = config.z_loss_coef
        self.routing_balance_coef = config.routing_balance_coef

    def compute_balance_loss(self, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculate auxiliary load balancing loss based on router outputs.
        Uses the logic similar to the original implementation provided.

        Args:
            topk_weights: Weights for selected experts. Shape [num_tokens, top_k]
            topk_indices: Indices of selected experts. Shape [num_tokens, top_k]

        Returns:
            Balance loss (scalar tensor).
        """
        num_tokens, k = topk_weights.shape
        if num_tokens == 0:
             return torch.tensor(0.0, device=topk_weights.device, dtype=topk_weights.dtype)

        # Create one-hot encoding for selected experts for each token/choice pair
        # Shape: [num_tokens, top_k, n_routed_experts]
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts).to(torch.float32) # Use float32 for accumulation

        # Calculate load per expert: sum of weights assigned to each expert across all tokens/choices
        # Shape: [num_tokens, top_k, n_routed_experts]
        load_per_token_expert_choice = expert_mask * topk_weights.unsqueeze(-1).float()

        # Mean load per expert: Average weight assigned to each expert across all tokens
        # Sum over the k choices, then average over the tokens
        # Shape: [n_routed_experts]
        mean_load_per_expert = load_per_token_expert_choice.sum(dim=1).mean(dim=0)

        # Calculate dispatch frequency per expert: fraction of times each expert was chosen
        # Sum over the k choices (counts how many times an expert was chosen for a token, max k)
        # Then average over the tokens
        # Shape: [n_routed_experts]
        dispatch_count_per_expert = expert_mask.sum(dim=(0, 1)) # Total count for each expert
        # Avoid division by zero if an expert is never chosen (shouldn't happen with k>0)
        # Normalizing by total potential dispatches (num_tokens * k) gives frequency?
        # Let's match the original formula: mean(sum over k choices)
        dispatch_freq_per_expert = expert_mask.sum(dim=1).float().mean(dim=0) # Fraction of tokens * k that went to expert i


        # Calculate balance loss using the formula: n_experts * sum(mean_load * dispatch_freq)
        balance_loss = self.n_routed_experts * torch.sum(mean_load_per_expert * dispatch_freq_per_expert)

        return balance_loss.to(topk_weights.dtype) # Cast back to original dtype


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MoE layer.

        Args:
            hidden_states: Input tensor. Shape [batch_size, seq_len, hidden_size]

        Returns:
            final_output: Output tensor after MoE. Shape [batch_size, seq_len, hidden_size]
            aux_loss: Combined auxiliary loss (z-loss + balance loss) during training, else None.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Preserve original input for residual connection with shared experts
        residuals = hidden_states
        # Flatten input for routing and expert processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # --- Routing ---
        # Get routing decisions: indices [num_tokens, k], weights [num_tokens, k], logits [num_tokens, E]
        topk_indices, topk_weights, router_logits = self.gate(hidden_states) # Use hidden_states (unflattened) as input to router

        # --- Expert Dispatch and Computation ---
        # Initialize output tensor
        final_hidden_states_flat = torch.zeros_like(hidden_states_flat)

        # Combine expert indices and token indices for efficient processing (optional, simple loop is clearer)
        # Flat expert indices: [num_tokens * k]
        flat_expert_indices = topk_indices.view(-1)
        # Corresponding token indices: [num_tokens * k]
        token_indices_for_experts = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.top_k)
        # Flat weights: [num_tokens * k]
        flat_topk_weights = topk_weights.view(-1)

        # Process each expert
        for expert_idx in range(self.n_routed_experts):
            # Find which flattened token/k pairs selected this expert
            mask = (flat_expert_indices == expert_idx)
            selected_indices = torch.where(mask)[0]

            if selected_indices.numel() > 0:
                # Get the original token indices corresponding to these selections
                original_token_pos = token_indices_for_experts[selected_indices]

                # Get the inputs for this expert
                expert_inputs = hidden_states_flat[original_token_pos]

                # Compute expert output
                expert_outputs = self.experts[expert_idx](expert_inputs)

                # Get the weights for these specific inputs
                expert_weights = flat_topk_weights[selected_indices]

                # Weight the outputs and add to the final tensor at the correct token positions
                # Use index_add_ for sparse addition (handles multiple experts per token)
                final_hidden_states_flat.index_add_(0, original_token_pos, expert_outputs * expert_weights.unsqueeze(-1))


        # Reshape output back to [B, S, D]
        routed_output = final_hidden_states_flat.view(batch_size, seq_len, hidden_dim)

        # --- Add Shared Expert Output ---
        final_output = routed_output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals) # Apply shared experts to original input
            final_output = final_output + shared_output

        # --- Auxiliary Loss Calculation (Training Only) ---
        aux_loss = None
        if self.training and (self.z_loss_coef > 0 or self.routing_balance_coef > 0):
            # Calculate z-loss (encourages router logits to be spread out)
            logsumexp_logits = torch.logsumexp(router_logits, dim=-1)
            z_loss = torch.mean(logsumexp_logits.float()**2) # Calculate in float32

            # Calculate balance loss
            balance_loss = self.compute_balance_loss(topk_weights, topk_indices)

            # Combine losses
            total_aux_loss = (self.routing_balance_coef * balance_loss +
                              self.z_loss_coef * z_loss)
            aux_loss = total_aux_loss.to(hidden_states.dtype) # Cast back to original dtype

        return final_output, aux_loss

class DeepseekV3DecoderLayer(nn.Module):
    """
    A single layer for the DeepSeek V3 Transformer model.

    Includes self-attention and either a dense MLP or a Mixture-of-Experts layer,
    with residual connections and layer normalization.
    """
    def __init__(self, config: DeepSeekMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.n_embd # Use config name

        # --- Layer Normalization before Attention ---
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.norm_eps)

        # --- Self Attention ---
        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        # --- Layer Normalization before MLP/MoE ---
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.norm_eps)

        # --- MLP or MoE Layer ---
        # Check if a config param exists to switch between Dense/MoE (like HF's first_k_dense_replace)
        # Defaulting to always use MoE based on the class name, but adding placeholder conditional logic
        first_k_dense_replace = getattr(config, 'first_k_dense_replace', 0) # Default to 0 if not in config

        if layer_idx >= first_k_dense_replace:
             # Use Mixture of Experts Layer
             self.mlp = DeepseekV3MoE(config)
        else:
             # Use standard Dense MLP layer (make sure intermediate_size is set correctly in config)
             # Ensure config has 'intermediate_size' for dense layers if this branch is used
             dense_intermediate_size = getattr(config, 'intermediate_size', config.n_embd * 4) # Example default
             self.mlp = DeepseekV3MLP(config, intermediate_size=dense_intermediate_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, # hidden_states
               Optional[torch.Tensor], # attention_weights
               Optional[Tuple[torch.Tensor, torch.Tensor]], # present_key_value (cache)
               Optional[torch.Tensor]]: # auxiliary_loss
        """
        Forward pass for the Decoder Layer.

        Args:
            hidden_states: Input tensor [B, S, D]
            position_embeddings: Tuple of (cos, sin) for RoPE [B, MaxSeqLen, Dim]
            attention_mask: Mask for attention [B, 1, S_q, S_kv] or compatible
            past_key_value: Tuple of (cached_key, cached_value) for this layer
            cache_position: Absolute positions of tokens [S]
            use_cache: Whether to return the updated KV cache
            output_attentions: Whether to return attention weights

        Returns:
            Tuple containing:
                - hidden_states: Output tensor [B, S, D]
                - attention_weights: Optional tensor from attention layer
                - present_key_value: Optional tuple for updated KV cache
                - auxiliary_loss: Optional tensor from MoE layer (during training)
        """
        # --- Self Attention Block ---
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=normalized_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attn_output = attn_outputs[0]         # Main output [B, S, D]
        attn_weights = attn_outputs[1]      # Optional attention weights
        present_key_value = attn_outputs[2] # Optional updated cache tuple

        # First residual connection
        hidden_states = residual + attn_output

        # --- MLP / MoE Block ---
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)

        # The MLP/MoE forward method might return aux loss
        if isinstance(self.mlp, DeepseekV3MoE):
             mlp_output, auxiliary_loss = self.mlp(normalized_hidden_states)
        else:
             # Dense MLP doesn't return aux loss
             mlp_output = self.mlp(normalized_hidden_states)
             auxiliary_loss = None # Ensure aux_loss is defined

        # Second residual connection
        hidden_states = residual + mlp_output

        return hidden_states, attn_weights, present_key_value, auxiliary_loss

# ==============================================================================
# Final Step: Implement the Main DeepSeekMoE Model Class
# ==============================================================================

class DeepSeekMoE(nn.Module):
    """
    DeepSeek V3 MoE Language Model implementation from scratch.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Input Embeddings ---
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # --- Rotary Position Embeddings ---
        # Instantiated once and shared across layers
        # Requires device during init if inv_freq calculation needs it
        # We'll determine device in forward pass, so pass None for now
        self.rope = DeepseekV3RotaryEmbedding(config, device=None)

        # --- Transformer Blocks ---
        self.h = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)]
        )

        # --- Final Layer Normalization ---
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)

        # --- Language Model Head ---
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # --- Weight Tying ---
        if config.tie_word_embeddings:
            self.wte.weight = self.lm_head.weight

        # --- Initialize Weights ---
        self.apply(self._init_weights)

        # --- Report Parameter Count (optional) ---
        # Call self._report_params() after initialization if needed

    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # Special handling for MoE gate projection? HF initializes router weights differently sometimes.
            # Check DeepseekV3 specific initialization if available. Defaulting to normal.
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # Handle padding_idx? Check if config specifies one
            # if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
            #     if module.padding_idx is not None:
            #           with torch.no_grad():
            #               module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
             # Initialize RMSNorm weight to 1
             if hasattr(module, 'weight') and module.weight is not None:
                 nn.init.ones_(module.weight)
        # Add specific initializations for other layers if needed


    def _report_params(self):
        """Calculates and prints total and active parameters (similar to original)."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_moe_params = 0
        n_shared_params = 0

        for block in self.h:
            if isinstance(block.mlp, DeepseekV3MoE):
                 # Access experts within the MoE layer
                 n_moe_params += sum(p.numel() for name, p in block.mlp.experts.named_parameters() if p.requires_grad)
                 if block.mlp.shared_experts is not None:
                     n_shared_params += sum(p.numel() for name, p in block.mlp.shared_experts.named_parameters() if p.requires_grad)

        n_non_moe_params = n_params - n_moe_params - n_shared_params
        n_active_params = n_non_moe_params + n_shared_params
        # Use correct config names for expert counts
        if self.config.n_routed_experts > 0:
             active_ratio = self.config.num_experts_per_tok / self.config.n_routed_experts
             n_active_params += (n_moe_params * active_ratio)

        print(f"Model Parameters: Total={n_params/1e6:.2f}M, Active={n_active_params/1e6:.2f}M")
        print(f" Breakdown: Non-MoE={n_non_moe_params/1e6:.2f}M, Shared={n_shared_params/1e6:.2f}M, MoE Experts={n_moe_params/1e6:.2f}M")


    @staticmethod
    def _prepare_4d_causal_attention_mask(
        attention_mask: Optional[torch.Tensor], # Optional 2D mask [B, S_kv]
        input_shape: Tuple[int, int], # (batch_size, query_seq_len)
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Creates a 4D causal attention mask suitable for SDPA.
        If attention_mask is provided, it combines it with the causal mask.
        Handles prefill and decoding stages.

        Args:
            attention_mask: Optional 2D mask indicating padding tokens.
            input_shape: Shape of the query tensor (batch_size, query_seq_len).
            dtype: Data type for the mask.
            device: Device for the mask.
            past_key_values_length: Length of the KV cache.

        Returns:
            A 4D attention mask [B, 1, S_q, S_kv] or None if not needed.
        """
        batch_size, query_seq_len = input_shape
        key_value_seq_len = past_key_values_length + query_seq_len

        if key_value_seq_len <= 1 and attention_mask is None:
            # No mask needed for single token generation without padding
            return None

        # Start with causal mask
        # Shape: [S_q, S_kv]
        causal_mask = torch.full(
            (query_seq_len, key_value_seq_len),
            fill_value=torch.finfo(dtype).min, # Use float min for mask
            dtype=dtype,
            device=device,
        )

        # Apply triangular mask for causality
        # Only consider positions relative to the start of the combined sequence
        start_index = past_key_values_length
        end_index = past_key_values_length + query_seq_len
        triangle_mask = torch.tril(torch.ones((query_seq_len, key_value_seq_len), device=device, dtype=torch.bool), diagonal=past_key_values_length)
        causal_mask.masked_fill_(triangle_mask, 0.0) # Set allowed positions to 0.0

        # Expand to 4D: [1, 1, S_q, S_kv]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Incorporate padding mask if provided
        if attention_mask is not None:
            # Assumes attention_mask is [B, S_kv] with 1 for non-padded, 0 for padded
            # Expand padding mask: [B, 1, 1, S_kv]
            padding_mask_expanded = attention_mask[:, None, None, :].to(dtype=dtype)
            # Combine: Where padding_mask is 0 (padded), set causal_mask to -inf
            # Need to broadcast causal_mask to batch size
            causal_mask = causal_mask.expand(batch_size, 1, query_seq_len, key_value_seq_len).clone() # Clone to avoid modifying original
            causal_mask = torch.where(padding_mask_expanded == 0, torch.finfo(dtype).min, causal_mask)

        # Final shape: [B, 1, S_q, S_kv] or [1, 1, S_q, S_kv] if no padding mask
        # SDPA generally expects [B, H, S_q, S_kv], but broadcasting handles the H dimension.
        # Check if batch dimension needs explicit expansion if attention_mask was None.
        if attention_mask is None and batch_size > 1:
             causal_mask = causal_mask.expand(batch_size, 1, query_seq_len, key_value_seq_len)

        return causal_mask


    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Optional 2D padding mask [B, S]
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # List of tuples per layer
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None # Not implemented yet, but placeholder
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass of the DeepSeekMoE model.

        Args:
            input_ids: Input token indices. Shape: [batch_size, seq_len]
            targets: Target token indices for loss calculation. Shape: [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens (1 = keep, 0 = mask). Shape: [batch_size, seq_len]
            past_key_values: Optional list of KV cache tuples from previous steps.
            use_cache: Whether to use and return the KV cache.
            output_attentions: Whether attention weights should be returned (not fully supported).

        Returns:
            Tuple containing:
                - logits: Output logits. Shape: [batch_size, seq_len, vocab_size]
                - loss: Combined loss (main + auxiliary) if targets provided and training, else None.
                - present_key_values: Updated list of KV cache tuples if use_cache is True.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False # Default false

        # Determine past sequence length from cache
        past_kv_len = 0
        if past_key_values is not None:
            # Get length from the first layer's key cache shape [B, H_kv, S_kv, D]
             past_kv_len = past_key_values[0][0].shape[2]

        # Calculate absolute positions for RoPE and caching
        # Shape: [seq_len]
        cache_position = torch.arange(past_kv_len, past_kv_len + seq_len, device=device, dtype=torch.long)

        # --- Get Token Embeddings ---
        h = self.wte(input_ids) # [B, S, D]

        # --- Precompute RoPE frequencies for the necessary length ---
        # Need up to max_seq_len = past_kv_len + seq_len
        max_seq_len_needed = past_kv_len + seq_len
        # Ensure rope is on the correct device (needs input tensor `h` for device info)
        if self.rope.inv_freq.device != device:
             self.rope.to(device)
        # position_ids for RoPE calculation needs to cover the full range [0, max_seq_len_needed - 1]
        pos_ids_for_rope = torch.arange(max_seq_len_needed, device=device).unsqueeze(0) # [1, MaxSeqLen]
        position_embeddings = self.rope(h, pos_ids_for_rope) # Tuple (cos, sin), shape [1, MaxSeqLen, rope_dim]

        # --- Prepare Attention Mask ---
        # Creates a 4D mask [B, 1, S_q, S_kv] suitable for SDPA, combining causal and padding masks.
        causal_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, # Optional 2D padding mask [B, S_kv_total]
            (batch_size, seq_len),
            h.dtype,
            device,
            past_kv_len
        )

        # --- Initialize KV Cache storage if using cache ---
        present_key_values = [] if use_cache else None

        # --- Store auxiliary losses ---
        total_aux_loss = torch.tensor(0.0, device=device, dtype=h.dtype)
        aux_loss_count = 0

        # --- Forward through Transformer Blocks ---
        for i, block in enumerate(self.h):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, attn_weights, present_key_value, auxiliary_loss = block(
                h,
                position_embeddings=position_embeddings, # Pass precomputed full RoPE
                attention_mask=causal_mask,
                past_key_value=layer_past_key_value,
                cache_position=cache_position, # Pass absolute positions
                use_cache=use_cache,
                output_attentions=output_attentions
            )
            h = hidden_states # Update hidden states for the next layer

            if present_key_values is not None:
                present_key_values.append(present_key_value)

            if auxiliary_loss is not None and self.training:
                total_aux_loss += auxiliary_loss
                aux_loss_count += 1

        # --- Final Layer Normalization ---
        h = self.ln_f(h)

        # --- Language Model Head ---
        # Only calculate logits for the last token during generation if cache is used and seq_len is 1
        if use_cache and seq_len == 1 and past_kv_len > 0:
             logits = self.lm_head(h[:, -1:, :]) # [B, 1, V]
        else:
             logits = self.lm_head(h)           # [B, S, V]

        # --- Loss Calculation ---
        loss = None
        if targets is not None:
            # Flatten logits and targets for cross_entropy
            # Logits: [B*S, V], Targets: [B*S]
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1 # Common practice to ignore padding tokens if necessary
            )

            # Add auxiliary loss during training
            if self.training and aux_loss_count > 0:
                final_aux_loss = total_aux_loss / aux_loss_count
                loss = main_loss + final_aux_loss
            else:
                loss = main_loss

        return logits, loss, present_key_values # Return cache


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, master_process=True):
        """Configure AdamW optimizer with weight decay separation (same as original)."""
        # Ensure config is available if needed for optimization tweaks
        # config = self.config

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Create optimizer groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"Optimizer: Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
            print(f"Optimizer: Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"Optimizer: Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused) # Removed eps=1e-8 as it's often default
        return optimizer

# Helper function to create model from config (as in original train script)
def create_deepseek_from_config(config):
    """Helper function to create a DeepSeekMoE model from a config object"""
    # Assumes 'config' is your main training Config object which has
    # 'model_specific' attribute holding the DeepSeekMoEConfig instance
    model_config = config.model_specific
    return DeepSeekMoE(model_config)
