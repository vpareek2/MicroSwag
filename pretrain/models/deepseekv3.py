"""DeepSeek V3 MoE architecture as per https://arxiv.org/pdf/2412.19437"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils import rope

class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Router(nn.Module):
    """Expert routing module for DeepSeekMoE. Routes tokens to the most appropriate experts based on input features."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.n_embd
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts

        # Routing projection
        self.routing_weights = nn.Linear(self.input_dim, self.n_experts, bias=False)

        # Optional bias term for auxiliary-loss-free load balancing
        self.routing_bias = nn.Parameter(torch.zeros(self.n_experts))

        # Scale for routing weights
        self.scale = 1.0 / math.sqrt(self.input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten batch and sequence dimensions for routing
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]

        # Compute routing logits
        routing_logits = self.routing_weights(x_flat) * self.scale

        # Store original logits for computing weights later
        original_logits = routing_logits.clone()

        # Apply optional bias term for load balancing
        routing_logits = routing_logits + self.routing_bias

        # Get indices of top-k experts
        _, indices = torch.topk(routing_logits, self.n_active_experts, dim=-1)

        # Compute routing weights using original logits (without bias)
        # Apply softmax only on selected experts for each token
        mask = torch.zeros_like(original_logits).scatter_(-1, indices, 1.0)
        masked_logits = torch.where(mask > 0, original_logits, torch.tensor(float('-inf'), device=x.device))
        weights = F.softmax(masked_logits, dim=-1)
        weights = torch.gather(weights, -1, indices)

        # Reshape back to match input shape
        weights = weights.reshape(batch_size, seq_len, self.n_active_experts)
        indices = indices.reshape(batch_size, seq_len, self.n_active_experts)

        return weights, indices

    def compute_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Calculate auxiliary load balancing loss to ensure all experts are used equally."""
        # Mean usage of experts across batch dimension
        expert_usage = router_probs.mean(dim=[0, 1])

        # We want usage to be uniform across experts
        target_usage = torch.ones_like(expert_usage) / self.n_experts
        balance_loss = torch.mean((expert_usage - target_usage) ** 2)

        return balance_loss

class Expert(nn.Module):
    """Individual expert module in the Mixture of Experts. Each expert is a feed-forward network with SwiGLU activation"""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.n_embd
        self.hidden_dim = config.expert_ffn_dim

        # Expert feed-forward layers
        self.w1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.w3 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        # Initialize with scaled init for better stability
        self.w2.NANOGPT_SCALE_INIT = 1

        # Expert dropout
        self.dropout = nn.Dropout(getattr(config, 'expert_dropout', 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single expert."""
        # SwiGLU activation
        hidden = self.w1(x) * F.silu(self.w3(x))
        output = self.w2(hidden)
        return self.dropout(output)

class SharedExperts(nn.Module):
    """Shared experts that process all tokens."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.n_embd
        # Scale hidden dimension by number of shared experts
        self.hidden_dim = config.expert_ffn_dim * config.n_shared_experts

        # Shared expert layers
        self.w1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.w3 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        # Initialize with scaled init
        self.w2.NANOGPT_SCALE_INIT = 1

        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'expert_dropout', 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for shared experts."""
        # SwiGLU activation
        hidden = self.w1(x) * F.silu(self.w3(x))
        output = self.w2(hidden)
        return self.dropout(output)

class MoELayer(nn.Module):
    """Mixture of Experts layer for DeepSeekMoE. Combines a router network, multiple expert networks, and shared experts."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.n_embd
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts

        # Router for expert selection
        self.router = Router(config)

        # Create separate experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_experts)])

        # Shared experts that process all tokens
        self.shared_experts = SharedExperts(config) if config.n_shared_experts > 0 else None

        # For loss calculation
        self.z_loss_coef = getattr(config, 'z_loss_coef', 0.001)
        self.balance_coef = getattr(config, 'routing_balance_coef', 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the MoE layer."""
        batch_size, seq_len, hidden_dim = x.shape

        # Get expert weights and indices from router
        weights, indices = self.router(x)  # [batch_size, seq_len, n_active_experts]

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Create a mask to track which tokens go to which experts
        token_expert_mask = torch.zeros(
            batch_size, seq_len, self.n_experts,
            device=x.device, dtype=torch.bool
        )

        # Fill in the mask based on routing decisions
        for b in range(batch_size):
            for s in range(seq_len):
                for i in range(self.n_active_experts):
                    expert_idx = indices[b, s, i].item()
                    token_expert_mask[b, s, expert_idx] = True

        # Process each expert in parallel and aggregate results
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = token_expert_mask[:, :, expert_idx]
            if not mask.any():
                continue

            # Get weights for this expert
            expert_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
            for b in range(batch_size):
                for s in range(seq_len):
                    for i in range(self.n_active_experts):
                        if indices[b, s, i].item() == expert_idx:
                            expert_weights[b, s, 0] = weights[b, s, i]

            # Only process tokens routed to this expert
            # This is a simplification - a more efficient implementation would batch tokens
            expert_output = expert(x)
            output = output + expert_output * expert_weights

        # Add shared expert contribution if present
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)

        # Calculate auxiliary losses if needed
        aux_loss = None
        if self.training and (self.z_loss_coef > 0 or self.balance_coef > 0):
            # Expand weights to full expert selection for balance calculation
            full_routing_prob = torch.zeros(
                batch_size, seq_len, self.n_experts,
                device=x.device, dtype=x.dtype
            )
            for b in range(batch_size):
                for s in range(seq_len):
                    for i in range(self.n_active_experts):
                        expert_idx = indices[b, s, i].item()
                        expert_weight = weights[b, s, i]
                        full_routing_prob[b, s, expert_idx] = expert_weight

            # Balance loss to encourage equal expert utilization
            balance_loss = self.router.compute_balance_loss(full_routing_prob)

            # Z-loss to stabilize router outputs (prevent NaN)
            # This penalizes large logits that can cause numerical instability
            router_logits = self.router.routing_weights(x.reshape(-1, hidden_dim))
            z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))

            # Combine losses
            aux_loss = self.z_loss_coef * z_loss + self.balance_coef * balance_loss

        return output, aux_loss

class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) as used in DeepSeek V3"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Set latent dimension to a reasonable default if not specified
        self.latent_dim = getattr(config, 'latent_dim', 512)

        # Query projection (unchanged)
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)

        # MLA specific: KV compression to latent space
        self.wkv_down = nn.Linear(config.n_embd, self.latent_dim, bias=False)

        # Up-projections from latent space to compute K and V
        self.wk_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.wv_up = nn.Linear(self.latent_dim, self.head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        # For checkpointing compatibility
        self.wo.NANOGPT_SCALE_INIT = 1

        # For caching during inference
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                getattr(config, 'max_batch_size', 32),
                getattr(config, 'block_size', 1024),
                self.latent_dim
            ),
            persistent=False
        )

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # Calculate query in original space
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)

        # MLA: Project to compressed latent space for KV
        kv_latent = self.wkv_down(x)  # [B, T, latent_dim]

        # Handle KV caching during inference
        if start_pos is not None:
            # Store latent KV representation
            self.kv_cache[:B, start_pos:start_pos+T] = kv_latent
            # Get the full latent sequence from cache
            full_kv_latent = self.kv_cache[:B, :start_pos+T]
            cache_T = start_pos + T
        else:
            # During training
            full_kv_latent = kv_latent
            cache_T = T

        # Project latent representation up to get K and V
        # This happens for the whole sequence in the cache during inference
        # Note: K and V are now the same dim as a single head, not per head
        k = self.wk_up(full_kv_latent).unsqueeze(2)  # [B, cache_T, 1, head_dim]
        v = self.wv_up(full_kv_latent).unsqueeze(2)  # [B, cache_T, 1, head_dim]

        # Apply rotary positional embeddings to query and key
        # Apply RoPE to Q and K
        if freqs_cis is not None:
            # During inference, we need the full freqs_cis sequence
            full_freqs_cis = freqs_cis
            if start_pos is not None and cache_T > T:
                # Get the full freqs_cis from the beginning
                full_freqs_cis = freqs_cis[:cache_T]

            # Apply RoPE separately to q and k
            q = rope.apply_rotary_emb(q, None, freqs_cis=freqs_cis)[0]
            k = rope.apply_rotary_emb(k, None, freqs_cis=full_freqs_cis)[0]

        # Prepare for attention calculation
        q = q.transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.expand(B, cache_T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, cache_T, head_dim]
        v = v.expand(B, cache_T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, cache_T, head_dim]

        # Create causal mask if needed
        if mask is None and cache_T > 1:
            mask = torch.triu(torch.ones((cache_T, cache_T), device=x.device, dtype=torch.bool), diagonal=1)

        # Flash Attention for efficiency
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)

        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)

        return y

class Block(nn.Module):
    """DeepSeekMoE transformer block"""
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config)  # Changed from Attention to MLAttention
        self.moe_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        # Attention with residual connection
        h = x + self.attn(self.attn_norm(x), freqs_cis, start_pos, mask)

        # MoE with residual connection
        moe_output, aux_loss = self.moe(self.moe_norm(h))
        h = h + moe_output

        return h, aux_loss

class DeepSeekMoE(nn.Module):
    """DeepSeekMoE language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute rotary positional embeddings
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        use_scaled_rope = getattr(config, 'use_scaled_rope', False)
        self.freqs_cis = rope.precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,  # Extra context for potential future needs
            rope_theta,
            use_scaled_rope,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report model size
        n_params = sum(p.numel() for p in self.parameters())
        # Calculate approximate active parameters
        # (Non-MoE params) + (MoE params × active ratio) + (shared params)
        n_moe_params = sum(p.numel() for name, p in self.named_parameters()
                          if 'experts.' in name and 'shared_experts' not in name)
        n_shared_params = sum(p.numel() for name, p in self.named_parameters()
                              if 'shared_experts' in name)
        n_non_moe_params = n_params - n_moe_params - n_shared_params
        active_ratio = config.n_active_experts / config.n_experts
        n_active_params = n_non_moe_params + (n_moe_params * active_ratio) + n_shared_params

        print(f"DeepSeekMoE model with {n_params/1e6:.1f}M total parameters, {n_active_params/1e6:.1f}M active parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MoE layer using vectorized operations for efficiency.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of:
            - Output tensor of shape [batch_size, seq_len, hidden_dim]
            - Optional auxiliary loss (if training)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Get expert weights and indices from router
        weights, indices = self.router(x)  # [batch_size, seq_len, n_active_experts]

        # Reshape for efficient processing
        tokens_flat = x.reshape(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        flat_weights = weights.reshape(-1, self.n_active_experts)  # [batch_size * seq_len, n_active_experts]
        flat_indices = indices.reshape(-1, self.n_active_experts)  # [batch_size * seq_len, n_active_experts]

        # Initialize output
        output = torch.zeros_like(tokens_flat)  # [batch_size * seq_len, hidden_dim]

        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find all tokens routed to this expert
            # Create a mask where any active expert matches this expert index
            expert_mask = (flat_indices == expert_idx).any(dim=-1)  # [batch_size * seq_len]

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_inputs = tokens_flat[expert_mask]  # [n_selected, hidden_dim]

            # Process tokens with this expert
            expert_outputs = expert(expert_inputs)  # [n_selected, hidden_dim]

            # Get weight for each token with this expert
            expert_weights = torch.zeros(flat_indices.shape[0], device=x.device)
            for i in range(self.n_active_experts):
                # For each position where the i-th expert is the current expert, use the i-th weight
                mask_i = (flat_indices[:, i] == expert_idx)
                expert_weights[mask_i] = flat_weights[mask_i, i]

            # Apply weights only to tokens that use this expert
            output[expert_mask] += expert_outputs * expert_weights[expert_mask].unsqueeze(-1)

        # Reshape output back to input shape
        output = output.reshape(batch_size, seq_len, hidden_dim)

        # Add shared expert contribution if present
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)

        # Calculate auxiliary losses if needed
        aux_loss = None
        if self.training and (self.z_loss_coef > 0 or self.balance_coef > 0):
            # Compute full expert routing probability distribution
            full_routing_prob = torch.zeros(
                batch_size, seq_len, self.n_experts,
                device=x.device, dtype=x.dtype
            )

            # Vectorized approach to fill the distribution
            batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, seq_len, self.n_active_experts)
            seq_indices = torch.arange(seq_len, device=x.device).view(1, -1, 1).expand(batch_size, -1, self.n_active_experts)
            expert_indices = indices

            # Use scatter to fill full_routing_prob
            full_routing_prob.scatter_add_(
                2,
                expert_indices,
                weights
            )

            # Balance loss to encourage equal expert utilization
            balance_loss = self.router.compute_balance_loss(full_routing_prob)

            # Z-loss to stabilize router outputs (prevent NaN)
            # This penalizes large logits that can cause numerical instability
            router_logits = self.router.routing_weights(x.reshape(-1, hidden_dim))
            z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))

            # Combine losses
            aux_loss = self.z_loss_coef * z_loss + self.balance_coef * balance_loss

        return output, aux_loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        """Configure optimizer with weight decay on appropriate parameters"""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Weight decay for 2D parameters, no decay for others
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Print parameter counts
        if master_process:
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer with fused version if available
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def create_deepseek_from_config(config):
    """Helper function to create a DeepSeekMoE model from a config object"""
    model_config = config.model_specific
    return DeepSeekMoE(model_config)
