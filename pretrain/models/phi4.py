"""Microsoft Research Phi 4 as per https://arxiv.org/pdf/2412.08905"""
# Probably not going to pre-train with this model, it is too similar to llama3/gemma3. It is halfway in between the two,
# and MSFT Research said most of their success came from training recipe.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils import rope

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Broadcast key and value tensors for Grouped Query Attention (GQA)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

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

class Attention(nn.Module):
    """Grouped-Query Attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head

        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # For checkpointing compatibility
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Add QK-LayerNorm
        self.qk_layernorm = getattr(config, 'qk_layernorm', False)
        if self.qk_layernorm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dim

        # Calculate query, key, values for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH/NKH, HD)

        # Add QK-LayerNorm before applying RoPE
        if self.qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary positional embeddings (RoPE)
        q, k = rope.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # Apply Grouped Query Attention (GQA)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Reshape for attention calculation
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        # Flash Attention for efficiency
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)

        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """MLP with SwiGLU activation"""
    def __init__(self, config):
        super().__init__()
        # Calculate hidden dimension with Phi4's approach
        hidden_dim = 4 * config.n_embd

        # Apply custom dimension multiplier if specified
        ffn_dim_multiplier = getattr(config, 'ffn_dim_multiplier', None)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        # Round to multiple_of if specified
        multiple_of = getattr(config, 'multiple_of', 1)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Define projections
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        # For checkpointing compatibility
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # SwiGLU activation
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Phi4 transformer block with pre-normalization"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, getattr(config, 'norm_eps', 1e-5))
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd, getattr(config, 'norm_eps', 1e-5))
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Phi4(nn.Module):
    """Phi language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, getattr(config, 'norm_eps', 1e-5)),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute rotary positional embeddings
        rope_theta = getattr(config, 'rope_theta', 250000.0)  # as per paper
        use_scaled_rope = getattr(config, 'use_scaled_rope', False)
        self.freqs_cis = rope.precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,  # Extra context for potential future needs
            rope_theta,
            use_scaled_rope,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Print model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Phi4 model with {n_params/1e6:.1f}M parameters")

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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        device = idx.device

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Get token embeddings
        x = self.transformer.wte(idx)

        # Get frequency cis for this sequence length
        freqs_cis = self.freqs_cis[:T].to(device)

        # Create causal mask for attention
        mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, freqs_cis, None, mask)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        # Match your GPT-2 optimizer setup for consistency
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

def create_phi4_from_config(config):
    """Helper function to create a Phi model from a config object"""
    model_config = config.model_specific
    return Phi4(model_config)
