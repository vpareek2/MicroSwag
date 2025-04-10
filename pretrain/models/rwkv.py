"""RWKV model implementation - https://arxiv.org/abs/2305.13048"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RWKV_TimeMix(nn.Module):
    """RWKV Time Mixing module (replacement for self-attention)"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        # Initialize time_mix parameters with position-sensitive values
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0

            # Initialize time mixing coefficients
            time_mix_k = torch.ones(1, 1, config.n_embd)
            time_mix_v = torch.ones(1, 1, config.n_embd)
            time_mix_r = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x = i / config.n_embd
                time_mix_k[0, 0, i] = 1.0 - pow(x, ratio_1_to_almost0)
                time_mix_v[0, 0, i] = 1.0 - pow(x, ratio_1_to_almost0 * 0.7 + 0.3 * ratio_0_to_1)
                time_mix_r[0, 0, i] = 1.0 - pow(x, 0.5 * ratio_1_to_almost0)

            # Initialize time decay parameter
            time_decay = torch.ones(config.n_head, self.head_size)
            for h in range(config.n_head):
                for i in range(self.head_size):
                    x = (h * self.head_size + i) / config.n_embd
                    time_decay[h, i] = -5 + 8 * (x ** (0.7 + 1.3 * ratio_0_to_1))

            # Initialize time_first parameter (u in the paper)
            time_first = torch.ones(config.n_head, self.head_size)
            for h in range(config.n_head):
                for i in range(self.head_size):
                    time_first[h, i] = ratio_0_to_1 * (1.0 - (i / self.head_size))

            # Register parameters
            self.time_mix_k = nn.Parameter(time_mix_k)
            self.time_mix_v = nn.Parameter(time_mix_v)
            self.time_mix_r = nn.Parameter(time_mix_r)
            self.time_decay = nn.Parameter(time_decay)
            self.time_first = nn.Parameter(time_first)

        # Projection matrices
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Time shift for rolling tokens
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Layer Norm for each head
        self.ln_x = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        S = self.head_size

        # Time shift operation (shifting to obtain previous token's info)
        xx = self.time_shift(x)

        # Mix current and time-shifted (previous) tokens
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Project to key, value, receptance
        k = self.key(xk).view(B, T, H, S).transpose(1, 2)    # (B, H, T, S)
        v = self.value(xv).view(B, T, H, S).transpose(1, 2)  # (B, H, T, S)
        r = self.receptance(xr).view(B, T, H, S).transpose(1, 2)  # (B, H, T, S)

        # Compute time-weighted attention
        # This is a simplified implementation of the wkv operation
        # In practice, this should be implemented with CUDA kernels for efficiency
        output = torch.zeros_like(r)
        state = torch.zeros(B, H, S, S, device=x.device)  # (B, H, S, S)

        # Process each token sequentially
        for t in range(T):
            # Compute attention for current token
            kt = k[:, :, t, :]  # (B, H, S)
            vt = v[:, :, t, :]  # (B, H, S)

            # Compute outer product of k and v
            at = torch.einsum('bhs,bhs->bhs', kt, vt).unsqueeze(-1)  # (B, H, S, 1)

            # Apply time decay and first token bonus
            # In the real implementation, this would use the WKV CUDA kernel
            state = at + torch.exp(self.time_decay.unsqueeze(-1)) * state

            # Compute output using the receptance gate
            rt = r[:, :, t, :]  # (B, H, S)
            ot = torch.einsum('bhs,bhss->bhs', rt, self.time_first.unsqueeze(-1) * at + state)
            output[:, :, t, :] = ot

        # Reshape and apply layer norm
        output = output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        output = self.ln_x(output)

        # Project to output dimension
        return self.output(output)

class RWKV_ChannelMix(nn.Module):
    """RWKV Channel Mixing module (replacement for feed-forward network)"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # Initialize time_mix parameters with position-sensitive values
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0

            # Initialize time mixing coefficients
            time_mix_k = torch.ones(1, 1, config.n_embd)
            time_mix_r = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x = i / config.n_embd
                time_mix_k[0, 0, i] = pow(x, ratio_1_to_almost0)
                time_mix_r[0, 0, i] = pow(x, ratio_1_to_almost0)

            # Register parameters
            self.time_mix_k = nn.Parameter(time_mix_k)
            self.time_mix_r = nn.Parameter(time_mix_r)

        # Projection matrices
        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)

        # Time shift for rolling tokens
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x):
        # Time shift operation (shifting to obtain previous token's info)
        xx = self.time_shift(x)

        # Mix current and time-shifted (previous) tokens
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Apply channel mixing
        k = self.key(xk)
        k = torch.square(torch.relu(k))  # Square-ReLU non-linearity
        kv = self.value(k)

        # Apply receptance gating
        return torch.sigmoid(self.receptance(xr)) * kv

class Block(nn.Module):
    """RWKV Block combining Time Mixing and Channel Mixing"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # Layer Normalization
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

        # Special handling for first layer
        if self.layer_id == 0:
            self.ln0 = RMSNorm(config.n_embd)

        # Time Mixing and Channel Mixing
        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        # Apply layer norm to first layer input
        if self.layer_id == 0:
            x = self.ln0(x)

        # Time Mixing (attention alternative)
        x = x + self.att(self.ln1(x))

        # Channel Mixing (FFN alternative)
        x = x + self.ffn(self.ln2(x))

        return x

class RWKV(nn.Module):
    """RWKV Language Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        # RWKV Blocks
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        # Final layer norm and output projection
        self.ln_out = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Print model info
        n_params = sum(p.numel() for p in self.parameters())
        print(f"RWKV model with {n_params/1e6:.1f}M parameters")

    def _init_weights(self):
        """Initialize weights with special attention to certain components"""
        # Initialize embedding weights
        nn.init.normal_(self.emb.weight, std=0.02)

        # Initialize output layer with appropriate scaling
        nn.init.normal_(self.head.weight, std=0.02)

        # For linear layers, use special initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                # Key and receptance projections
                if hasattr(m, '_is_key_proj') and m._is_key_proj:
                    nn.init.normal_(m.weight, std=0.02 * 0.1)  # Smaller for keys
                elif hasattr(m, '_is_receptance_proj') and m._is_receptance_proj:
                    nn.init.normal_(m.weight, std=0.02)

                # Value and output projections get zero or small init
                elif hasattr(m, '_is_value_proj') or hasattr(m, '_is_output_proj'):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Input sequence length {T} exceeds model's context length {self.config.block_size}"

        # Get token embeddings
        x = self.emb(idx)

        # Forward through RWKV blocks
        for block in self.blocks:
            x = block(x)

        # Apply final layer norm
        x = self.ln_out(x)

        # Project to vocabulary
        logits = self.head(x)

        return logits

def create_rwkv_from_config(config):
    """Helper function to create an RWKV model from a config object"""
    model_config = config.model_specific
    return RWKV(model_config)
