import math
import time
import inspect
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from hellaswag import render_example, iterate_examples, get_most_likely_row

# Majority of this code is used from Andrej Karpathy's GPT-2 repro video:
# https://youtu.be/l8pRSuU81PU?si=gs1-6eTnYmWgI6kG

# -------------------------------------------------------------------
# MODEL ARCHITECTURE
# -------------------------------------------------------------------

# GPT-2 uses vanilla multi-head attention
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # 3 for query, key, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash Attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side
        y =  self.c_proj(y) # output projection
        return y
