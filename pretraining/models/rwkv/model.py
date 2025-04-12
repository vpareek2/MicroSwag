"""RWKV as per https://arxiv.org/abs/2305.13048 and https://github.com/BlinkDL/RWKV-LM"""
# Adapted for NanoTitan project from v7 reference implementation, there are differences in training as well which could impact performance

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint as grad_checkpoint # Use PyTorch's checkpoint

# --- CUDA Kernel Loading & Definition ---

RUN_CUDA_RWKV7g = None # Placeholder
_cuda_kernel_loaded = False

try:
    # Check required environment variables for kernel
    _HEAD_SIZE_STR = os.environ.get("RWKV_HEAD_SIZE", "64") # Default to 64 if not set
    HEAD_SIZE = int(_HEAD_SIZE_STR)
    _MY_TESTING = os.environ.get("RWKV_MY_TESTING", "")

    print(f"RWKV Model Init: HEAD_SIZE={HEAD_SIZE}, MY_TESTING='{_MY_TESTING}'")

    if 'x070' in _MY_TESTING:
        print("RWKV Kernel: x070 detected, attempting to load custom CUDA kernel...")
        _KERNEL_DIR = os.path.dirname(__file__) # Directory of this model.py file
        _CUDA_SOURCES = [
            os.path.join(_KERNEL_DIR, 'cuda', 'wkv7_cuda.cu'),
            os.path.join(_KERNEL_DIR, 'cuda', 'wkv7_op.cpp')
        ]
        print(f"RWKV Kernel: Compiling sources: {_CUDA_SOURCES}")

        if all(os.path.exists(f) for f in _CUDA_SOURCES):
            CHUNK_LEN = 16 # Specific to this kernel version
            flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

            try:
                load(
                    name="wind_backstepping",
                    sources=_CUDA_SOURCES,
                    verbose=True,
                    extra_cuda_cflags=flags,
                    is_python_module=False # Crucial: load as PyTorch extension, not Python module
                )
                print("RWKV Kernel: Custom CUDA kernel loaded successfully.")
                _cuda_kernel_loaded = True
            except Exception as e:
                print("\n" + "!"*60)
                print("!!! ERROR compiling RWKV CUDA kernel !!!")
                print("!!! Check your CUDA environment, compiler (nvcc, g++), and paths !!!")
                print(f"!!! Error details: {e}")
                print("!!! Model will likely fail if CUDA kernel is required.")
                print("!"*60 + "\n")
                # Do not raise here, allow model init but RUN_CUDA_RWKV7g will fail later if called
        else:
            print("\n" + "!"*60)
            print("!!! ERROR: RWKV CUDA kernel source files not found at expected paths:")
            for f in _CUDA_SOURCES:
                 print(f"!!!   - {f} {'(found)' if os.path.exists(f) else '(NOT FOUND)'}")
            print("!!! Cannot compile or load custom kernel.")
            print("!"*60 + "\n")

        if _cuda_kernel_loaded:
            class WindBackstepping(torch.autograd.Function):
                @staticmethod
                def forward(ctx, w, q, k, v, z, b):
                    B, T, H, C = w.shape
                    if T % CHUNK_LEN != 0: raise ValueError(f"Input T ({T}) must be divisible by CHUNK_LEN ({CHUNK_LEN})")
                    # Kernel expects bfloat16
                    orig_dtype = w.dtype
                    if orig_dtype != torch.bfloat16:
                        w, q, k, v, z, b = [t.to(torch.bfloat16) for t in [w, q, k, v, z, b]]
                    assert all(i.is_contiguous() for i in [w,q,k,v,z,b])

                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32, device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32, device=w.device)

                    torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
                    ctx.save_for_backward(w, q, k, v, z, b, s, sa)
                    # Return in original dtype if needed, but kernel likely continues in bf16
                    return y.to(orig_dtype)

                @staticmethod
                def backward(ctx, dy):
                    w, q, k, v, z, b, s, sa = ctx.saved_tensors
                    orig_dtype = dy.dtype
                    # Kernel expects bfloat16
                    if orig_dtype != torch.bfloat16:
                        dy = dy.to(torch.bfloat16)
                    assert all(i.is_contiguous() for i in [dy])

                    dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
                    torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
                    # Return gradients matching original input types
                    return dw.to(orig_dtype), dq.to(orig_dtype), dk.to(orig_dtype), dv.to(orig_dtype), dz.to(orig_dtype), db.to(orig_dtype)

            def _run_cuda_rwkv7g_impl(q, w, k, v, a, b):
                B, T, HC = q.shape
                H = HC // HEAD_SIZE
                if HC % HEAD_SIZE != 0: raise ValueError(f"Input HC ({HC}) must be divisible by HEAD_SIZE ({HEAD_SIZE})")
                input_dtype = q.dtype
                q, w, k, v, a, b = [t.view(B, T, H, HEAD_SIZE) for t in [q, w, k, v, a, b]]
                output = WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)
                return output.to(input_dtype) # Ensure output matches input dtype

            RUN_CUDA_RWKV7g = _run_cuda_rwkv7g_impl # Assign the actual function
        else:
            # Define a placeholder that errors if called
            def _run_cuda_rwkv7g_stub(*args, **kwargs):
                raise RuntimeError("RWKV CUDA kernel 'x070' was specified but failed to load. Cannot execute.")
            RUN_CUDA_RWKV7g = _run_cuda_rwkv7g_stub

    else:
        print("RWKV Kernel: 'x070' not in RWKV_MY_TESTING. Custom CUDA kernel not loaded.")
        def _run_cuda_rwkv7g_stub(*args, **kwargs):
            raise NotImplementedError("RWKV CUDA kernel 'x070' was not specified in RWKV_MY_TESTING. Cannot execute.")
        RUN_CUDA_RWKV7g = _run_cuda_rwkv7g_stub

except Exception as e:
    print(f"An unexpected error occurred during RWKV kernel setup: {e}")
    # Define a placeholder that errors if called
    def _run_cuda_rwkv7g_stub(*args, **kwargs):
        raise RuntimeError(f"RWKV CUDA kernel setup failed with error: {e}. Cannot execute.")
    RUN_CUDA_RWKV7g = _run_cuda_rwkv7g_stub

# --- Model Components ---

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.head_size = config.head_size
        self.n_head = config.n_embd // self.head_size
        assert config.n_embd % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = config.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1) if config.n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            # Parameters for time-mixing interpolation
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            # Orthogonal initialization helper
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False, "Unsupported shape for ortho_init"
                    return x

            # LoRA-like parameters for decay (w), alpha (a), value mixing (v), gate (g)
            D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5) # Base decay + offset

            D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C)) # Base alpha

            D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0) # Base value mixing offset

            D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            # Key scaling and attention parameters
            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85) # Scale for normalized key
            self.k_a = nn.Parameter(torch.ones(1, 1, C)) # Scale for alpha influence on key
            self.r_k = nn.Parameter(torch.zeros(H, N)) # Receptance-key interaction term

            # Standard layers
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # Group norm for output

            # Specific weight initializations
            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        input_dtype = x.dtype

        xx = self.time_shift(x) - x

        # Interpolate inputs based on time-shift parameters
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Calculate R, W, K, V, A, G components
        r = self.receptance(xr)
        w_lora_term = torch.tanh(xw @ self.w1) @ self.w2
        w = -F.softplus(-(self.w0 + w_lora_term)) - 0.5 # Effective decay, clamped <= -0.5

        k = self.key(xk)
        v = self.value(xv)

        # Value mixing with first layer's value (residual-like)
        if self.layer_id == 0:
            v_first = v
        else:
            v_lora_term = (xv @ self.v1) @ self.v2
            v = v + (v_first - v) * torch.sigmoid(self.v0 + v_lora_term)

        # Alpha (in-context learning rate) and Gate calculations
        a_lora_term = (xa @ self.a1) @ self.a2
        a = torch.sigmoid(self.a0 + a_lora_term)
        g_lora_term = xg @ self.g1
        g = torch.sigmoid(g_lora_term) @ self.g2

        # Prepare keys for CUDA kernel (normalized kk, alpha-influenced k)
        kk = k * self.k_k
        kk_float = kk.view(B, T, H, -1).float()
        kk_norm_float = F.normalize(kk_float, dim=-1, p=2.0)
        kk = kk_norm_float.view(B, T, C).to(input_dtype) # Cast back

        k = k * (1 + (a - 1) * self.k_a)

        # Call the custom CUDA kernel if loaded, otherwise it will raise error
        if RUN_CUDA_RWKV7g is None:
             raise RuntimeError("RUN_CUDA_RWKV7g function is not available. Kernel loading might have failed.")
        x_kernel_out = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)

        # Apply GroupNorm (in float32 for stability)
        ln_x_float = self.ln_x.float()
        x_float_input = x_kernel_out.view(B * T, C).float()
        x_float_output = ln_x_float(x_float_input).view(B, T, C)
        x = x_float_output.to(input_dtype) # Cast back

        # Add receptance-key-value interaction term and apply output projection
        term = (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)
        x = x + term.view(B, T, C)
        x = self.output(x * g)

        return x, v_first

class RWKV_CMix_x070(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
        self.value = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (config.n_embd**0.5), 0.5 / (config.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2 # Squared ReLU activation
        return self.value(k)

class Block(nn.Module):
    """A standard RWKV block mixing time-mixing (attention) and channel-mixing (FFN)."""
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0: # Input layernorm for the first block
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.att = RWKV_Tmix_x070(config, layer_id)
        self.ffn = RWKV_CMix_x070(config, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        # Attention block with residual connection
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        # FFN block with residual connection
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV(nn.Module):
    """The main RWKV language model."""
    def __init__(self, config):
        super().__init__()
        self.config = config # Expects an object with attributes like n_layer, n_embd, etc.

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Optional weight tying
        # if getattr(config, 'tie_weights', False):
        #     self.head.weight = self.emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.ctx_len, f"Input sequence length ({T}) exceeds model context length ({self.config.ctx_len})."

        x = self.emb(idx)

        # Initialize v_first for value mixing across layers
        v_first = torch.zeros_like(x)

        # Use gradient checkpointing if specified in config and in training mode
        use_grad_cp = getattr(self.config, 'grad_cp', 0) == 1

        for block in self.blocks:
            if use_grad_cp and self.training:
                 x, v_first = grad_checkpoint(block, x, v_first, use_reentrant=False)
            else:
                 x, v_first = block(x, v_first)

        x = self.ln_out(x)
        logits = self.head(x)

        # Calculate cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, master_process=True):
        """Set up the AdamW optimizer with parameter-specific weight decay."""
        # Parameters that should not decay: biases, LayerNorm/GroupNorm weights, 1D params
        # Parameters that should decay: Linear weights, Embedding weights (usually), 2D+ params
        no_decay = set()
        decay = set()

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Check for parameters that should generally not decay
            if p.dim() < 2 or "bias" in n or "ln" in n or ".bn" in n or "norm" in n or "time_" in n or "_mask" in n or "pos_emb" in n:
                 no_decay.add(n)
            # Specific RWKV parameters that might warrant different treatment (based on original grouping)
            # elif "att.w0" in n: # Original had 2x LR, treat as no_decay for now
            #     no_decay.add(n)
            else: # Assume remaining 2D+ parameters should decay
                 decay.add(n)

        # Sanity check: ensure all parameters are in one group
        all_params = {n for n, p in self.named_parameters() if p.requires_grad}
        inter = decay & no_decay
        union = decay | no_decay
        assert len(inter) == 0, f"Parameters found in both decay and no_decay sets: {inter}"
        assert len(union) == len(all_params), f"Some parameters were not assigned: {all_params - union}"

        param_dict = {pn: p for pn, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[n] for n in sorted(list(no_decay))], "weight_decay": 0.0},
            {"params": [param_dict[n] for n in sorted(list(decay))], "weight_decay": weight_decay},
        ]

        # Filter out empty groups
        optim_groups = [g for g in optim_groups if g['params']]

        if master_process:
            print(f"RWKV Optimizer: Grouping parameters...")
            num_decay_elems = sum(param_dict[n].numel() for n in decay)
            num_nodecay_elems = sum(param_dict[n].numel() for n in no_decay)
            print(f"  Decay group: {len(decay)} tensors, {num_decay_elems:,} elements")
            print(f"  No Decay group: {len(no_decay)} tensors, {num_nodecay_elems:,} elements")

        # Use AdamW eps from config if available, else default
        adam_eps = getattr(self.config, 'adam_eps', 1e-8) # Check if adam_eps is in RWKVConfig

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"RWKV Optimizer: Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=adam_eps, fused=use_fused)
        return optimizer

# Helper function for instantiation from NanoTitan config
def create_rwkv_from_config(config_main):
    """Helper function to create an RWKV model from the main NanoTitan config object"""
    model_config = config_main.model_specific # This should be an RWKVConfig instance
    # Add grad_cp setting from system config if needed by RWKV __init__ or forward
    if hasattr(config_main, 'system') and hasattr(config_main.system, 'use_compile'):
         # Assuming grad_cp logic aligns with use_compile for simplicity, adjust if needed
         setattr(model_config, 'grad_cp', 1 if config_main.system.use_compile else 0)

    return RWKV(model_config)
