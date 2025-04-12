import math
from dataclasses import dataclass, field

# --- Fixed Project Constraints ---
TARGET_VOCAB_SIZE = 50304
TARGET_BLOCK_SIZE = 1024 # Not used in RWKV param count
TARGET_PARAMS = 124_500_000
PARAM_TOLERANCE = 4_000_000 # +/- ~3% tolerance

# --- RWKV Architectural Defaults ---
RWKV_HEAD_SIZE = 64 # Fixed by kernel

# --- Configuration Dataclass for Calculation ---
@dataclass
class RWKVConfigCalc:
    n_layer: int
    n_embd: int
    vocab_size: int = TARGET_VOCAB_SIZE
    head_size: int = RWKV_HEAD_SIZE
    ctx_len: int = TARGET_BLOCK_SIZE # Keep for consistency if model uses it

    # Derived
    n_head: int = field(init=False)

    def __post_init__(self):
         if self.n_embd % self.head_size != 0:
             raise ValueError(f"n_embd ({self.n_embd}) must be divisible by head_size ({self.head_size})")
         self.n_head = self.n_embd // self.head_size

# --- Parameter Calculation Function ---
def calculate_rwkv_params(config: RWKVConfigCalc, verbose: bool = False) -> int:
    """Calculates total trainable parameters for the RWKV v7 model."""

    C = config.n_embd
    L = config.n_layer
    V = config.vocab_size
    H = config.n_head
    N = config.head_size # Should be C // H

    total_params = 0

    # --- Shared Components ---
    # Embeddings
    params_emb = V * C
    total_params += params_emb
    # Final LayerNorm (ln_out) - Assuming weight & bias
    params_ln_out = 2 * C
    total_params += params_ln_out
    # LM Head (head) - No bias
    params_head = C * V
    total_params += params_head

    # --- Per-Layer Components ---
    params_per_block_total = 0
    for layer_id in range(L):
        params_block = 0

        # LayerNorms (ln1, ln2 - weight & bias)
        params_block += 2 * (2 * C)
        # Input LayerNorm (ln0 - weight & bias, only layer 0)
        if layer_id == 0:
            params_block += 2 * C

        # Attention Block (RWKV_Tmix_x070)
        params_att = 0
        # Time mixing params (x_r, x_w, x_k, x_v, x_a, x_g)
        params_att += 6 * C
        # Base decay, alpha, value mix params (w0, a0, v0)
        params_att += 3 * C
        # Key scaling params (k_k, k_a)
        params_att += 2 * C
        # Receptance-key interaction (r_k)
        params_att += H * N # == C
        # LoRA dimensions (approximate based on C)
        D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))
        D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))
        D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))
        D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))
        # LoRA projection pairs (e.g., w1, w2)
        params_att += 2 * (C * D_DECAY_LORA) # w1 + w2
        params_att += 2 * (C * D_AAA_LORA)   # a1 + a2
        params_att += 2 * (C * D_MV_LORA)    # v1 + v2
        params_att += 2 * (C * D_GATE_LORA)  # g1 + g2
        # Linear layers (receptance, key, value, output) - no bias
        params_att += 4 * (C * C)
        # GroupNorm (ln_x) - weight & bias
        params_att += 2 * C
        params_block += params_att

        # FFN Block (RWKV_CMix_x070)
        params_ffn = 0
        # Time mixing param (x_k)
        params_ffn += C
        # Linear layers (key, value) - no bias
        # key: C -> 4*C, value: 4*C -> C
        params_ffn += (C * (4 * C)) + ((4 * C) * C)
        params_block += params_ffn

        params_per_block_total += params_block

    total_params += params_per_block_total

    if verbose:
        print(f"  Embeddings: {params_emb:,}")
        print(f"  LM Head: {params_head:,}")
        print(f"  Final LN: {params_ln_out:,}")
        print(f"  Params Per Block (avg): {params_per_block_total // L:,}")
        # Can add more detailed breakdown here if needed

    return total_params

# --- Search Script ---
print(f"Target parameters: {TARGET_PARAMS:,} (+/- {PARAM_TOLERANCE:,})")
print(f"Fixed vocab_size: {TARGET_VOCAB_SIZE}, head_size: {RWKV_HEAD_SIZE}")
print("-" * 40)

results = []

# Define reasonable search ranges
n_layer_range = range(10, 21) # RWKV might need more layers for same params
n_embd_range = range(512, 897, 64) # Must be multiple of head_size=64

print("Starting search...")
count = 0
for n_layer in n_layer_range:
    for n_embd in n_embd_range:
        # Check constraint
        if n_embd % RWKV_HEAD_SIZE != 0: continue

        count += 1
        if count % 10 == 0: print(f"Checked {count} configs...")

        try:
            config = RWKVConfigCalc(
                n_layer=n_layer,
                n_embd=n_embd
            )
            params = calculate_rwkv_params(config)

            if abs(params - TARGET_PARAMS) <= PARAM_TOLERANCE:
                 results.append({
                     "params": params,
                     "config": config,
                     "diff": params - TARGET_PARAMS
                 })
                 print(f"  ---> Found: L={n_layer}, D={n_embd} => Params={params:,} (Diff={params - TARGET_PARAMS:,})")

        except ValueError as e:
             # print(f"Skipping invalid config: {e}")
             pass

# --- Report Results ---
print("\n" + "=" * 40)
print(f"Search Complete. Checked {count} configurations.")
print("RWKV Configurations within tolerance:")
print("=" * 40)
if not results:
    print(f"No RWKV configurations found within {PARAM_TOLERANCE:,} of {TARGET_PARAMS:,}.")
    print("Consider adjusting search ranges (n_layer, n_embd).")
else:
    results.sort(key=lambda x: abs(x['diff']))
    for result in results:
        cfg = result['config']
        print(f"Params: {result['params']:,} (Diff: {result['diff']:,})")
        print(f"  Config: L={cfg.n_layer}, D={cfg.n_embd}")
        print("-" * 20)
    print(f"\nClosest RWKV configuration found:")
    best_result = results[0]
    cfg = best_result['config']
    print(f"Params: {best_result['params']:,} (Diff: {best_result['diff']:,})")
    print(f"  Config: L={cfg.n_layer}, D={cfg.n_embd}")
    print("=" * 40)
