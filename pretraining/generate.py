"""Generate text from a trained model"""
import torch
import tiktoken
import argparse
import os
from contextlib import nullcontext

# Assuming your model classes are importable like this
import models.gpt2
import models.llama3
import models.gemma3
import models.mistral
import models.deepseekv3
import models.rwkv
# Add other model imports if needed (e.g., models.phi4)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained NanoTitan model")
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture type (e.g., llama, gpt2, gemma3, mistral, deepseek, rwkv)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default="Hello I am a language model,",
                        help='Input prompt to start generation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate per sample')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling (e.g., 0.8, 1.0). Higher is more random.')
    parser.add_argument('--top_k', type=int, default=None, # Default is None (no top-k)
                        help='Only sample from the top k most likely tokens. (e.g., 50)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda", "cpu"). Auto-detects if None.')
    return parser.parse_args()

def load_model(model_type, checkpoint_path, device):
    """Loads the model checkpoint and configuration."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model_config = checkpoint['config']

    print(f"Instantiating model: {model_type}")
    if model_type == "gpt2":
        model = models.gpt2.GPT(checkpoint_model_config)
    elif model_type == "llama":
        model = models.llama3.LLaMA(checkpoint_model_config)
    elif model_type == "gemma3":
        model = models.gemma3.Gemma3(checkpoint_model_config)
    elif model_type == "mistral":
        model = models.mistral.Mistral(checkpoint_model_config)
    elif model_type == "deepseek":
        # Ensure DeepSeekMoE __init__ takes the specific config object
        model = models.deepseekv3.DeepSeekMoE(checkpoint_model_config)
    elif model_type == "rwkv":
        # --- RWKV Specific Check ---
        # Ensure necessary env vars are set for the kernel logic in model.py
        # You might need to set these *before* running the script
        head_size_env = os.environ.get("RWKV_HEAD_SIZE")
        testing_env = os.environ.get("RWKV_MY_TESTING")
        if not head_size_env or testing_env != 'x070':
             raise RuntimeError(f"RWKV generation requires environment variables RWKV_HEAD_SIZE (found: {head_size_env}) and RWKV_MY_TESTING='x070' (found: '{testing_env}') to be set, matching the compiled kernel.")
        print(f"  RWKV Env Vars OK: RWKV_HEAD_SIZE={head_size_env}, RWKV_MY_TESTING={testing_env}")
        # Set grad_cp=0 for inference if it relies on config
        if hasattr(checkpoint_model_config, 'grad_cp'):
             checkpoint_model_config.grad_cp = 0
        model = models.rwkv.RWKV(checkpoint_model_config)
    # Add other models like Phi4 if needed
    # elif model_type == "phi4":
    #     model = models.phi4.Phi4(checkpoint_model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load the state dict
    state_dict = checkpoint['model']
    # Remove 'module.' prefix if saved from DDP
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

    # Handle potential missing keys or unexpected keys
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Warning: Missing keys during state_dict load: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: Unexpected keys during state_dict load: {load_result.unexpected_keys}")

    model.eval() # Set to evaluation mode
    model.to(device)
    print("Model loaded successfully.")
    return model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate sequence autoregressively.
    idx is (B, T) array of indices in the current context.
    """
    # Determine the model's block size from its config
    # Need a consistent way to access block_size/ctx_len
    if hasattr(model.config, 'block_size'):
        block_size = model.config.block_size
    elif hasattr(model.config, 'ctx_len'): # RWKV uses ctx_len
        block_size = model.config.ctx_len
    else:
        # Fallback or raise error if block size cannot be determined
        print("Warning: Could not determine model block size from config. Using default 1024.")
        block_size = 1024

    for _ in range(max_new_tokens):
        # If the sequence context grows too long, crop it to block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward the model to get the logits for the next token
        # Handle different model output formats
        outputs = model(idx_cond)
        if isinstance(outputs, tuple): # Handle models returning (logits, loss, ...)
            logits = outputs[0]
        else: # Assume model just returns logits
             logits = outputs

        # Focus only on the logits for the last time step
        logits = logits[:, -1, :] # (B, vocab_size)

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Optionally apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set logits not in the top k to -inf
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1) # (B, vocab_size)

        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx


def main():
    args = parse_args()

    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             args.device = 'mps'
        else:
             args.device = 'cpu'
    print(f"Using device: {args.device}")
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"

    # Set seed
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model(args.model, args.checkpoint_path, args.device)

    # Setup tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Encode the prompt
    start_ids = enc.encode(args.prompt, allowed_special={'<|endoftext|>'})
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...] # Add batch dimension

    # Setup context manager for autocast
    ctx = nullcontext()
    if device_type == 'cuda':
        # Use bfloat16 for generation if available, float16 otherwise
        pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=pt_dtype)
        print(f"Using autocast with dtype: {pt_dtype}")

    # Generation loop
    print(f"Generating {args.num_samples} samples...")
    for k in range(args.num_samples):
        print(f"\n--- Sample {k+1}/{args.num_samples} ---")
        with ctx:
            y = generate(model, x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

        # Decode and print
        output_tokens = y[0].tolist() # Get tokens for the first (and only) batch item
        try:
            decoded_output = enc.decode(output_tokens)
            print(decoded_output)
        except Exception as e:
            print(f"Error during decoding: {e}")
            print(f"Raw tokens: {output_tokens}")
        print("--------------------")

if __name__ == "__main__":
    main()
