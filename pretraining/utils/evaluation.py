import torch
import torch.nn.functional as F
from .hellaswag import render_example, iterate_examples, get_most_likely_row
# NOTE: No need to import RWKV class if checking by name

# Define CHUNK_LEN for RWKV padding - must match the value used in models/rwkv/model.py
RWKV_CHUNK_LEN = 16
# Define a padding token ID (0 is common, or use enc.eot_token if that's standard for your tokenizer padding)
PAD_TOKEN_ID = 0

def evaluate_hellaswag(model, device, device_type, ddp_rank, ddp_world_size, distributed_utils):
    """Evaluate model on HellaSwag validation set."""
    num_correct_norm = 0
    num_total = 0

    # Use unwrapped (uncompiled) model for evaluation
    model_for_eval = distributed_utils.unwrap_model(model)
    model_for_eval.eval()

    # Check if the model is RWKV by class name string
    is_rwkv = type(model_for_eval).__name__ == 'RWKV'
    if is_rwkv and ddp_rank == 0: # Print only from rank 0 for clarity
        print("Detected RWKV model, will pad HellaSwag inputs to multiple of 16.")

    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue

        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # --- START RWKV PADDING ---
        if is_rwkv:
            B, T = tokens.shape
            # Calculate padding length needed
            pad_len = (RWKV_CHUNK_LEN - (T % RWKV_CHUNK_LEN)) % RWKV_CHUNK_LEN
            if pad_len > 0:
                # Pad tokens tensor
                padding_tokens = torch.full((B, pad_len), PAD_TOKEN_ID, dtype=tokens.dtype, device=device)
                tokens = torch.cat([tokens, padding_tokens], dim=1)

                # Pad mask tensor (mask pads with 0, indicating non-completion tokens)
                padding_mask = torch.zeros((B, pad_len), dtype=mask.dtype, device=device)
                mask = torch.cat([mask, padding_mask], dim=1)
                # Optional: print padding info for debugging
                # if ddp_rank == 0: print(f"HellaSwag rank {ddp_rank}: Padded example {i} from T={T} to T={tokens.shape[1]}")
        # --- END RWKV PADDING ---


        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Pass potentially padded tokens
                logits, loss = model_for_eval(tokens)
            # Pass the potentially padded mask to get_most_likely_row.
            # The loss calculation inside get_most_likely_row uses the mask,
            # so the padded positions (mask=0) will be correctly ignored.
            pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp_world_size > 1:
        # Ensure reduction happens on the correct device
        num_total = distributed_utils.all_reduce_sum(num_total, True, device)
        num_correct_norm = distributed_utils.all_reduce_sum(num_correct_norm, True, device)

    # calculate accuracy
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0

    return {
        "accuracy": acc_norm,
        "correct": num_correct_norm,
        "total": num_total
    }
