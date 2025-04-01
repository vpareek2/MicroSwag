import torch
import torch.nn.functional as F
from hellaswag import render_example, iterate_examples

def get_most_likely_row(tokens, mask, logits):
    """Helper function for HellaSwag eval."""
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def evaluate_hellaswag(model, device, device_type, ddp_rank, ddp_world_size, distributed_utils):
    """Evaluate model on HellaSwag validation set."""
    num_correct_norm = 0
    num_total = 0

    # Use unwrapped (uncompiled) model for evaluation
    model_for_eval = distributed_utils.unwrap_model(model)
    model_for_eval.eval()

    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue

        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model_for_eval(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp_world_size > 1:
        num_total = distributed_utils.all_reduce_sum(num_total, True, device)
        num_correct_norm = distributed_utils.all_reduce_sum(num_correct_norm, True, device)

    # calculate accuracy
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0

    return {
        "accuracy": acc_norm,
        "correct": num_correct_norm,
        "total": num_total
    }
