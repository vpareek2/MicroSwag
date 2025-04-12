import torch
import torch.nn.functional as F
from .hellaswag import render_example, iterate_examples, get_most_likely_row

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
