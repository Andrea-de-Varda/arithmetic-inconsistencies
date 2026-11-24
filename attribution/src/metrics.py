"""Full-sequence teacher-forcing metrics for attribution patching.

Adapted from LLM_Modularity_Attribution-main/src/metrics.py:43-117.
"""

import torch


def compute_sequence_log_prob(logits, input_ids, prompt_len):
    """Sum of log P(token_i | tokens_<i) over the answer tokens only.

    Args:
        logits: [seq_len, vocab] (single example).
        input_ids: [seq_len].
        prompt_len: int, count of prompt tokens.

    Returns scalar tensor.
    """
    answer_logits = logits[prompt_len - 1:-1, :]
    answer_labels = input_ids[prompt_len:]

    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_labels.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


def get_teacher_forcing_diff(correct_log_probs, incorrect_log_probs):
    """Mean over batch of (log P(correct) - log P(incorrect))."""
    return (correct_log_probs - incorrect_log_probs).mean()


def get_accuracy(correct_log_probs, incorrect_log_probs):
    """Fraction of items where correct log prob > incorrect log prob."""
    return (correct_log_probs > incorrect_log_probs).float().mean().item()
