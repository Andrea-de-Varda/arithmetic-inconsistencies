"""Greedy decoding utilities for stage 1.

For each item we decode exactly K_i tokens given a prompt, where
K_i = max(token_len(y_i), token_len(y_prime_i)). Both the clean and the
corrupted decode get the SAME K_i so that the two responses are
comparable as teacher-forcing answers in stage 2.
"""

from collections import defaultdict
import torch
from tqdm import tqdm


def k_for_item(tokenizer, y, y_prime):
    """K_i = max number of tokens in y vs y_prime (both BPE, no special tokens)."""
    k_y = len(tokenizer(str(y), add_special_tokens=False)["input_ids"])
    k_yp = len(tokenizer(str(y_prime), add_special_tokens=False)["input_ids"])
    return max(k_y, k_yp)


def _bucket_by_k(prompts, ks):
    """Group indices by K_i so that each batch has constant K (no min/max-token mix)."""
    buckets = defaultdict(list)
    for i, k in enumerate(ks):
        buckets[k].append(i)
    return buckets


def greedy_decode_fixed_k(model, tokenizer, prompts, ks, batch_size=32, desc="decode"):
    """Greedy-decode `ks[i]` tokens for `prompts[i]`. Returns list of strings.

    Padding is left-side (set on tokenizer), so the new tokens for every
    item in the batch are simply the last ks[i] tokens of the output
    (max(ks_in_batch) tokens emitted, but since we bucket by K it's exact).
    """
    device = next(model.parameters()).device
    out = [None] * len(prompts)
    buckets = _bucket_by_k(prompts, ks)

    for k, indices in tqdm(sorted(buckets.items()), desc=desc):
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_prompts = [prompts[i] for i in batch_idx]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,  # BOS at start of each prompt
            ).to(device)

            with torch.no_grad():
                gen = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=k,
                    min_new_tokens=k,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_tokens = gen[:, enc["input_ids"].shape[1] :]  # [B, k]
            for j, i in enumerate(batch_idx):
                text = tokenizer.decode(new_tokens[j], skip_special_tokens=True)
                # Preserve a leading space if the first generated token had one
                first_id = new_tokens[j, 0].item()
                first_tok_str = tokenizer.convert_ids_to_tokens(first_id)
                if first_tok_str.startswith(("▁", "Ġ")) and not text.startswith(" "):
                    text = " " + text
                out[i] = text
    return out
