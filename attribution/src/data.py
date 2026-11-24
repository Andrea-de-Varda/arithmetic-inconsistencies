"""Data loading and fold construction for arithmetic attribution.

Dataset layout (one JSON per format) is at
  arithmetic-dataset/arith_dataset_<format>_counterfactual.json
each item has:
  id, x, x_prime, y, y_prime, ground_truth, ground_truth_prime,
  split, has_carry, n_terms, n_digits

The attribution fold is a deterministic random subset of 2000 ids,
shared across all formats (item ids align by construction).
"""

import json
import os
import torch
import numpy as np


FORMATS = ("numeric", "english", "spanish", "italian")
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "arithmetic-dataset")
)
DEFAULT_FOLD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "attribution_fold_ids.json")
)


# One-shot exemplars, one per format.
# Without these, Llama-3.1-8B base does not know what format to emit
# and accuracy collapses (we observed ~24% on numeric vs ~97% with the exemplar).
ONE_SHOT_EXEMPLARS = {
    "numeric": "7 + 5 = 12\n",
    "english": "seven plus five equals twelve\n",
    "spanish": "siete más cinco es igual a doce\n",
    "italian": "sette più cinque fa dodici\n",
}


def make_prompt(x, format_name):
    """Prepend the per-format one-shot exemplar to the raw problem string."""
    return ONE_SHOT_EXEMPLARS[format_name] + x


def format_path(format_name, data_dir=DEFAULT_DATA_DIR):
    return os.path.join(data_dir, f"arith_dataset_{format_name}_counterfactual.json")


def load_format(format_name, data_dir=DEFAULT_DATA_DIR):
    """Return list of dicts for the given format."""
    with open(format_path(format_name, data_dir), "r") as f:
        return json.load(f)


def build_attribution_fold(
    n_items=2000,
    seed=42,
    fold_path=DEFAULT_FOLD_PATH,
    data_dir=DEFAULT_DATA_DIR,
    overwrite=False,
):
    """Create or read the fold-id list. The same list applies to every format.

    Reads the numeric file (any format would do, ids are aligned), draws
    `n_items` ids without replacement, sorts them, writes to disk.
    """
    if os.path.exists(fold_path) and not overwrite:
        with open(fold_path, "r") as f:
            ids = json.load(f)
        assert len(ids) == n_items, (
            f"Existing fold has {len(ids)} ids but n_items={n_items}; "
            f"pass overwrite=True or delete {fold_path}"
        )
        return ids

    items = load_format("numeric", data_dir=data_dir)
    all_ids = sorted(item["id"] for item in items)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_ids, size=n_items, replace=False)
    chosen_sorted = sorted(chosen.tolist())

    os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    with open(fold_path, "w") as f:
        json.dump(chosen_sorted, f, indent=2)
    print(f"Wrote {len(chosen_sorted)} fold ids to {fold_path}")
    return chosen_sorted


def subset_to_fold(items, fold_ids):
    """Filter `items` to those whose id is in `fold_ids`, preserving fold order."""
    by_id = {item["id"]: item for item in items}
    missing = [i for i in fold_ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} fold ids not present in items (e.g. {missing[:3]})")
    return [by_id[i] for i in fold_ids]


def prepare_sequence_inputs(prompts, answers, tokenizer):
    """Tokenize prompt+answer pairs for full-sequence teacher forcing.

    The prompt is tokenised with add_special_tokens=True so the BOS token is
    prepended (Llama base degrades sharply without it). The answer is then
    tokenised without specials and concatenated, so BOS appears only once at
    the start of the sequence — matching how stage 1 generates.

    Returns:
        all_input_ids: list of 1D long tensors [seq_len_i]
        prompt_lens:   list of int
    """
    all_input_ids = []
    prompt_lens = []
    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids
        all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))
    return all_input_ids, prompt_lens


def pad_sequence_inputs(input_ids_list, pad_token_id=0):
    """Right-pad a list of variable-length input_ids to the same length."""
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded = torch.full(
        (len(input_ids_list), max_len), pad_token_id, dtype=torch.long
    )
    for i, ids in enumerate(input_ids_list):
        padded[i, : ids.shape[0]] = ids
    return padded
