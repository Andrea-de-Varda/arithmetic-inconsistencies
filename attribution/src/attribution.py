"""Attribution patching with per-item accumulation, raw-PyTorch implementation.

Uses HuggingFace AutoModelForCausalLM directly with forward-pre-hooks. We
deliberately avoid nnsight here because the cluster's nnsight version mishandles
`hook.requires_grad_(True)` inside a trace, so backward fails with
"element 0 of tensors does not require grad and does not have a grad_fn".

Per batch, three forward passes:
  A. corrupted prompt + model_corrupted_response  (no hooks; gives "incorrect" log probs)
  B. clean prompt + model_clean_response          (hooks capture clean activations)
  C. corrupted prompt + model_clean_response      (hooks capture corrupted activations
                                                    AND grad of normalized metric)

Hook idiom (forward_pre_hook on mlp.down_proj / self_attn.o_proj):
  detach + clone + requires_grad_(True) the input, replace it, capture the
  replacement. After backward, the captured tensor has .grad.

Reduce, per item, at the LAST prompt-token position:
  attr_i = grad_i ⊙ (clean_act_i − corrupted_act_i)

Output shapes per format:
  neuron per-item:  [n_kept, num_layers, intermediate_size]   (fp16 to save disk)
  head per-item:    [n_kept, num_layers, num_heads]
  aggregate (mean): [num_layers, intermediate_size]           (fp32)
"""

import gc
import torch
import numpy as np
from tqdm import tqdm

from .data import pad_sequence_inputs
from .metrics import compute_sequence_log_prob, get_teacher_forcing_diff


def _build_attention_mask(input_ids_padded, original_lens, pad_token_id):
    """1 for real tokens, 0 for right-padded positions."""
    mask = torch.ones_like(input_ids_padded)
    for i, n in enumerate(original_lens):
        if n < input_ids_padded.shape[1]:
            mask[i, n:] = 0
    return mask


def _hook_targets(model, family, want_mlp):
    """Return list of nn.Module to hook (one per layer)."""
    if family == "gpt2":
        layers = model.transformer.h
        return [(l.mlp.c_proj if want_mlp else l.attn.c_proj) for l in layers]
    layers = model.model.layers
    return [(l.mlp.down_proj if want_mlp else l.self_attn.o_proj) for l in layers]


class _Capturer:
    """Registers forward_pre_hooks; on each forward, replaces hook input with
    a detached+clone, optionally enables grad, retains the replacement so we
    can read it (and its .grad) later."""

    def __init__(self, modules, enable_grad: bool):
        self.modules = modules
        self.enable_grad = enable_grad
        self.captures: list = [None] * len(modules)
        self._handles = []

    def __enter__(self):
        for li, mod in enumerate(self.modules):
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(li)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, li):
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.detach().clone()
            if self.enable_grad:
                x_new.requires_grad_(True)
            self.captures[li] = x_new
            return (x_new,) + tuple(inputs[1:])
        return hook


def _per_example_log_prob_gpu(logits, input_ids_cpu, prompt_len):
    """Same as metrics.compute_sequence_log_prob but on GPU and grad-friendly.

    logits: [seq_len, vocab] on GPU.
    input_ids_cpu: [seq_len] on CPU (we move only the answer slice).
    Returns scalar tensor (on logits.device) summed over answer tokens.
    """
    answer_logits = logits[prompt_len - 1:-1, :]
    answer_labels = input_ids_cpu[prompt_len:].to(answer_logits.device)
    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_lps = log_probs.gather(1, answer_labels.unsqueeze(1)).squeeze(1)
    return token_lps.sum()


def _run_sequence_eval(model, input_ids_list, prompt_lens, batch_size, pad_id, desc):
    """Forward-only pass; returns [N] tensor of conditional log probs (CPU)."""
    n = len(input_ids_list)
    nbatch = (n + batch_size - 1) // batch_size
    out = []
    for b in tqdm(range(nbatch), desc=f"  {desc}", leave=False):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        batch_ids = input_ids_list[s:e]
        batch_plens = prompt_lens[s:e]
        seq_lens = [t.shape[0] for t in batch_ids]
        padded = pad_sequence_inputs(batch_ids, pad_token_id=pad_id).to(model.device)
        amask = _build_attention_mask(padded, seq_lens, pad_id).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=padded, attention_mask=amask).logits
        for i in range(len(batch_ids)):
            lp = compute_sequence_log_prob(
                logits[i, :seq_lens[i], :].cpu().float(),
                batch_ids[i],
                batch_plens[i],
            )
            out.append(lp)
        del logits
    return torch.stack(out)


def compute_baselines(model, model_config,
                      cc_ids, cc_plens, ci_ids, ci_plens,
                      rc_ids, rc_plens, ri_ids, ri_plens,
                      batch_size, pad_id):
    """Four forward passes → clean and corrupted teacher-forcing-diff baselines."""
    print("  baselines: clean+correct...")
    cc = _run_sequence_eval(model, cc_ids, cc_plens, batch_size, pad_id, "clean+correct")
    print("  baselines: clean+incorrect...")
    ci = _run_sequence_eval(model, ci_ids, ci_plens, batch_size, pad_id, "clean+incorrect")
    print("  baselines: corrupted+correct...")
    rc = _run_sequence_eval(model, rc_ids, rc_plens, batch_size, pad_id, "corrupt+correct")
    print("  baselines: corrupted+incorrect...")
    ri = _run_sequence_eval(model, ri_ids, ri_plens, batch_size, pad_id, "corrupt+incorrect")

    clean_baseline = get_teacher_forcing_diff(cc, ci).item()
    corrupted_baseline = get_teacher_forcing_diff(rc, ri).item()
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "clean_baseline": clean_baseline,
        "corrupted_baseline": corrupted_baseline,
        "clean_correct_lps": cc.tolist(),
        "clean_incorrect_lps": ci.tolist(),
        "corr_correct_lps": rc.tolist(),
        "corr_incorrect_lps": ri.tolist(),
    }


def _attribution_loop(model, model_config,
                      cc_ids, cc_plens,
                      rc_ids, rc_plens,
                      ri_ids, ri_plens,
                      clean_baseline, corrupted_baseline,
                      batch_size, pad_id,
                      want_mlp, item_reduce_fn, per_item_shape,
                      desc):
    """Returns list (length n_items) of np.ndarrays of shape per_item_shape."""
    n = len(cc_ids)
    nbatch = (n + batch_size - 1) // batch_size
    per_item = []
    family = model_config["family"]
    targets = _hook_targets(model, family, want_mlp=want_mlp)
    n_layers = len(targets)

    for b in tqdm(range(nbatch), desc=desc):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        bc_batch = cc_ids[s:e]; bc_plens = cc_plens[s:e]
        rc_batch = rc_ids[s:e]; rc_pl = rc_plens[s:e]
        ri_batch = ri_ids[s:e]; ri_pl = ri_plens[s:e]
        bc_seq_lens = [t.shape[0] for t in bc_batch]
        rc_seq_lens = [t.shape[0] for t in rc_batch]
        ri_seq_lens = [t.shape[0] for t in ri_batch]

        bc_padded = pad_sequence_inputs(bc_batch, pad_id).to(model.device)
        rc_padded = pad_sequence_inputs(rc_batch, pad_id).to(model.device)
        ri_padded = pad_sequence_inputs(ri_batch, pad_id).to(model.device)

        bc_mask = _build_attention_mask(bc_padded, bc_seq_lens, pad_id).to(model.device)
        rc_mask = _build_attention_mask(rc_padded, rc_seq_lens, pad_id).to(model.device)
        ri_mask = _build_attention_mask(ri_padded, ri_seq_lens, pad_id).to(model.device)

        # Pass A: corrupted+incorrect, log probs only
        with torch.no_grad():
            inc_logits = model(input_ids=ri_padded, attention_mask=ri_mask).logits
        batch_inc_lps = []
        for i in range(len(ri_batch)):
            lp = compute_sequence_log_prob(
                inc_logits[i, :ri_seq_lens[i], :].cpu().float(),
                ri_batch[i], ri_pl[i],
            )
            batch_inc_lps.append(lp.detach())
        del inc_logits

        # Pass B: clean+correct, capture activations (no grad)
        with _Capturer(targets, enable_grad=False) as cap_clean:
            with torch.no_grad():
                _ = model(input_ids=bc_padded, attention_mask=bc_mask)
        clean_caps = cap_clean.captures  # list of CPU/GPU tensors, no grad

        # Pass C: corrupted+correct, capture + backward
        with _Capturer(targets, enable_grad=True) as cap_corr:
            outputs = model(input_ids=rc_padded, attention_mask=rc_mask)
            logits = outputs.logits
            batch_correct_lps = []
            for i in range(len(rc_batch)):
                batch_correct_lps.append(
                    _per_example_log_prob_gpu(
                        logits[i, :rc_seq_lens[i], :], rc_batch[i], rc_pl[i]
                    )
                )
            correct_lps = torch.stack(batch_correct_lps)
            incorrect_lps = torch.stack(batch_inc_lps).to(correct_lps.device)
            tf_diff = (correct_lps - incorrect_lps).mean()
            normalized = (tf_diff - corrupted_baseline) / (
                clean_baseline - corrupted_baseline
            )
            normalized.backward()
        corr_caps = cap_corr.captures  # list of GPU tensors with .grad

        # Reduce per item
        for i in range(len(bc_batch)):
            c_pos = bc_plens[i] - 1
            r_pos = rc_pl[i] - 1
            attr_layers = []
            for li in range(n_layers):
                grad = corr_caps[li].grad
                if grad is None:
                    attr_layers.append(np.zeros(per_item_shape[1:]))
                    continue
                attr = item_reduce_fn(
                    grad[i, r_pos, :],
                    clean_caps[li][i, c_pos, :],
                    corr_caps[li][i, r_pos, :],
                )
                attr_layers.append(attr)
            per_item.append(np.stack(attr_layers, axis=0))

        del clean_caps, corr_caps, logits, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return per_item


def run_neuron_attribution(model, model_config,
                           cc_ids, cc_plens,
                           rc_ids, rc_plens,
                           ri_ids, ri_plens,
                           clean_baseline, corrupted_baseline,
                           batch_size, pad_id):
    intermediate_size = model_config["intermediate_size"]
    num_layers = model_config["num_layers"]

    def reduce_one(grad_pos, cln_pos, corr_pos):
        attr = grad_pos * (cln_pos - corr_pos)
        return attr.float().detach().cpu().numpy()

    return _attribution_loop(
        model, model_config,
        cc_ids, cc_plens, rc_ids, rc_plens, ri_ids, ri_plens,
        clean_baseline, corrupted_baseline, batch_size, pad_id,
        want_mlp=True,
        item_reduce_fn=reduce_one,
        per_item_shape=(num_layers, intermediate_size),
        desc="neuron attribution",
    )


def run_head_attribution(model, model_config,
                         cc_ids, cc_plens,
                         rc_ids, rc_plens,
                         ri_ids, ri_plens,
                         clean_baseline, corrupted_baseline,
                         batch_size, pad_id):
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_layers = model_config["num_layers"]

    def reduce_one(grad_pos, cln_pos, corr_pos):
        attr_per_dim = grad_pos * (cln_pos - corr_pos)
        attr_per_head = attr_per_dim.reshape(num_heads, head_dim).sum(dim=1)
        return attr_per_head.float().detach().cpu().numpy()

    return _attribution_loop(
        model, model_config,
        cc_ids, cc_plens, rc_ids, rc_plens, ri_ids, ri_plens,
        clean_baseline, corrupted_baseline, batch_size, pad_id,
        want_mlp=False,
        item_reduce_fn=reduce_one,
        per_item_shape=(num_layers, num_heads),
        desc="head attribution",
    )
