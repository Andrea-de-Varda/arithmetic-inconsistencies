"""Feature extraction for the predictor comparison (GPU), one model per call.

For every item we run ONE teacher-forced forward pass over
    make_prompt(x, fmt) + model_clean_response
and read, at the last prompt token (position prompt_len-1, the decision point,
matching src/attribution.py):

  - residual-stream hidden state, all n_layers+1 layers    -> probe features x1a
  - mlp.down_proj input activation, all n_layers layers      -> probe features x1b
  - entropy of softmax(logits[prompt_len-1])                 -> x3 (first-token entropy)
  - mean log P(response token | prefix) over the response    -> x2 (own-response logprob)

Two item groups:
  probe pool : numeric problems from the control dataset
               (control_balanced/data, 20k items; responses from
               ensemble/results/_pool_responses/<model>/numeric.json, written
               by ensemble/scripts/slurm/stage1_pool_responses.sbatch, or for
               Llama-3.1-8B the control pipeline's own 20k run), after removing
               every problem whose prompt string also occurs in the main
               dataset (ids are NOT aligned across the two datasets, so
               exclusion is by content). Then incorrect items up to half of
               n_pool, correct items for the rest. Needs reps + label only.
  test set   : the main verbal kept items (English/Spanish/Italian; the ids
               in the main overlap CSV, the same items that carry the circuit
               loadings). Needs reps + x2 + x3 + label.

Outputs (gitignored; DELETED after probe_and_controls.py runs):
  ensemble/results/<model>/features/probe/numeric_{resid,mlp}.npy + numeric_ids.json
  ensemble/results/<model>/features/test/<fmt>_{resid,mlp}.npy    + <fmt>_ids.json
  ensemble/results/<model>/features/labels_conf.csv   (id,format,split,y,x2,x3)
  ensemble/results/<model>/pool_info.json             (committed: pool composition)

Feature arrays are float16 [N, n_layers(+1), width].

Usage on the cluster (from attribution/):
  python ensemble/scripts/extract_features.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # attribution/ on sys.path -> `import src.*`

import numpy as np
import torch
from tqdm import tqdm

from src.model import load_model_hf
from src.data import (make_prompt, prepare_sequence_inputs, pad_sequence_inputs,
                      load_format, DEFAULT_DATA_DIR)
from src.attribution import _Capturer, _build_attention_mask, _hook_targets


VERBAL_FORMATS = ("english", "spanish", "italian")


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


# ---------------------------------------------------------------------------
# item selection
# ---------------------------------------------------------------------------

def load_pool_responses(attr_root, mid):
    """Numeric responses on the control dataset, giving label +
    model_clean_response for every candidate pool item.

    Preferred source is the dedicated probe-pool stage 1
    (ensemble/scripts/slurm/stage1_pool_responses.sbatch), which decodes only
    as many items as the model's accuracy requires and skips the sign-flipped
    direction. Llama-3.1-8B already has the full 20,000-item numeric run from
    the accuracy-balanced control, so that is used as a fallback rather than
    decoding it a second time. Both files come from the same control dataset
    and the same stage-1 code path, so they are interchangeable here."""
    candidates = [
        os.path.join(attr_root, "ensemble", "results", "_pool_responses",
                     model_short(mid), "numeric.json"),
        os.path.join(attr_root, "control_balanced", "results", "responses",
                     model_short(mid), "numeric.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  pool responses: {os.path.relpath(p, attr_root)}")
            recs = json.load(open(p))["records"]
            return {r["id"]: r for r in recs}
    raise FileNotFoundError(
        f"no pool responses for {mid}; looked in:\n  " +
        "\n  ".join(candidates) +
        "\nRun ensemble/scripts/slurm/stage1_pool_responses.sbatch first.")


def build_probe_pool(attr_root, mid, n_pool, seed, main_data_dir):
    """Probe pool: control-dataset numeric problems whose prompt string does
    not occur anywhere in the main dataset (content-level exclusion), taking
    every incorrect item up to half the pool and filling the rest with correct
    items. Returns (records, info).

    The half-pool cap on the minority class is what keeps the pool usable
    across the whole panel. Incorrect items are the scarce class for the
    accurate models (Llama-3.1-8B has ~400 of them, all of which are taken,
    as in v1), but they are the *majority* for
    the weak ones (Llama-3.2-1B is below 50% on numeric), where taking all of
    them would leave no correct items at all and the probe would have nothing
    to discriminate."""
    responses = load_pool_responses(attr_root, mid)
    main_x = {it["x"] for it in load_format("numeric", main_data_dir)}
    cand = [i for i, r in responses.items() if r["x"] not in main_x]
    n_excluded = len(responses) - len(cand)
    incorrect = sorted(i for i in cand if not responses[i]["clean_match_gold"])
    correct = sorted(i for i in cand if responses[i]["clean_match_gold"])
    n_inc_avail, n_cor_avail = len(incorrect), len(correct)
    if not incorrect or not correct:
        raise ValueError(
            f"{mid}: probe pool has only one class "
            f"({n_inc_avail} incorrect, {n_cor_avail} correct); "
            f"decode more items in stage 1")

    rng = np.random.default_rng(seed)
    n_incorrect = min(n_inc_avail, n_pool // 2)
    if n_incorrect < n_inc_avail:
        incorrect = sorted(rng.choice(incorrect, size=n_incorrect,
                                      replace=False).tolist())
    n_correct = min(n_cor_avail, n_pool - len(incorrect))
    correct_sel = sorted(rng.choice(correct, size=n_correct, replace=False).tolist())
    pool_ids = incorrect + correct_sel
    info = {"model": mid, "n_candidates": len(responses),
            "n_excluded_by_content": n_excluded,
            "n_incorrect_available": n_inc_avail,
            "n_correct_available": n_cor_avail,
            "n_pool": len(pool_ids), "n_pool_incorrect": len(incorrect),
            "n_pool_correct": len(correct_sel), "seed": seed}
    print(f"  probe pool: {len(pool_ids)} numeric items "
          f"({len(incorrect)} incorrect + {len(correct_sel)} correct); "
          f"{n_excluded} candidates excluded because their prompt occurs in "
          f"the main dataset")
    return [responses[i] for i in pool_ids], info


def load_test_items(attr_root, mid):
    """Verbal test items = the ids in the main overlap CSV (the kept items that
    carry circuit loadings). Pull x + model_clean_response + label from the
    main responses JSON."""
    csv_path = os.path.join(attr_root, "results", "overlap", model_short(mid),
                            "item_level_predictions_neuron.csv")
    rows = list(csv.DictReader(open(csv_path)))
    ids_by_fmt = {}
    for r in rows:
        if r.get("component", "neuron") != "neuron":
            continue
        ids_by_fmt.setdefault(r["format"], []).append(r["id"])

    items_by_fmt = {}
    for fmt in VERBAL_FORMATS:
        ids = ids_by_fmt.get(fmt, [])
        if not ids:
            print(f"  [warn] no test ids for {fmt} in {os.path.basename(csv_path)}")
            continue
        rp = os.path.join(attr_root, "results", "responses", model_short(mid),
                          f"{fmt}.json")
        by_id = {r["id"]: r for r in json.load(open(rp))["records"]}
        recs = [by_id[i] for i in ids if i in by_id]
        items_by_fmt[fmt] = recs
        print(f"  test/{fmt}: {len(recs)} items")
    return items_by_fmt


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(model, tokenizer, config, records, fmt, batch_size, want_conf):
    """Return (resid [N,L+1,H] fp16, mlp [N,L,I] fp16, x2 [N], x3 [N]).

    x2 (mean response logprob) and x3 (first-token entropy) are only meaningful
    for the test items; want_conf gates their computation (nan otherwise)."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    mlp_modules = _hook_targets(model, config["family"], want_mlp=True)

    n = len(records)
    resid_out = mlp_out = None
    x2_out, x3_out = [], []
    for s in tqdm(range(0, n, batch_size), desc=f"  extract/{fmt}", leave=False):
        batch = records[s:s + batch_size]
        prompts = [make_prompt(r["x"], fmt) for r in batch]
        answers = [r.get("model_clean_response", "") or "" for r in batch]
        ids_list, plens = prepare_sequence_inputs(prompts, answers, tokenizer)
        seq_lens = [t.shape[0] for t in ids_list]
        padded = pad_sequence_inputs(ids_list, pad_token_id=pad_id).to(device)
        amask = _build_attention_mask(padded, seq_lens, pad_id).to(device)

        with _Capturer(mlp_modules, enable_grad=False) as cap:
            out = model(input_ids=padded, attention_mask=amask,
                        output_hidden_states=True)
        hidden = out.hidden_states           # tuple len L+1, each [B,T,H]
        logits = out.logits                  # [B,T,V]

        for i in range(len(batch)):
            dp = plens[i] - 1                 # decision-point position
            r_vec = torch.stack([h[i, dp, :] for h in hidden], dim=0)   # [L+1, H]
            m_vec = torch.stack([cap.captures[li][i, dp, :]
                                 for li in range(len(mlp_modules))], dim=0)  # [L, I]
            if resid_out is None:   # preallocate once the widths are known
                resid_out = np.zeros((n,) + tuple(r_vec.shape), dtype=np.float16)
                mlp_out = np.zeros((n,) + tuple(m_vec.shape), dtype=np.float16)
            resid_out[s + i] = r_vec.float().cpu().numpy().astype(np.float16)
            mlp_out[s + i] = m_vec.float().cpu().numpy().astype(np.float16)

            if want_conf:
                lp_dp = torch.log_softmax(logits[i, dp, :].float(), dim=-1)
                ent = float(-(lp_dp.exp() * lp_dp).sum().item())
                y_start, y_end = plens[i], seq_lens[i]
                if y_end > y_start:
                    ans_logits = logits[i, y_start - 1:y_end - 1, :].float()
                    ans_ids = padded[i, y_start:y_end]
                    ans_lp = torch.log_softmax(ans_logits, dim=-1)
                    tok_lp = ans_lp.gather(1, ans_ids.unsqueeze(1)).squeeze(1)
                    x2 = float(tok_lp.mean().item())
                else:
                    x2 = float("nan")
            else:
                ent, x2 = float("nan"), float("nan")
            x3_out.append(ent)
            x2_out.append(x2)
        del out, hidden, logits

    return resid_out, mlp_out, np.array(x2_out), np.array(x3_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--main-data-dir", default=DEFAULT_DATA_DIR,
                    help="main dataset dir (for content-level exclusion)")
    args = ap.parse_args()

    ens_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ensemble/
    attr_root = os.path.dirname(ens_root)                                   # attribution/
    model_dir = os.path.join(ens_root, "results", model_short(args.model))
    out_dir = os.path.join(model_dir, "features")
    os.makedirs(os.path.join(out_dir, "probe"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

    print(f"extract_features  model={args.model}")
    probe_records, pool_info = build_probe_pool(attr_root, args.model, args.n_pool,
                                                args.seed, args.main_data_dir)
    json.dump(pool_info, open(os.path.join(model_dir, "pool_info.json"), "w"),
              indent=2)
    test_by_fmt = load_test_items(attr_root, args.model)

    model, tokenizer, config = load_model_hf(args.model)

    label_rows = []

    # --- probe pool (numeric): reps + label only ---
    resid, mlp, _, _ = extract(model, tokenizer, config, probe_records,
                               "numeric", args.batch_size, want_conf=False)
    np.save(os.path.join(out_dir, "probe", "numeric_resid.npy"), resid)
    np.save(os.path.join(out_dir, "probe", "numeric_mlp.npy"), mlp)
    json.dump([r["id"] for r in probe_records],
              open(os.path.join(out_dir, "probe", "numeric_ids.json"), "w"))
    for r in probe_records:
        label_rows.append({"id": r["id"], "format": "numeric", "split": "probe",
                           "y": int(bool(r["clean_match_gold"])),
                           "x2": "", "x3": ""})
    print(f"  wrote probe reps: resid{resid.shape} mlp{mlp.shape}")
    del resid, mlp

    # --- verbal test: reps + x2 + x3 + label ---
    for fmt, recs in test_by_fmt.items():
        resid, mlp, x2, x3 = extract(model, tokenizer, config, recs, fmt,
                                     args.batch_size, want_conf=True)
        np.save(os.path.join(out_dir, "test", f"{fmt}_resid.npy"), resid)
        np.save(os.path.join(out_dir, "test", f"{fmt}_mlp.npy"), mlp)
        json.dump([r["id"] for r in recs],
                  open(os.path.join(out_dir, "test", f"{fmt}_ids.json"), "w"))
        for k, r in enumerate(recs):
            label_rows.append({"id": r["id"], "format": fmt, "split": "test",
                               "y": int(bool(r["clean_match_gold"])),
                               "x2": float(x2[k]), "x3": float(x3[k])})
        print(f"  wrote test/{fmt} reps: resid{resid.shape} mlp{mlp.shape}")
        del resid, mlp

    lab_path = os.path.join(out_dir, "labels_conf.csv")
    with open(lab_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "format", "split", "y", "x2", "x3"])
        w.writeheader()
        w.writerows(label_rows)
    print(f"wrote {lab_path}  ({len(label_rows)} rows)")
    print("done.")


if __name__ == "__main__":
    main()
