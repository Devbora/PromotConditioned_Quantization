import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
from copy import deepcopy
import random
import math
from collections import Counter
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Generate DPO preference pairs for quantization policy.")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--output", type=str, default="KL_dpo_quantization_dataset_fullpolicy.json")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--num-eval-texts", type=int, default=60)
parser.add_argument("--hardware-practical", action="store_true", default=True, help="Use [16, 8, 4] instead of [16, 8, 6, 4]")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

MODEL_NAME = args.model
OUTPUT_PATH = args.output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 512
BATCH_SIZE = args.batch_size
NUM_EVAL_TEXTS = args.num_eval_texts
MIN_TEXT_LENGTH = 50
HARDWARE_PRACTICAL = args.hardware_practical
QUANT_OPTIONS = [16, 8, 4] if HARDWARE_PRACTICAL else [16, 8, 6, 4]
QUANT_LABELS = [f"INT{b}" for b in QUANT_OPTIONS]
GROUP_SIZE = 128
EMBEDDING_TARGET_DIM = 64
PROJECTION_SEED = 42
NUM_PROTECTED_FIRST = 1
NUM_PROTECTED_LAST = 1
NUM_ALPHAS_PER_PROMPT = 7
ALPHA_RANGE = (0.1, 0.95)
ALPHA_ANCHORS = [0.1, 0.5, 0.9]
MIN_SCORE_MARGIN = 0.02
KL_TOLERANCE = 0.02
PENALTY_ONSET = 0.4
PENALTY_STRENGTH = 2.0
ADAPTIVE_MARGIN_FRAC = 0.03

#DATA
DATASET_SOURCES = [
        {
            "name": "wikipedia",
            "dataset_id": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "validation",
            "text_column": "text",
        },
        {
            "name": "code",
            "dataset_id": "google-research-datasets/mbpp",
            "dataset_config": "full",
            "split": "test",
            "text_column": "code",
            "instruction_column": "text",
        },
        {
            "name": "math",
            "dataset_id": "openai/gsm8k",
            "dataset_config": "main",
            "split": "test",
            "text_column": "question",
            "answer_column": "answer",
        },
        {
            "name": "conversation",
            "dataset_id": "tatsu-lab/alpaca",
            "dataset_config": None,
            "split": "train",
            "text_column": "output",
            "instruction_column": "instruction",
        },
        {
            "name": "creative_writing",
            "dataset_id": "roneneldan/TinyStories",
            "dataset_config": None,
            "split": "validation",
            "text_column": "text",
        },
    ]
#LOAD MODEL
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()

if hasattr(model.model, "layers"):
    layers = model.model.layers
elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layers = model.transformer.h
else:
    raise ValueError("Could not auto-detect layer structure.")
num_layers = len(layers)
embedding_dim = model.config.hidden_size
print(f"Model has {num_layers} layers, embedding_dim={embedding_dim}")
protected_layers = (
    set(range(NUM_PROTECTED_FIRST))
        | set(range(num_layers - NUM_PROTECTED_LAST, num_layers))
    )
print(f"Protected layers (always FP16): {sorted(protected_layers)}")
@torch.inference_mode()
def extract_prompt_features(input_ids):
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        ids_2d = input_ids.unsqueeze(0).to(DEVICE)
    else:
        ids_2d = input_ids.to(DEVICE)
    embeddings = model.model.embed_tokens(ids_2d)  # (1, seq_len, hidden_dim)
    mean_emb = embeddings.mean(dim=1).squeeze(0)  # (hidden_dim,)
    return {
        "embedding_raw": mean_emb.cpu().float().tolist(),
        "num_tokens": int(ids_2d.shape[-1]),
    }
def load_source_texts(src):
    name = src["name"]
    print(f"  Loading: {name}")
    ds_kwargs = {
        "path": src["dataset_id"],
        "split": src["split"],
    }
    if src.get("dataset_config"):
        ds_kwargs["name"] = src["dataset_config"]
    if src.get("streaming"):
        ds_kwargs["streaming"] = True
    try:
        ds = load_dataset(**ds_kwargs)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return []
    texts = []
    text_col = src["text_column"]
    if src.get("streaming"):
        for i, item in enumerate(ds):
            if i >= NUM_EVAL_TEXTS * 2:
                break
            t = item.get(text_col, "")
            if len(t) > MIN_TEXT_LENGTH:
                texts.append(t)
    else:
        for item in ds:
            if "answer_column" in src:
                t = item.get(text_col, "") + "\n\nAnswer: " + item.get(src["answer_column"], "")
            elif "instruction_column" in src:
                t = f"Instruction: {item.get(src['instruction_column'], '')}\n\nResponse: {item.get(text_col, '')}"
            else:
                t = item.get(text_col, "")
            if len(t) > MIN_TEXT_LENGTH:
                texts.append(t)

    texts = texts[:NUM_EVAL_TEXTS]
    print(f"    Got {len(texts)} texts")
    return texts
print("\n" + "=" * 60)
print("LOADING EVALUATION PROMPTS (per-chunk)")
print("=" * 60)

all_prompts = []
for src in DATASET_SOURCES:
    texts = load_source_texts(src)
    if not texts:
        continue
    full_text = "\n\n".join(texts)
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    chunk_idx = 0
    for i in range(0, len(input_ids) - MAX_LENGTH, MAX_LENGTH):
        chunk = input_ids[i : i + MAX_LENGTH]
        features = extract_prompt_features(chunk)
        all_prompts.append({
                "source": src["name"],
                "chunk_idx": chunk_idx,
                "features": features,
                "input_ids": chunk.unsqueeze(0).to(DEVICE),  # [1, seq_len]
            })
        chunk_idx += 1
    print(f"    {chunk_idx} prompt chunks from {src['name']}")
random.seed(args.seed)
random.shuffle(all_prompts)
print(f"\n✓ Loaded {len(all_prompts)} individual prompts (shuffled)")
print(f"\nRandom projection: {embedding_dim}D → {EMBEDDING_TARGET_DIM}D (seed={PROJECTION_SEED})")
rng = np.random.RandomState(PROJECTION_SEED)
projection_matrix = rng.normal(0, 1.0 / np.sqrt(EMBEDDING_TARGET_DIM),
    size=(embedding_dim, EMBEDDING_TARGET_DIM)).astype(np.float32)
raw_embeddings = np.array([p["features"]["embedding_raw"] for p in all_prompts], dtype=np.float32)
reduced_embeddings = raw_embeddings @ projection_matrix  # (N, target_dim)
for i, prompt in enumerate(all_prompts):
    prompt["features"]["embedding"] = reduced_embeddings[i].tolist()
    del prompt["features"]["embedding_raw"]
print(f"  Features per prompt: {EMBEDDING_TARGET_DIM} (projected) + 1 (num_tokens) + 1 (alpha) = {EMBEDDING_TARGET_DIM + 2}D")
#helper functions
@torch.inference_mode()
def calculate_perplexity(input_ids, return_logits=False):
    labels = input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    avg_loss = outputs.loss.item()
    ppl = math.exp(min(avg_loss, 100))
    if return_logits:
        return ppl, outputs.logits  # logits: (1, seq_len, vocab_size)
    return ppl
@torch.inference_mode()
def calculate_kl_divergence(input_ids, reference_logits):
    outputs = model(input_ids=input_ids)
    quant_logits = outputs.logits
    ref_probs = F.softmax(reference_logits, dim=-1)
    quant_log_probs = F.log_softmax(quant_logits, dim=-1)
    kl_per_token = F.kl_div(
        quant_log_probs, ref_probs, reduction='none'
        ).sum(dim=-1)
    return kl_per_token.mean().item()
@torch.inference_mode()
def calculate_batch_perplexity(input_ids_batch):
    labels = input_ids_batch
    outputs = model(input_ids=input_ids_batch, labels=labels)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(input_ids_batch.size(0), -1)
    avg_loss_per_seq = loss.mean(dim=1).tolist()
    return [math.exp(min(l, 100)) for l in avg_loss_per_seq]

def simulate_quantization(tensor, bits, group_size=GROUP_SIZE):
    if bits == 16:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    if tensor.dim() < 2:
        abs_max = tensor.abs().max()
        scale = torch.clamp(abs_max / qmax, min=1e-7)
        return torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
    if bits >= 8:
        abs_max = tensor.abs().max(dim=1, keepdim=True).values
        scale = torch.clamp(abs_max / qmax, min=1e-7)
        return torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
    else:
        out_features, in_features = tensor.shape
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            tensor_padded = F.pad(tensor, (0, pad_size), value=0.0)
        else:
            pad_size = 0
            tensor_padded = tensor
            num_groups = tensor_padded.shape[1] // group_size
            t_grouped = tensor_padded.reshape(out_features, num_groups, group_size)
            abs_max = t_grouped.abs().max(dim=2, keepdim=True).values
            scale = torch.clamp(abs_max / qmax, min=1e-7)
            q = torch.clamp(torch.round(t_grouped / scale), qmin, qmax)
            dequant = (q * scale).reshape(out_features, -1)
            if pad_size > 0:
                dequant = dequant[:, :in_features]
            return dequant

def apply_fake_quant_to_layer(layer, bits):
        if bits == 16:
            return
        with torch.no_grad():
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.copy_(simulate_quantization(module.weight, bits))

def calculate_total_cost_mb(policy):
    total = 0.0
    for layer_idx, bits in enumerate(policy):
        for module in layers[layer_idx].modules():
            if not isinstance(module, nn.Linear):
                continue
            numel = module.weight.numel()
            out_features, in_features = module.weight.shape
            weight_mb = (numel * bits / 8.0) / (1024 ** 2)
            if bits < 16 and bits >= 8:
                scale_mb = (out_features * 2) / (1024 ** 2)
            elif bits < 8:
                num_groups = math.ceil(in_features / GROUP_SIZE) * out_features
                scale_mb = (num_groups * 2) / (1024 ** 2)
            else:
                scale_mb = 0.0
            total += weight_mb + scale_mb
        for name, p in layers[layer_idx].named_parameters():
            if 'weight' not in name:
                total += (p.numel() * 2) / (1024 ** 2)
    return total

def apply_full_policy(policy):
    for layer_idx, bits in enumerate(policy):
        apply_fake_quant_to_layer(layers[layer_idx], bits)

@torch.inference_mode()
def measure_layer_sensitivity(input_ids, baseline_ppl, reference_logits):
        sensitivity = {}
        for layer_idx in range(num_layers):
            if layer_idx in protected_layers:
                sensitivity[layer_idx] = {
                    b: {"ppl_delta": 0.0, "kl_div": 0.0}
                    for b in QUANT_OPTIONS if b < 16
                }
                continue
            sensitivity[layer_idx] = {}
            layer = layers[layer_idx]
            layer_backup = {k: v.clone() for k, v in layer.state_dict().items()}
            sensitivity_bits = [b for b in QUANT_OPTIONS if b < 16]
            for bits in sensitivity_bits:
                apply_fake_quant_to_layer(layer, bits)
                ppl = calculate_perplexity(input_ids)
                kl = calculate_kl_divergence(input_ids, reference_logits)
                sensitivity[layer_idx][bits] = {
                    "ppl_delta": ppl - baseline_ppl,
                    "kl_div": kl,
                }
                layer.load_state_dict(layer_backup)

            del layer_backup
        return sensitivity

def construct_policies_from_sensitivity(sensitivity):
        policies = []
        names = []
        unprotected = [i for i in range(num_layers) if i not in protected_layers]
        has_int6 = 6 in QUANT_OPTIONS
        lowest_bits = min(b for b in QUANT_OPTIONS if b < 16)
        sorted_by_low = sorted(unprotected, key=lambda i: sensitivity[i][lowest_bits]["kl_div"])
        sorted_by_int8 = sorted(unprotected, key=lambda i: sensitivity[i][8]["kl_div"])
        if has_int6:
            sorted_by_int6 = sorted(unprotected, key=lambda i: sensitivity[i][6]["kl_div"])
        n_unp = len(unprotected)
        def base_policy(default=16):
            p = [default] * num_layers
            for i in protected_layers:
                p[i] = 16
            return p
        #baseline
        policies.append([16] * num_layers)
        names.append("all_fp16")
        #all int8
        policies.append(base_policy(8))
        names.append("all_int8")
        #all int6
        if has_int6:
            policies.append(base_policy(6))
            names.append("all_int6")
        #all int4
        policies.append(base_policy(lowest_bits))
        names.append(f"all_int{lowest_bits}")
        #sensitivity 50% int4 rest int8
        p = base_policy(8)
        for i in sorted_by_low[: n_unp // 2]:
            p[i] = lowest_bits
        policies.append(p)
        names.append("sens_50pct_low")
        #sensitivity tiered
        if has_int6:
            p = base_policy(16)
            q1 = n_unp // 4
            q2 = n_unp // 2
            q3 = 3 * n_unp // 4
            for i in sorted_by_low[:q1]:
                p[i] = 4
            for i in sorted_by_low[q1:q2]:
                p[i] = 6
            for i in sorted_by_low[q2:q3]:
                p[i] = 8
            policies.append(p)
            names.append("sens_tiered_4tier")
            p = base_policy(8)
            t1 = n_unp // 3
            t2 = 2 * n_unp // 3
            for i in sorted_by_low[:t1]:
                p[i] = 4
            for i in sorted_by_low[t1:t2]:
                p[i] = 6
            policies.append(p)
            names.append("sens_tiered_aggr")
        else:
            p = base_policy(16)
            t1 = n_unp // 3
            t2 = 2 * n_unp // 3
            for i in sorted_by_low[:t1]:
                p[i] = 4
            for i in sorted_by_low[t1:t2]:
                p[i] = 8
            policies.append(p)
            names.append("sens_tiered_3tier")
        #conservative
        median_int8 = float(np.median([sensitivity[i][8]["kl_div"] for i in unprotected]))
        p = base_policy(16)
        for i in unprotected:
            if sensitivity[i][8]["kl_div"] <= median_int8:
                p[i] = 8
        policies.append(p)
        names.append("conservative")
        #moderate
        if has_int6:
            median_int6 = float(np.median([sensitivity[i][6]["kl_div"] for i in unprotected]))
            p = base_policy(16)
            for i in unprotected:
                if sensitivity[i][6]["kl_div"] <= median_int6:
                    p[i] = 6
            policies.append(p)
            names.append("moderate_int6")
        else:
            p = base_policy(16)
            for i in sorted_by_low[: n_unp // 4]:
                p[i] = 4
            policies.append(p)
            names.append("moderate_int4")
        #greedy cost-optimal
        median_kl_int8 = float(np.median([sensitivity[i][8]["kl_div"] for i in unprotected]))
        kl_budget = median_kl_int8 * n_unp * 0.5
        avail_sorted = sorted([b for b in QUANT_OPTIONS if b < 16], reverse=True)
        tier_transitions = list(zip([16] + avail_sorted[:-1], avail_sorted))

        steps = []
        for i in unprotected:
            for from_bits, to_bits in tier_transitions:
                params = sum(pr.numel() for pr in layers[i].parameters())
                mb_saved = (params * (from_bits - to_bits) / 8.0) / (1024 ** 2)
                kl_cost = sensitivity[i][to_bits]["kl_div"]
                if from_bits < 16:
                    kl_cost = max(0.0, sensitivity[i][to_bits]["kl_div"]
                                - sensitivity[i][from_bits]["kl_div"])
                efficiency = mb_saved / (kl_cost + 1e-6)
                steps.append((i, from_bits, to_bits, mb_saved, kl_cost, efficiency))
        steps.sort(key=lambda x: x[5], reverse=True)

        p = base_policy(16)
        cumulative_kl = 0.0
        for layer_i, from_bits, to_bits, _, kl_cost, _ in steps:
            if p[layer_i] == from_bits and (cumulative_kl + kl_cost) <= kl_budget:
                p[layer_i] = to_bits
                cumulative_kl += kl_cost
        policies.append(p)
        names.append("greedy_optimal")
        #Anti-sensitivity
        sorted_by_low_desc = list(reversed(sorted_by_low))
        p = base_policy(8)
        for i in sorted_by_low_desc[: n_unp // 2]:
            p[i] = lowest_bits
        policies.append(p)
        names.append("anti_sensitivity")
        #Aggressive 75%
        p = base_policy(8)
        for i in sorted_by_low[: 3 * n_unp // 4]:
            p[i] = lowest_bits
        policies.append(p)
        names.append("aggressive_75pct")
        #Sensitivity mixed
        if has_int6:
            p = base_policy(8)
            for i in sorted_by_low[: n_unp // 4]:
                p[i] = 4
            for i in sorted_by_low[n_unp // 4 : n_unp // 2]:
                p[i] = 6
            policies.append(p)
            names.append("sens_mixed_int4_int6")
        else:
            p = base_policy(8)
            for i in sorted_by_low[: n_unp // 3]:
                p[i] = 4
            policies.append(p)
            names.append("sens_mixed_int4")

        return policies, names

def snapshot_model_weights():
        return {idx: {k: v.clone() for k, v in layers[idx].state_dict().items()}
                for idx in range(num_layers)}

def restore_model_weights(snapshot):
        for idx in range(num_layers):
            layers[idx].load_state_dict(snapshot[idx])

print("\n" + "=" * 60)
print("SENSITIVITY-DRIVEN POLICY EVALUATION")
print("=" * 60)
print(f"  Total prompts: {len(all_prompts)}")
dataset_out = []

for prompt_idx, prompt in enumerate(all_prompts):
    source_name = prompt["source"]
    chunk_idx = prompt["chunk_idx"]
    prompt_features = prompt["features"]
    input_ids = prompt["input_ids"]

    print(f"\n{'━' * 60}")
    print(f"PROMPT {prompt_idx + 1}/{len(all_prompts)} "
        f"(source={source_name}, chunk={chunk_idx})")
    print(f"  Features: embedding_dim={len(prompt_features.get('embedding', []))}, "
        f"num_tokens={prompt_features['num_tokens']}")
    #baseline_ppl
    baseline_ppl, reference_logits = calculate_perplexity(input_ids, return_logits=True)
    print(f"  Baseline PPL (FP16): {baseline_ppl:.2f}")
    #per_layer
    sensitivity = measure_layer_sensitivity(input_ids, baseline_ppl, reference_logits)
    unprotected = [i for i in range(num_layers) if i not in protected_layers]
    sorted_sens = sorted(unprotected, key=lambda i: sensitivity[i][4]["kl_div"], reverse=True)
    print(f"  Most INT4-sensitive layers (KL): ", end="")
    print(", ".join(f"L{i}(KL={sensitivity[i][4]['kl_div']:.4f})" for i in sorted_sens[:5]))
    #construct_policies
    candidate_policies, policy_names = construct_policies_from_sensitivity(sensitivity)
    num_policies = len(candidate_policies)
    model_snapshot = snapshot_model_weights()
    policy_results = []
    for pol_idx, policy in enumerate(candidate_policies):
        restore_model_weights(model_snapshot)
        apply_full_policy(policy)
        ppl = calculate_perplexity(input_ids)
        kl = calculate_kl_divergence(input_ids, reference_logits)
        cost = calculate_total_cost_mb(policy)
        policy_results.append({
            "policy_idx": pol_idx,
            "policy_name": policy_names[pol_idx],
            "quant_config": policy,
            "ppl": ppl,
            "ppl_delta": ppl - baseline_ppl,
            "kl_div": kl,
            "cost_mb": cost,
        })

    restore_model_weights(model_snapshot)
    del model_snapshot
    del reference_logits
    kl_divs = [r["kl_div"] for r in policy_results]
    costs = [r["cost_mb"] for r in policy_results]
    min_kl, max_kl = min(kl_divs), max(kl_divs)
    min_cost, max_cost = min(costs), max(costs)
    kl_norms = [(kl - min_kl) / max(max_kl - min_kl, 1e-4) for kl in kl_divs]
    cost_norms = [(c - min_cost) / max(max_cost - min_cost, 1e-4) for c in costs]
    for r in policy_results:
        bits_summary = Counter(r["quant_config"])
        desc = " | ".join(f"INT{b}:{c}" for b, c in sorted(bits_summary.items()))
        print(f"  {r['policy_name']:24s} | PPL {r['ppl']:>12.2f} (Δ{r['ppl_delta']:>+11.2f}) | "
            f"KL {r['kl_div']:>8.4f} | Cost {r['cost_mb']:>7.1f} MB | [{desc}]")
    sensitivity_out = {}
    for i in range(num_layers):
        layer_sens = {}
        for b in QUANT_OPTIONS:
            if b < 16:
                layer_sens[f"int{b}_ppl_delta"] = round(sensitivity[i][b]["ppl_delta"], 4)
                layer_sens[f"int{b}_kl_div"] = round(sensitivity[i][b]["kl_div"], 6)
        sensitivity_out[str(i)] = layer_sens
    #score based on alpha
    np.random.seed(42 + prompt_idx)
    sampled_alphas = np.random.uniform(*ALPHA_RANGE, size=NUM_ALPHAS_PER_PROMPT)
    all_alphas = np.unique(np.concatenate([sampled_alphas, ALPHA_ANCHORS]))
    all_alphas = np.clip(all_alphas, *ALPHA_RANGE)
    all_alphas = sorted(all_alphas.tolist())
    for alpha in all_alphas:
        scored_policies = deepcopy(policy_results)
        # Nonlinear scoring with tolerance floor, progressive penalty,
        # and smooth cost exponent (see CONFIGS for hyperparameters)
        for r, kn, cn in zip(scored_policies, kl_norms, cost_norms):
            safe_kn = max(0.0, kn - KL_TOLERANCE)
            if kn > PENALTY_ONSET:
                excess = kn - PENALTY_ONSET
                safe_kn += PENALTY_STRENGTH * excess ** 2
            cost_exponent = 2.0 - alpha
            cost_penalty = cn ** cost_exponent
            r["score"] = (alpha * safe_kn) + ((1 - alpha) * cost_penalty)
        ranked = sorted(scored_policies, key=lambda x: x["score"])
        for rank, r in enumerate(ranked):
            r["rank"] = rank
        print(f"  α={alpha:.1f}: ", end="")
        print(" > ".join(f"{r['policy_name']}" for r in ranked[:5]), end="")
        print(" > ..." if len(ranked) > 5 else "")
        score_spread = ranked[-1]["score"] - ranked[0]["score"]
        effective_margin = max(MIN_SCORE_MARGIN, ADAPTIVE_MARGIN_FRAC * score_spread)
        dpo_pairs = []
        for i, j in combinations(range(len(ranked)), 2):
            score_chosen = ranked[i]["score"]
            score_rejected = ranked[j]["score"]
            margin = score_rejected - score_chosen
            if margin >= effective_margin:
                dpo_pairs.append({
                    "chosen_idx": ranked[i]["policy_idx"],
                    "rejected_idx": ranked[j]["policy_idx"],
                    "margin": round(margin, 4),
                })
        entry = {
            "source": source_name,
            "chunk_idx": chunk_idx,
            "prompt_features": {**prompt_features, "alpha": alpha},
            "baseline_ppl": baseline_ppl,
            "layer_sensitivity": sensitivity_out,
            "policies": scored_policies,
            "ranking": [r["policy_idx"] for r in ranked],
            "dpo_pairs": dpo_pairs,
        }
        dataset_out.append(entry)

    num_alphas_this_prompt = len(all_alphas)
    total_pairs = sum(len(e["dpo_pairs"]) for e in dataset_out[-num_alphas_this_prompt:])
    print(f"  → {num_alphas_this_prompt} alphas (continuous), {total_pairs} total pairs")

with open(OUTPUT_PATH, "w") as f:
    json.dump(dataset_out, f, indent=4)
sensitivity_keys = []
for b in QUANT_OPTIONS:
    if b < 16:
        sensitivity_keys.extend([f"int{b}_ppl_delta", f"int{b}_kl_div"])
meta = {
    "model_name": MODEL_NAME,
    "num_layers": num_layers,
    "quant_options": QUANT_OPTIONS,
    "hardware_practical": HARDWARE_PRACTICAL,
    "protected_layers": sorted(protected_layers),
    "policy_names": policy_names,
    "num_policies_per_prompt": len(policy_names),
    "feature_type": "semantic_embedding",
    "embedding_dim": embedding_dim,
    "embedding_target_dim": EMBEDDING_TARGET_DIM,
    "projection_type": "gaussian_random",
    "projection_seed": PROJECTION_SEED,
    "scalar_features": ["num_tokens"],
    "conditioning_variables": ["alpha"],
    "alpha_sampling": "continuous",
    "alpha_range": list(ALPHA_RANGE),
    "alpha_anchors": ALPHA_ANCHORS,
    "sensitivity_keys": sensitivity_keys,
    "quality_metric": "kl_divergence",
    "score_formula": "alpha * shaped_kl(kn) + (1-alpha) * cn^(2-alpha); tolerance=0.02, penalty_onset=0.4",
    "num_prompts": len(all_prompts),
    "num_entries": len(dataset_out),
    "num_sources": len(DATASET_SOURCES),
    "total_dpo_pairs": sum(len(e["dpo_pairs"]) for e in dataset_out),
}
meta_path = OUTPUT_PATH.replace(".json", "_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)
print(f"\nDataset saved to {OUTPUT_PATH}")
print(f"Metadata saved to {meta_path}")
print(f"  {len(all_prompts)} prompts, {len(dataset_out)} entries (continuous alpha)")
print(f"  {meta['total_dpo_pairs']} total DPO pairs")

