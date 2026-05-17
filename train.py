import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import math
import argparse
import numpy as np
from copy import deepcopy
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Train prompt-conditioned quantization policy via DPO")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--data", type=str, default="KL_dpo_quantization_dataset_fullpolicy.json")
parser.add_argument("--meta", type=str, default="KL_dpo_quantization_dataset_fullpolicy_metadata.json")
parser.add_argument("--policy-path", type=str, default="quant_policy_fullpolicy.pth")
parser.add_argument("--output-dir", type=str, default="quantized_model")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--group-size", type=int, default=128)
parser.add_argument("--warmup-epochs", type=int, default=150)
parser.add_argument("--warmup-lr", type=float, default=1e-3)
parser.add_argument("--dpo-epochs", type=int, default=500)
parser.add_argument("--dpo-lr", type=float, default=5e-5)
parser.add_argument("--dpo-beta", type=float, default=0.1)
parser.add_argument("--margin-weight", type=float, default=0.3)
parser.add_argument("--warmup-temperature", type=float, default=0.1)
parser.add_argument("--confidence-threshold", type=float, default=0.4)
parser.add_argument("--boundary-layer-weight", type=float, default=1.5)
parser.add_argument("--num-boundary-layers", type=int, default=2)
parser.add_argument("--kv-cache-quant", action="store_true", default=False)
parser.add_argument("--kv-cache-options", type=int, nargs="+", default=[16, 8, 4])
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--skip-demo", action="store_true", help="Skip the demo inference section")
args = parser.parse_args()

MODEL_NAME = args.model
DATA_JSON_PATH = args.data
META_JSON_PATH = args.meta
POLICY_PATH = args.policy_path
OUTPUT_DIR = args.output_dir
DEVICE = args.device
GROUP_SIZE = args.group_size
WARMUP_EPOCHS = args.warmup_epochs
WARMUP_LR = args.warmup_lr
DPO_EPOCHS = args.dpo_epochs
DPO_LR = args.dpo_lr
DPO_BETA = args.dpo_beta
MARGIN_WEIGHT = args.margin_weight
WARMUP_TEMPERATURE = args.warmup_temperature
CONFIDENCE_THRESHOLD = args.confidence_threshold
BOUNDARY_LAYER_WEIGHT = args.boundary_layer_weight
NUM_BOUNDARY_LAYERS = args.num_boundary_layers
KV_CACHE_QUANT = args.kv_cache_quant
KV_CACHE_OPTIONS = args.kv_cache_options

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


@torch.inference_mode()
def extract_semantic_features(model, tokenizer, text, projection_matrix=None):
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"].to(DEVICE)
    embeddings = model.model.embed_tokens(input_ids)
    mean_emb = embeddings.mean(dim=1).squeeze(0)
    emb_np = mean_emb.cpu().float().numpy()
    if projection_matrix is not None:
        emb_np = emb_np @ projection_matrix
    num_tokens = input_ids.shape[-1]
    return emb_np.tolist() + [float(num_tokens)]

class PromptConditionedQuantPolicy(nn.Module):
    def __init__(self, num_input_features, num_layers, num_quant_options,
                 hidden_dim, kv_cache_options):
        super().__init__()
        self.num_layers = num_layers
        self.num_quant_options = num_quant_options
        self.kv_cache_options = kv_cache_options

        self.fc1 = nn.Linear(num_input_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + 1, hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.out = nn.Linear(hidden_dim, num_layers * num_quant_options)
        if kv_cache_options > 0:
            self.kv_out = nn.Linear(hidden_dim, num_layers * kv_cache_options)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, input_features):
        alpha = input_features[:, -1:]
        x = self.drop(self.act(self.ln1(self.fc1(input_features))))
        x = torch.cat([x, alpha], dim=-1)
        x = self.drop(self.act(self.ln2(self.fc2(x))))
        x = torch.cat([x, alpha], dim=-1)
        x = self.act(self.ln3(self.fc3(x)))
        logits = self.out(x).view(-1, self.num_layers, self.num_quant_options)
        if self.kv_cache_options > 0:
            kv_logits = self.kv_out(x).view(-1, self.num_layers, self.kv_cache_options)
            return logits, kv_logits
        return logits

def full_policy_dpo_loss(policy_net, ref_net, input_features,
                         chosen_configs, rejected_configs,
                         margins=None, layer_weights=None,
                         beta=DPO_BETA, margin_weight=MARGIN_WEIGHT, return_acc=False):
    # Policy logits: (B, L, Q)
    pi_out = policy_net(input_features)
    pi_logits = pi_out[0] if isinstance(pi_out, tuple) else pi_out
    with torch.no_grad():
        ref_out = ref_net(input_features)
        ref_logits = ref_out[0] if isinstance(ref_out, tuple) else ref_out
    pi_logprobs = F.log_softmax(pi_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)
    # Per-layer log-probs for chosen and rejected configs
    pi_chosen_per_layer = pi_logprobs.gather(2, chosen_configs.unsqueeze(-1)).squeeze(-1)   # (B, L)
    pi_rejected_per_layer = pi_logprobs.gather(2, rejected_configs.unsqueeze(-1)).squeeze(-1)
    ref_chosen_per_layer = ref_logprobs.gather(2, chosen_configs.unsqueeze(-1)).squeeze(-1)
    ref_rejected_per_layer = ref_logprobs.gather(2, rejected_configs.unsqueeze(-1)).squeeze(-1)
    # Apply layer sensitivity priors: weight near-boundary layers higher
    if layer_weights is not None:
        lw = layer_weights.unsqueeze(0)  # (1, L)
        pi_chosen_per_layer = pi_chosen_per_layer * lw
        pi_rejected_per_layer = pi_rejected_per_layer * lw
        ref_chosen_per_layer = ref_chosen_per_layer * lw
        ref_rejected_per_layer = ref_rejected_per_layer * lw
    # Sum over layers for full-config log-prob
    pi_chosen = pi_chosen_per_layer.sum(dim=1)
    pi_rejected = pi_rejected_per_layer.sum(dim=1)
    ref_chosen = ref_chosen_per_layer.sum(dim=1)
    ref_rejected = ref_rejected_per_layer.sum(dim=1)
    advantage = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    if margins is not None:
        loss = -F.logsigmoid(beta * advantage - margin_weight * margins).mean()
    else:
        loss = -F.logsigmoid(beta * advantage).mean()
    # KL regularisation (scaled by num_layers to match advantage scale)
    kl_loss_per_layer = F.kl_div(
        pi_logprobs.reshape(-1, policy_net.num_quant_options),
        F.softmax(ref_logits.reshape(-1, ref_net.num_quant_options), dim=-1),
        reduction='batchmean'
    )
    kl_loss_seq = kl_loss_per_layer * policy_net.num_layers
    loss = loss + 0.01 * kl_loss_seq
    if return_acc:
        acc = (advantage > 0).float().mean().item()
        return loss, acc
    return loss

def simulate_quantization(tensor, bits, group_size=GROUP_SIZE):
    if bits == 16:
        return tensor
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
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

def calculate_cost(layer, bits):
    params = sum(p.numel() for p in layer.parameters())
    return (params * (bits / 8.0)) / (1024 ** 2)

#load
print(f"Loading dataset from {DATA_JSON_PATH}...")
with open(DATA_JSON_PATH) as f:
    data = json.load(f)
with open(META_JSON_PATH) as f:
    metadata = json.load(f)

num_layers = metadata["num_layers"]
policy_names = metadata.get("policy_names", [])
quality_metric = metadata.get("quality_metric", "kl_divergence")
score_formula = metadata.get("score_formula", "N/A")

QUANT_OPTIONS = metadata.get("quant_options", [16, 8, 6, 4])
QUANT_LABELS = [f"INT{b}" for b in QUANT_OPTIONS]
BITS_TO_IDX = {b: i for i, b in enumerate(QUANT_OPTIONS)}
num_quant_options = len(QUANT_OPTIONS)
feature_type = metadata.get("feature_type", "heuristic")
projection_matrix = None
embedding_dim = metadata["embedding_dim"]
target_dim = metadata.get("embedding_target_dim", embedding_dim)
scalar_features = metadata.get("scalar_features", ["num_tokens"])
conditioning_vars = metadata.get("conditioning_variables", ["alpha"])
num_input_features = target_dim + len(scalar_features) + len(conditioning_vars)
print(f"  Feature type: semantic_embedding (raw={embedding_dim}, projected={target_dim})")
proj_seed = metadata.get("projection_seed", 42)
if target_dim < embedding_dim:
    rng = np.random.RandomState(proj_seed)
    projection_matrix = rng.normal(
        0, 1.0 / np.sqrt(target_dim),
        size=(embedding_dim, target_dim)
    ).astype(np.float32)
    print(f"  Projection: gaussian_random (seed={proj_seed})")
print(f"  {len(data)} entries, {metadata.get('num_policies_per_prompt', '?')} policies/prompt, {num_layers} layers")
print(f"  Input dim: {num_input_features}")
print(f"  Quant options: {QUANT_OPTIONS} ({'hardware-practical' if len(QUANT_OPTIONS) == 3 else 'with INT6'})")
print(f"  Alpha: {metadata.get('alpha_sampling', 'discrete')} {metadata.get('alpha_range', metadata.get('alpha_values', 'N/A'))}")
print(f"  Quality metric: {quality_metric}")
print(f"  Score formula: {score_formula}")
print(f"  Total DPO pairs in dataset: {metadata.get('total_dpo_pairs', 'N/A')}")
def flatten_features(prompt_features):
    vec = prompt_features["embedding"][:]
    for k in scalar_features:
        vec.append(float(prompt_features[k]))
    for k in conditioning_vars:
        vec.append(float(prompt_features[k]))
    return vec

all_features = []
all_chosen = []
all_rejected = []
all_margins = []
skipped_pairs = 0
for entry in data:
    feat_vec = flatten_features(entry["prompt_features"])

    policies_by_idx = {p["policy_idx"]: p["quant_config"] for p in entry["policies"]}

    for pair in entry["dpo_pairs"]:
        chosen_idx = pair["chosen_idx"]
        rejected_idx = pair["rejected_idx"]

        if chosen_idx not in policies_by_idx or rejected_idx not in policies_by_idx:
            skipped_pairs += 1
            continue

        chosen_policy = policies_by_idx[chosen_idx]
        rejected_policy = policies_by_idx[rejected_idx]

        chosen_indices = [BITS_TO_IDX[b] for b in chosen_policy]
        rejected_indices = [BITS_TO_IDX[b] for b in rejected_policy]

        all_features.append(feat_vec)
        all_chosen.append(chosen_indices)
        all_rejected.append(rejected_indices)
        all_margins.append(pair["margin"])

print(f"  {len(all_features)} total DPO pairs loaded")
if skipped_pairs > 0:
    print(f"Skipped {skipped_pairs} pairs with missing policy indices")

warmup_features = []
warmup_targets = []

for entry in data:
    feat_vec = flatten_features(entry["prompt_features"])

    policies = entry["policies"]
    scores = torch.tensor([p["score"] for p in policies], dtype=torch.float32)
    weights = F.softmax(-scores / WARMUP_TEMPERATURE, dim=0).numpy()

    soft_target = np.zeros((num_layers, num_quant_options), dtype=np.float32)
    for w, p in zip(weights, policies):
        for layer_idx, bits in enumerate(p["quant_config"]):
            opt_idx = BITS_TO_IDX[bits]
            soft_target[layer_idx, opt_idx] += w

    warmup_features.append(feat_vec)
    warmup_targets.append(soft_target)

print(f"  {len(warmup_features)} soft warm-up targets (one per entry)")
print(f"\n  Best policy per entry (showing first 10):")
for entry in data[:10]:
    best_idx = entry["ranking"][0]
    best_result = next(p for p in entry["policies"] if p["policy_idx"] == best_idx)
    bits_summary = Counter(best_result["quant_config"])
    desc = " | ".join(f"INT{b}:{c}" for b, c in sorted(bits_summary.items()))
    alpha_val = entry["prompt_features"].get("alpha", "?")
    label = f"{entry['source']}[{entry['chunk_idx']}] α={alpha_val}"
    pname = best_result.get("policy_name", f"policy_{best_idx}")
    kl_val = best_result.get("kl_div", 0.0)
    score_val = best_result.get("score", 0.0)
    print(f"    {label:35s} → {pname} "
          f"(KL {kl_val:.4f}, Score {score_val:.4f}, "
          f"{best_result['cost_mb']:.1f} MB) [{desc}]")
if len(data) > 10:
    print(f"    ... and {len(data) - 10} more entries")

# DPO tensors
features_t = torch.tensor(all_features, dtype=torch.float32).to(DEVICE)
chosen_t = torch.tensor(all_chosen, dtype=torch.long).to(DEVICE)
rejected_t = torch.tensor(all_rejected, dtype=torch.long).to(DEVICE)
margins_raw = torch.tensor(all_margins, dtype=torch.float32).to(DEVICE)

margin_min = margins_raw.min()
margin_max = margins_raw.max()
margins_norm = (margins_raw - margin_min) / (margin_max - margin_min + 1e-8)
print(f"  Margin stats: min={margin_min:.4f}, max={margin_max:.4f}, "
      f"mean={margins_raw.mean():.4f}, std={margins_raw.std():.4f}")

# Warm-up tensors
warmup_feat_t = torch.tensor(warmup_features, dtype=torch.float32).to(DEVICE)
warmup_targets_t = torch.tensor(np.array(warmup_targets), dtype=torch.float32).to(DEVICE)
dataset_size = len(features_t)
val_size = int(0.1 * dataset_size)
train_size = dataset_size - val_size
indices = torch.randperm(dataset_size)
train_idx, val_idx = indices[:train_size], indices[train_size:]
feat_mean = features_t[train_idx].mean(dim=0)
feat_std = features_t[train_idx].std(dim=0)
feat_mean[-1] = 0.0
feat_std[-1] = 1.0
features_norm = (features_t - feat_mean) / (feat_std + 1e-8)
features_norm = torch.clamp(features_norm, min=-3.0, max=3.0)
warmup_feat_norm = (warmup_feat_t - feat_mean) / (feat_std + 1e-8)
warmup_feat_norm = torch.clamp(warmup_feat_norm, min=-3.0, max=3.0)

train_features = features_norm[train_idx]
train_chosen = chosen_t[train_idx]
train_rejected = rejected_t[train_idx]
train_margins = margins_norm[train_idx]
val_features = features_norm[val_idx]
val_chosen = chosen_t[val_idx]
val_rejected = rejected_t[val_idx]
val_margins = margins_norm[val_idx]

# Warm-up Train/Validation split (90/10)
warmup_size = len(warmup_feat_norm)
warmup_val_size = int(0.1 * warmup_size)
warmup_train_size = warmup_size - warmup_val_size

warmup_indices = torch.randperm(warmup_size)
warmup_train_idx = warmup_indices[:warmup_train_size]
warmup_val_idx = warmup_indices[warmup_train_size:]

warmup_train_feat = warmup_feat_norm[warmup_train_idx]
warmup_train_tgt = warmup_targets_t[warmup_train_idx]
warmup_val_feat = warmup_feat_norm[warmup_val_idx]
warmup_val_tgt = warmup_targets_t[warmup_val_idx]

layer_weights = torch.ones(num_layers, device=DEVICE)
for i in range(min(NUM_BOUNDARY_LAYERS, num_layers)):
    layer_weights[i] *= BOUNDARY_LAYER_WEIGHT
    layer_weights[num_layers - 1 - i] *= BOUNDARY_LAYER_WEIGHT
print(f"  Layer weights (first/last {NUM_BOUNDARY_LAYERS}): {BOUNDARY_LAYER_WEIGHT}x")

print(f"  DPO Train/Val split:    {train_size} / {val_size}")
print(f"  Warmup Train/Val split: {warmup_train_size} / {warmup_val_size}")

#warm-up
hidden_dim = 256 if num_input_features > 50 else 128

policy_net = PromptConditionedQuantPolicy(
    num_input_features=num_input_features,
    num_layers=num_layers,
    num_quant_options=num_quant_options,
    hidden_dim=hidden_dim,
    kv_cache_options=len(KV_CACHE_OPTIONS) if KV_CACHE_QUANT else 0,
).to(DEVICE)
print(f"  Policy network: {num_input_features}D → {hidden_dim}h → {num_layers}×{num_quant_options} "
      f"({'+ KV cache head' if KV_CACHE_QUANT else 'weights only'})")
print(f"  Parameters: {sum(p.numel() for p in policy_net.parameters()):,}")

warmup_opt = optim.AdamW(policy_net.parameters(), lr=WARMUP_LR, weight_decay=1e-4)
warmup_sched = optim.lr_scheduler.CosineAnnealingLR(warmup_opt, WARMUP_EPOCHS)

print(f"\nSupervised Warm-Up ({WARMUP_EPOCHS} epochs, {warmup_train_size} soft targets)...")
for epoch in range(WARMUP_EPOCHS):
    out = policy_net(warmup_train_feat)
    logits = out[0] if isinstance(out, tuple) else out
    log_probs = F.log_softmax(logits.reshape(-1, num_quant_options), dim=-1)
    loss = F.kl_div(
        log_probs,
        warmup_train_tgt.reshape(-1, num_quant_options),
        reduction='batchmean'
    )
    warmup_opt.zero_grad()
    loss.backward()
    warmup_opt.step()
    warmup_sched.step()
    if (epoch + 1) % 50 == 0:
        preds = logits.argmax(dim=-1)
        tgt_labels = warmup_train_tgt.argmax(dim=-1)
        train_acc = (preds == tgt_labels).float().mean().item()
        with torch.no_grad():
            val_out = policy_net(warmup_val_feat)
            val_logits = val_out[0] if isinstance(val_out, tuple) else val_out
            val_preds = val_logits.argmax(dim=-1)
            val_tgt_labels = warmup_val_tgt.argmax(dim=-1)
            val_acc = (val_preds == val_tgt_labels).float().mean().item()

            val_log_probs = F.log_softmax(val_logits.reshape(-1, num_quant_options), dim=-1)
            val_loss = F.kl_div(
                val_log_probs,
                warmup_val_tgt.reshape(-1, num_quant_options),
                reduction='batchmean'
            ).item()

        print(f"  Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} Top-1 Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Top-1 Acc: {val_acc:.2%}")

ref_net = deepcopy(policy_net).to(DEVICE)
ref_net.eval()
dpo_opt = optim.AdamW(policy_net.parameters(), lr=DPO_LR, weight_decay=1e-4)

print(f"\nDPO Fine-Tuning ({DPO_EPOCHS} epochs, β={DPO_BETA}, "
      f"margin_weight={MARGIN_WEIGHT})...")
for epoch in range(DPO_EPOCHS):
    loss, train_acc = full_policy_dpo_loss(
        policy_net, ref_net,
        train_features, train_chosen, train_rejected,
        margins=train_margins, layer_weights=layer_weights, return_acc=True
    )
    dpo_opt.zero_grad()
    loss.backward()
    dpo_opt.step()
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            val_dpo_loss, val_acc = full_policy_dpo_loss(
                policy_net, ref_net,
                val_features, val_chosen, val_rejected,
                margins=val_margins, layer_weights=layer_weights, return_acc=True
            )
        print(f"  Epoch {epoch+1:3d} | Train cDPO: {loss.item():.4f} Acc: {train_acc:.2%} | "
              f"Val cDPO: {val_dpo_loss:.4f} Acc: {val_acc:.2%}")


save_dict = {
    "model_state_dict": policy_net.state_dict(),
    "feat_mean": feat_mean.cpu(),
    "feat_std": feat_std.cpu(),
    "num_layers": num_layers,
    "num_input_features": num_input_features,
    "feature_type": feature_type,
    "hidden_dim": hidden_dim,
    "quant_options": QUANT_OPTIONS,
    "quant_labels": QUANT_LABELS,
    "quality_metric": quality_metric,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "kv_cache_quant": KV_CACHE_QUANT,
    "layer_weights": layer_weights.cpu(),
}
if feature_type == "semantic_embedding":
    save_dict["embedding_dim"] = embedding_dim
    save_dict["embedding_target_dim"] = target_dim
    save_dict["projection_seed"] = proj_seed
torch.save(save_dict, POLICY_PATH)
print(f"\nPolicy saved to {POLICY_PATH}")
#test
if not args.skip_demo:
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    if hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Could not auto-detect layer structure.")

    policy_net.eval()

    def quantize_for_prompt(prompt_text, alpha=0.7):
        feats = extract_semantic_features(model, tokenizer, prompt_text, projection_matrix)
        feats.append(alpha)
        x = torch.tensor([feats], dtype=torch.float32).to(DEVICE)
        x = (x - feat_mean) / (feat_std + 1e-8)
        x = torch.clamp(x, min=-3.0, max=3.0)
        with torch.no_grad():
            out = policy_net(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1)[0]
        decisions = []
        fallback_count = 0
        for i in range(probs.shape[0]):
            layer_probs = probs[i]
            max_prob, max_idx = layer_probs.max(dim=0)

            if max_prob.item() < CONFIDENCE_THRESHOLD:
                top2 = layer_probs.topk(2).indices
                safer_idx = top2[0] if QUANT_OPTIONS[top2[0]] > QUANT_OPTIONS[top2[1]] else top2[1]
                decisions.append(safer_idx.item())
                fallback_count += 1
            else:
                decisions.append(max_idx.item())
        quant_map = {}
        total_orig, total_quant = 0, 0
        for i, layer in enumerate(layers):
            bits_idx = decisions[i]
            bits = QUANT_OPTIONS[bits_idx]
            label = QUANT_LABELS[bits_idx]
            apply_fake_quant_to_layer(layer, bits)
            params = sum(p.numel() for p in layer.parameters())
            orig_mb = (params * 16 / 8.0) / (1024 ** 2)
            quant_mb = (params * bits / 8.0) / (1024 ** 2)
            total_orig += orig_mb
            total_quant += quant_mb
            quant_map[str(i)] = {
                "decision": label,
                "confidence": {QUANT_LABELS[k]: round(probs[i][k].item(), 4) for k in range(len(QUANT_LABELS))},
                "fallback": max_prob.item() < CONFIDENCE_THRESHOLD,
            }

        if fallback_count > 0:
            print(f"  {fallback_count}/{len(layers)} layers used confidence fallback (threshold={CONFIDENCE_THRESHOLD})")

        return quant_map, total_orig, total_quant

    demo_prompts = {
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))",
        "math": "If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance covered?",
        "creative": "The old lighthouse keeper watched as the storm clouds gathered on the horizon, their dark shapes twisting into forms that reminded him of ancient dragons.",
        "technical": "The transformer architecture uses multi-head self-attention to compute contextual representations of input tokens in parallel.",
    }

    demo_alphas = [0.5, 0.7, 0.9]

    print("\n" + "=" * 70)
    print("DEMO: PROMPT-CONDITIONED QUANTIZATION WITH ALPHA CONTROL")
    print("=" * 70)

    print("\nBacking up clean FP16 weights...")
    clean_state_dicts = {i: {k: v.clone() for k, v in layer.state_dict().items()}
                         for i, layer in enumerate(layers)}

    for prompt_type, prompt_text in demo_prompts.items():
        print(f"\n{'─' * 60}")
        print(f"  Prompt type: {prompt_type}")
        print(f"  Text: \"{prompt_text[:80]}...\"")

        for alpha in demo_alphas:
            for i, layer in enumerate(layers):
                layer.load_state_dict(clean_state_dicts[i])

            qmap, orig_mb, quant_mb = quantize_for_prompt(prompt_text, alpha=alpha)

            decisions_list = [qmap[str(i)]["decision"] for i in range(num_layers)]
            counts = Counter(decisions_list)
            desc = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            savings = (1 - quant_mb / orig_mb) * 100
            print(f"    α={alpha:.1f}: [{desc}] → {orig_mb:.1f} → {quant_mb:.1f} MB ({savings:.1f}% saved)")
            print(f"           Per-layer: [{', '.join(d.replace('INT','') for d in decisions_list)}]")
else:
    print("\nSkipping demo (--skip-demo)")

