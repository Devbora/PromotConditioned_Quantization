
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import time
import re
import numpy as np
from collections import Counter

class PromptConditionedQuantPolicy(nn.Module):
    def __init__(self, num_input_features, num_layers, num_quant_options,
                 hidden_dim, kv_cache_options=0):
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

@torch.inference_mode()
def extract_semantic_features(model, tokenizer, text, projection_matrix=None):
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"].to(model.device)
    embeddings = model.model.embed_tokens(input_ids)
    mean_emb = embeddings.mean(dim=1).squeeze(0)
    emb_np = mean_emb.cpu().float().numpy()

    if projection_matrix is not None:
        emb_np = emb_np @ projection_matrix

    num_tokens = input_ids.shape[-1]
    return emb_np.tolist() + [float(num_tokens)]

QUANT_OPTIONS = [16, 8, 4]
QUANT_LABELS = ["INT16", "INT8", "INT4"]
BITS_TO_IDX = {16: 0, 8: 1, 4: 2}
GPTQ_SUPPORTED_BITS = {4, 8}

def build_dynamic_gptq_config(quant_plan, num_layers):
    layer_bits = {int(k): v["bits"] for k, v in quant_plan.items()}
    gptq_layer_bits = {}
    promoted_layers = []
    for idx, bits in layer_bits.items():
        if bits >= 8:
            gptq_layer_bits[idx] = 8
            if bits == 16:
                promoted_layers.append(idx)
        else:
            gptq_layer_bits[idx] = 4
    gptq_counts = Counter(gptq_layer_bits.values())
    
    if gptq_counts:
        base_bits = max(gptq_counts, key=gptq_counts.get)
    else:
        base_bits = 4
    dynamic = {}
    for layer_idx in range(num_layers):
        bits = gptq_layer_bits.get(layer_idx, base_bits)
        
        if bits != base_bits:
            pattern = rf"+:.*\.{layer_idx}\..*"
            dynamic[pattern] = {"bits": bits}
    
    return base_bits, dynamic, promoted_layers


def prepare_calibration_data(num_samples=256, dataset_name="wikitext"):

    try:
        from datasets import load_dataset
        
        print(f"  Loading calibration data from '{dataset_name}'...")
        
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text_column = "text"
        elif dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            text_column = "text"
        else:
            dataset = load_dataset(dataset_name, split="train")
            text_column = "text"
        calibration_data = []
        for sample in dataset:
            text = sample[text_column].strip()
            if len(text) < 64:
                continue
            
            calibration_data.append(text)
            
            if len(calibration_data) >= num_samples:
                break
        
        print(f" Prepared {len(calibration_data)} calibration text samples")
        return calibration_data    
    except ImportError:
        print("'datasets' package not found. Install with: pip install datasets")
        raise RuntimeError("GPTQ requires calibration data. Install 'datasets': pip install datasets")


def save_as_gptq(model_path, output_dir, quant_plan, num_layers,
                 calibration_samples=256,
                 group_size=128, desc_act=False, device="cuda"):
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError:
        print("'gptqmodel' package required. Install with: pip install gptqmodel")
        return None
    
    from transformers import AutoTokenizer
    print(f"\n{'='*60}")
    print("  Building Dynamic GPTQ Config from Policy Decisions")
    print(f"{'='*60}")
    base_bits, dynamic, promoted_layers = build_dynamic_gptq_config(quant_plan, num_layers)
    # Summary
    layer_bits = {int(k): v["bits"] for k, v in quant_plan.items()}
    bit_counts = Counter(layer_bits.values())
    print(f"\n  Base quantization: {base_bits}-bit GPTQ")
    print(f"  Policy assignments: {dict(sorted(bit_counts.items()))}")
    if promoted_layers:
        print(f"  FP16→8bit promoted (for format consistency): layers {promoted_layers}")
    
    overridden = [i for i in range(num_layers) if dynamic.get(rf"+:.*\.{i}\..*")]
    if overridden:
        print(f"  Layers with dynamic override: {len(overridden)}")
    print(f"  Dynamic config entries: {len(dynamic)}")
    quant_config = QuantizeConfig(
        bits=base_bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,
        dynamic=dynamic if dynamic else None,
    )
    
    print(f"\n  QuantizeConfig: bits={base_bits}, group_size={group_size}, "
          f"desc_act={desc_act}, sym=True")
    print(f"\n  Loading model for GPTQ quantization: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = GPTQModel.load(
        model_path,
        quant_config,
        trust_remote_code=True,
    )
    print(f"\n  Preparing calibration data...")
    calibration_data = prepare_calibration_data(
        num_samples=calibration_samples,
    )
    print(f"\n{'='*60}")
    print("  Running GPTQ Quantization (Hessian-optimized)")
    print(f"{'='*60}")
    t0 = time.time()
    model.quantize(calibration_data, batch_size=1)
    t1 = time.time()
    
    print(f"\nQuantization completed in {t1-t0:.1f}s")
    print(f"\n  Saving quantized model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    total_size = 0
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            total_size += os.path.getsize(fpath)
    
    print(f"    Model saved, Total size: {total_size / (1024**2):.1f} MB")
    return output_dir


def apply_policy(model_path, policy_path, output_path, alpha, prompt, device,
                 calibration_samples=256, group_size=128):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading policy from: {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    num_layers = checkpoint["num_layers"]
    num_input_features = checkpoint.get("num_input_features", 7)
    quant_options = checkpoint.get("quant_options", QUANT_OPTIONS)
    quant_labels = checkpoint.get("quant_labels", QUANT_LABELS)
    policy_net = PromptConditionedQuantPolicy(
        num_input_features, num_layers, len(quant_options), checkpoint.get("hidden_dim", 128)
    ).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()
    feat_mean, feat_std = checkpoint["feat_mean"].to(device), checkpoint["feat_std"].to(device)
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_type = checkpoint.get("feature_type", "heuristic")
    print(f"Loading model embeddings for semantic feature extraction...")
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    temp_model.eval()
    embedding_dim = checkpoint.get("embedding_dim")
    target_dim = checkpoint.get("embedding_target_dim")
    proj = np.random.RandomState(checkpoint.get("projection_seed", 42)).normal(
        0, 1.0/np.sqrt(target_dim), (embedding_dim, target_dim)
    ).astype(np.float32) if target_dim else None
    feats = extract_semantic_features(temp_model, tokenizer, prompt, proj)
    del temp_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    feats.append(alpha)
    x = torch.tensor([feats], dtype=torch.float32).to(device)
    fm, fs = feat_mean.clone(), feat_std.clone()
    fm[-1], fs[-1] = 0.0, 1.0
    x = torch.clamp((x - fm) / (fs + 1e-8), -3.0, 3.0)
    with torch.no_grad():
        logits = policy_net(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        probs = F.softmax(logits, dim=-1)[0]
        decisions = probs.argmax(dim=-1)
    quant_map = {}
    print(f"\n{'='*70}\n  QUANTIZATION PLAN (Alpha={alpha})\n{'='*70}\n")
    for i in range(num_layers):
        b_idx = decisions[i].item()
        bits = quant_options[b_idx]
        label = quant_labels[b_idx]
        gptq_bits = bits
        if bits >= 8:
            gptq_bits = 8
        else:
            gptq_bits = 4
        
        gptq_note = ""
        if bits != gptq_bits:
            gptq_note = f" → GPTQ {gptq_bits}-bit"
        
        quant_map[str(i)] = {
            "layer": i,
            "bits": gptq_bits,
            "original_bits": bits,
            "label": label,
            "confidence": {
                quant_labels[k]: round(probs[i][k].item(), 4)
                for k in range(len(quant_labels))
            }
        }
        conf = probs[i][b_idx].item() * 100
        print(f"  Layer {i:2d} | Policy: {bits}-bit{gptq_note} (Conf: {conf:.1f}%)")
    # Summary
    gptq_bits_list = [v["bits"] for v in quant_map.values()]
    counts = Counter(gptq_bits_list)
    desc = " | ".join(f"GPTQ-{b}bit: {c} layers" for b, c in sorted(counts.items()))
    print(f"\n  GPTQ Plan: {desc}")
    n_promoted = sum(1 for v in quant_map.values() if v["original_bits"] == 16)
    if n_promoted > 0:
        print(f"  Note: {n_promoted} FP16 layers promoted to GPTQ-8bit for format consistency")
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        map_path = os.path.join(output_path, "quantization_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "alpha": alpha,
                "prompt": prompt,
                "method": "GPTQ",
                "group_size": group_size,
                "calibration_samples": calibration_samples,
                "layers": quant_map
            }, f, indent=4)
        print(f"\n  ✓ Quantization map saved to {map_path}")
        alpha_str = str(alpha).replace('.', '')
        gptq_output = os.path.join(output_path, f"gptq_alpha_{alpha_str}")
        t0 = time.time()
        result = save_as_gptq(
            model_path=model_path,
            output_dir=gptq_output,
            quant_plan=quant_map,
            num_layers=num_layers,
            calibration_samples=calibration_samples,
            group_size=group_size,
            device=device,
        )
        t1 = time.time()
        if result:
            print(f"\n{'='*60}")
            print(f"  GPTQ quantization completed in {t1-t0:.1f}s")
            print(f"  Output: {gptq_output}")
            print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply prompt-conditioned quantization policy via GPTQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 4/8/16-bit mixed GPTQ quantization
  python apply_policy_gptq.py --model_path Devbora29/qwen2.5_3b_instruct_finetune --alpha 0.7
  
  # Aggressive compression
  python apply_policy_gptq.py --model_path Devbora29/qwen2.5_3b_instruct_finetune --alpha 0.5

  # Quality-first (more FP16 layers)
  python apply_policy_gptq.py --model_path Devbora29/qwen2.5_3b_instruct_finetune --alpha 0.9
  
  # More calibration samples for better quality
  python apply_policy_gptq.py --model_path Devbora29/qwen2.5_3b_instruct_finetune --calibration_samples 256
"""
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--policy_path", type=str, default="quant_policy_fullpolicy.pth",
                        help="Path to trained policy checkpoint")
    parser.add_argument("--output_path", type=str, default="./quantized_model",
                        help="Output directory for quantized model")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Quality-cost tradeoff (0.5=aggressive, 0.9=quality-first)")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity",
                        help="Prompt for conditioning the quantization policy")
    parser.add_argument("--calibration_samples", type=int, default=256,
                        help="Number of calibration samples for GPTQ Hessian computation")
    parser.add_argument("--group_size", type=int, default=128,
                        help="GPTQ group size (128 is standard, 32 for more granular)")
    parser.add_argument("--desc_act", action="store_true",
                        help="Use descending activation order (slower but sometimes better)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPTQ quantization on CPU is very slow. GPU recommended.")
    
    apply_policy(
        args.model_path, args.policy_path, args.output_path,
        args.alpha, args.prompt, device,
        calibration_samples=args.calibration_samples,
        group_size=args.group_size,
    )
