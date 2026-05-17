
### Requirements

```bash
pip install torch transformers datasets numpy
pip install gptqmodel optimum
pip install autoawq accelerate pandas tabulate pyyaml tqdm
```

### Instructions
```bash
python generate_dataset.py -model "model_path" #check the file for more args
```

This produces:
- `*_fullpolicy.json` — the DPO training dataset
- `*_fullpolicy_metadata.json` — model/feature/scoring metadata


Point `DATA_JSON_PATH` and `META_JSON_PATH` in `train_and_quantize.py` to the outputs from Step 1:

```bash
python train.py
```

This produces:
- `quant_policy_fullpolicy.pth` — the trained policy checkpoint


```bash
python apply_policy.py \
  --model_path <HuggingFace model ID or local path> \
  --policy_path quant_policy_fullpolicy.pth \
  --alpha 0.7 \
  --output_path ./quantized_model
```
