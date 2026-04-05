# Replication of Chaudhary et al. 2025 — New Model Families

Replication and extension of ["Evaluation Awareness Scales Predictably in Open-Weights Large Language Models"](https://arxiv.org/abs/2509.13333) (Chaudhary et al. 2025), extending the scaling law (Figure 1) to Qwen 3.5 and Gemma 4 families.

## What Changed from the Original Code

The original pipeline uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer`, which does not support Qwen 3.5 or Gemma 4 architectures. To handle these, we extracted the `HookedTransformerShim` class from the authors' own notebook (`probe_eval.ipynb`) and adapted it:

### HookedTransformerShim (`shim.py`)

The shim wraps any HuggingFace model via `AutoModelForCausalLM` and mimics the TransformerLens API by using `output_hidden_states=True` and mapping `blocks.{layer}.hook_resid_pre` to `hidden_states[layer]`.

Changes from the original notebook version:
- **Removed OLMo-specific code paths** (OLMoTokenizerFast, OLMoForCausalLM) -- not needed for our model families
- **Added `text_config` fallback** (`getattr(cfg_hf, "text_config", cfg_hf)`) -- Gemma 3/4 models nest their config under `text_config` rather than at the top level
- **Fixed device_map logic** -- original used `device_map="auto"` for any non-CPU device; changed to only use it for multi-GPU CUDA setups
- **Removed debug print statements**

### Combined Pipeline (`run.py`)

The original used two separate scripts (`generate_vectors.py` + `analyze_probe.py`). We combined them into a single `run.py` with:
- `--model` flag for the HuggingFace model ID
- `--chunk-size` flag (default 10) for multi-layer chunking in Stage 2

Multi-layer chunking hooks N layers per forward pass instead of 1, reducing forward passes by ~Nx with identical results (read-only hooks).

### No Methodology Changes

The methodology is identical to the original:
- Same contrastive dataset (203 prompt pairs)
- Same SAD stages_oversight evaluation (400 prompts)
- Same AUROC calculation (`sklearn.metrics.roc_curve` + `auc`)
- Same metric: `|AUROC - 0.5|` (absolute distance from chance)

## Known Discrepancy

**Gemma 3 270m-it**: Our replication produces a slightly different AUROC than the paper's reported value. The difference is small (~0.01) but present. We were unable to identify the source -- it may be due to differences in tokenizer behavior, floating point precision (we use float32), or minor version differences in the HuggingFace model weights.

## Models

| Family | Models | Status |
|--------|--------|--------|
| Gemma 3 (anchor) | 270m-it, 1b-it, 4b-it, 12b-it, 27b-it | Anchor for validation against paper |
| Qwen 3.5 | 0.8B, 2B, 4B, 9B, 27B | New family |
| Gemma 4 | E2B-it, E4B-it, 31B-it | New family |

## Results

<!-- Update this section after downloading results from GPU -->

```
model,size,AUROC,Layers,best_layer,dataset
```

Results are appended to `results/results.csv` as each model completes.

## How to Run

```bash
# 1. Install dependencies and set up SAD data
bash setup.sh

# 2. Run a single model
python3 run.py --model google/gemma-3-270m-it

# 3. Run all 13 models sequentially
bash run_all.sh
```

## Repo Structure

```
run.py                  # Combined Stage 1 (steering vectors) + Stage 2 (probe evaluation)
shim.py                 # HookedTransformerShim -- architecture-agnostic HF wrapper
run_all.sh              # Batch runner for all 13 models
setup.sh                # pip install + SAD data setup
datasets/               # Contrastive training data (203 prompt pairs)
results/                # Output: per-model vectors, probe scores, results.csv
original_code/          # Chaudhary et al. reference implementation
  src/                  # Core modules (vector_generation, probe_analysis, utils)
  scripts/              # Original CLI scripts (uses TransformerLens)
  notebooks/            # Exploratory notebooks
  probe_eval.ipynb      # Notebook containing the original HookedTransformerShim
  figures/              # Paper figures
```

## References

- Chaudhary et al. (2025). ["Evaluation Awareness Scales Predictably in Open-Weights Large Language Models."](https://arxiv.org/abs/2509.13333)
- Nguyen et al. (2025). ["Probing and Steering Evaluation Awareness of Language Models."](https://arxiv.org/abs/2507.01786)
