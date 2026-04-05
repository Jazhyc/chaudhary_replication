**Description**

Project: Mitigating and Evaluating Evaluation Awareness using Scaling Laws. We aim to emulate https://arxiv.org/pdf/2507.01786, testing on different model sizes and families. This repo includes the experiment pipeline needed to recreate the experiments.

Results: Linear probes trained with simple contrastive data generalised to distinguish real evaluation and deployment prompts.

**Usage**

1. Clone the repo
2. Open experiment pipeline ```probe_eval.ipynb```
3. Replace model variables with desired model
4. Follow steps to generate probes and AUROC scores
5. Use ```layer_absolute.py``` to visualize models and their highest scoring layer
6. Use ```model_absolute.py``` to visualize layers in each model
