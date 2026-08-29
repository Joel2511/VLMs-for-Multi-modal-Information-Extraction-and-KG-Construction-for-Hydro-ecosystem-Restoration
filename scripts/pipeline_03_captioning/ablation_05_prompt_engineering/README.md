# Ablation 5: Prompt Engineering

This folder contains NVILA-Lite-8B prompt-engineering experiments.

| Script | Prompt configuration |
|---|---|
| `run_p0_baseline.py` | Baseline prompt |
| `run_p1_visual_primitives.py` | Visual primitive guardrail |
| `run_p2_verification_refusal.py` | Verification/refusal gate |
| `run_p3_counterfactual_contrast.py` | Counterfactual contrast |
| `run_p4_self_correction.py` | Internal self-correction |
| `run_p5_domain_expert.py` | Domain-expert persona |

The selected configuration for the final workflow is P3, Counterfactual
Contrast. It is used by `scripts/run_demo.sh`.

All other scripts remain important for reproducing the thesis ablation
comparison.
