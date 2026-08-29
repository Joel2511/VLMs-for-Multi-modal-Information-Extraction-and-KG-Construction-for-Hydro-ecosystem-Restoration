# Ablation 2: Retrieval-Context Captioning

This folder contains captioning scripts that use retrieved text context for
image-grounded caption generation.

The scripts evaluate the same candidate vision-language model families under a
retrieval-context configuration. This ablation supports model selection before
the later summary, weighted-context, prompt-engineering, and decoding studies.

The selected final workflow uses NVILA-Lite-8B and is demonstrated through the
A5-P3 script in `../ablation_05_prompt_engineering/`.
