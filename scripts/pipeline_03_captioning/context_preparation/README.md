# Captioning Context Preparation

These scripts prepare text context used by captioning experiments.

- `summarize_documents.py` generates document-level summaries using the
  configured Qwen2-VL summarisation model.
- `build_weighted_context.py` combines document summaries with image–text
  retrieval results to create per-image weighted context.

The selected end-to-end workflow runs both scripts before NVILA A5-P3
counterfactual-contrast captioning.

Generated summaries and weighted-context JSON files are written to `outputs/`
or to an explicitly supplied run-output directory.
