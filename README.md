# inventory-ai-demo

This Streamlit app audits inventory data with classical ML and optional Hugging Face models (zero-shot classification and sentence-transformer embeddings) to surface confidence, anomalies, and semantic duplicates.

Set `ENABLE_HF_MODELS=true` in the environment to enable the Hugging Face hosted models. Provide a token via `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_API_TOKEN`, or `HUGGINGFACE_TOKEN` (environment variable or Streamlit secrets) for the Inference API. The app runs a lightweight Inference API connection test at startup and falls back to local signals if the hosted models cannot be reached.

Example environment setup:
```bash
export ENABLE_HF_MODELS=true
export HF_TOKEN="your_hf_token"
```

Or add the settings to `.streamlit/secrets.toml`:
```toml
ENABLE_HF_MODELS = true
HF_TOKEN = "your_hf_token"
```
