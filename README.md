# inventory-ai-demo

This Streamlit app audits inventory data with classical ML and optional Hugging Face models (zero-shot classification and sentence-transformer embeddings) to surface confidence, anomalies, and semantic duplicates.

Set `ENABLE_HF_MODELS=true` in the environment to enable the Hugging Face hosted models. Provide a token via `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` (environment variable or Streamlit secrets) for the Inference API.
