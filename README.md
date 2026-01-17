# inventory-ai-demo

This Streamlit app audits inventory data with classical ML and optional Hugging Face models (zero-shot classification and sentence-transformer embeddings) to surface confidence, anomalies, and semantic duplicates.

## Hugging Face Integration

Set `ENABLE_HF_MODELS=true` in the environment to enable the Hugging Face hosted models. Provide a token via `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_API_TOKEN`, or `HUGGINGFACE_TOKEN` (environment variable or Streamlit secrets) for the Inference API.

### Connection Diagnostics

The app includes comprehensive connection diagnostics:
- **Pre-flight checks**: DNS resolution and TCP connectivity tests before attempting API calls
- **Token validation**: Verifies token format (should start with 'hf_' and be at least 20 characters)
- **Detailed error messages**: Specific error messages for different failure types (DNS, HTTP errors, timeouts, etc.)
- **Automatic retries**: Transient failures (503, 429, model loading) are automatically retried with exponential backoff
- **Diagnostic panel**: View real-time connection status in the Technical Methodology tab

### Common Issues and Solutions

**DNS Resolution Failed / Network Unreachable:**
- The Hugging Face API may be blocked by a corporate firewall or network policy
- Contact your network administrator to whitelist `api-inference.huggingface.co`
- The app will automatically fall back to local ML models

**Invalid Token:**
- Ensure your token starts with `hf_` and is at least 20 characters long
- Obtain a valid token from https://huggingface.co/settings/tokens

**Connection Timeout:**
- Network may be slow or API temporarily unreachable
- The app will automatically retry with exponential backoff
- If persistent, check your internet connectivity

### Setup Examples

**Option 1: Environment Variables**
```bash
export ENABLE_HF_MODELS=true
export HF_TOKEN="your_hf_token"
```

**Option 2: Streamlit Secrets File**

Copy the example secrets file and fill in your values:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your actual token
```

Example `.streamlit/secrets.toml`:
```toml
ENABLE_HF_MODELS = true
HF_TOKEN = "hf_your_actual_token_here"
```

**⚠️ Security Note**: Never commit `.streamlit/secrets.toml` to version control! This file is already in `.gitignore`.

### Fallback Behavior

If the Hugging Face API cannot be reached, the app automatically falls back to local ML models:
- **Zero-shot classification** → Rule-based categorization with TF-IDF clustering
- **Embeddings** → TF-IDF features for anomaly detection and clustering

The application will continue to function with reduced AI capabilities but full functionality for inventory analysis.
