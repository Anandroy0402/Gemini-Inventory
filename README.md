# inventory-ai-demo

This Streamlit app audits inventory data with classical ML and optional Gemini models (classification and embeddings) to surface confidence, anomalies, and semantic duplicates.

## Gemini Integration

Set `ENABLE_GEMINI_MODELS=true` in the environment to enable the Gemini hosted models. Provide a key via `GEMINI_API_KEY` (environment variable or Streamlit secrets) for the Gemini API.

### Connection Diagnostics

The app includes comprehensive connection diagnostics:
- **Pre-flight checks**: DNS resolution and TCP connectivity tests before attempting API calls
- **API key validation**: Verifies key format (at least 20 characters)
- **Detailed error messages**: Specific error messages for different failure types (DNS, HTTP errors, timeouts, etc.)
- **Automatic retries**: Transient failures (503, 429, model loading) are automatically retried with exponential backoff
- **Diagnostic panel**: View real-time connection status in the Technical Methodology tab

### Common Issues and Solutions

**DNS Resolution Failed / Network Unreachable:**
- The Gemini API may be blocked by a corporate firewall or network policy
- Contact your network administrator to whitelist `generativelanguage.googleapis.com`
- The app will automatically fall back to local ML models

**Invalid API Key:**
- Ensure your API key is at least 20 characters long
- Obtain a valid key from https://aistudio.google.com/app/apikey

**Connection Timeout:**
- Network may be slow or API temporarily unreachable
- The app will automatically retry with exponential backoff
- If persistent, check your internet connectivity

### Setup Examples

**Option 1: Environment Variables**
```bash
export ENABLE_GEMINI_MODELS=true
export GEMINI_API_KEY="your_gemini_api_key"
```

**Option 2: Streamlit Secrets File**

Copy the example secrets file and fill in your values:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your actual key
```

Example `.streamlit/secrets.toml`:
```toml
ENABLE_GEMINI_MODELS = true
GEMINI_API_KEY = "your_actual_key_here"
```

**⚠️ Security Note**: Never commit `.streamlit/secrets.toml` to version control! This file is already in `.gitignore`. Avoid sharing API keys in PRs or issues; rotate keys immediately if exposed.

### Fallback Behavior

If the Gemini API cannot be reached, the app automatically falls back to local ML models:
- **Classification** → Rule-based categorization with TF-IDF clustering
- **Embeddings** → TF-IDF features for anomaly detection and clustering

The application will continue to function with reduced AI capabilities but full functionality for inventory analysis.
