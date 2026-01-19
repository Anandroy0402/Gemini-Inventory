import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
try:
    import tomllib
except ImportError:
    # Fallback for Python < 3.11
    import tomli as tomllib
import time
from pathlib import Path
from difflib import SequenceMatcher
from urllib import request, error
try:
    from streamlit.errors import StreamlitSecretNotFoundError
except ImportError:
    class StreamlitSecretNotFoundError(RuntimeError):
        """Fallback when Streamlit secret errors are unavailable."""

# Advanced AI/ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor Pro", layout="wide", page_icon="🛡️")

def get_streamlit_secrets():
    """
    Load secrets from .streamlit/secrets.toml file.
    
    Returns:
        dict: Dictionary of secrets from the TOML file. Returns empty dict if file
              doesn't exist or cannot be parsed.
    
    Note:
        This function safely handles missing files and parse errors by returning
        an empty dictionary rather than raising exceptions.
    """
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        with secrets_path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

def resolve_bool_setting(key, default=False):
    """
    Resolve a boolean setting from environment variables or secrets.
    
    Checks in order:
    1. Environment variable
    2. File-based secrets (.streamlit/secrets.toml)
    3. Streamlit secrets (st.secrets)
    4. Default value
    
    Args:
        key: The setting key to look up
        default: Default value if key is not found (default: False)
    
    Returns:
        bool: The resolved boolean value
    
    Note:
        String values are converted to boolean by checking if they equal "true"
        (case-insensitive after stripping whitespace).
    """
    value = os.getenv(key)
    if value is None:
        # Try file-based secrets first (more reliable)
        secrets = get_streamlit_secrets()
        if key in secrets:
            value = secrets[key]
        else:
            # Fall back to st.secrets as last resort
            try:
                value = st.secrets.get(key)
            except (AttributeError, KeyError):
                value = None
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
DEFAULT_PRODUCT_GROUP = "Consumables & General"
MIN_DISTANCE_THRESHOLD = 1e-8  # Replace zero distances to avoid divide-by-zero in confidence calculations.
COMPARISON_WINDOW_SIZE = 50  # Windowed comparisons keep duplicate checks lightweight.
FUZZY_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_SIMILARITY_THRESHOLD = 0.9
GEMINI_BATCH_SIZE = 16
GEMINI_CLASSIFICATION_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "text-embedding-004"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_API_HOSTNAME = "generativelanguage.googleapis.com"
GEMINI_API_TIMEOUT = 30
ENABLE_GEMINI_MODELS = resolve_bool_setting("ENABLE_GEMINI_MODELS", default=False)
GEMINI_CONFIDENCE_MIN_THRESHOLD = 0.8
GEMINI_CONFIDENCE_MIN_TARGET = 0.6
GEMINI_CONFIDENCE_MAX_TARGET = 0.98
GEMINI_CONNECTION_CACHE_TTL = 30
GEMINI_CONNECTION_TEST_TEXT = "Inventory audit connection check."
GEMINI_MAX_RETRIES = 2  # Number of retries for transient failures
GEMINI_RETRY_DELAY = 2  # Base delay between retries in seconds
GEMINI_MAX_RETRY_DELAY = 10  # Maximum delay cap for exponential backoff
GEMINI_API_KEY_MIN_LENGTH = 20  # Minimum length for valid Gemini API keys
GEMINI_MAX_WORKERS = 2
GEMINI_REQUEST_DELAY = 0.1  # Throttle concurrent calls to avoid API rate limits.
GEMINI_LABEL_SIMILARITY_THRESHOLD = 0.8  # Similarity floor for choosing closest label match.
GEMINI_API_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
GEMINI_REQUEST_SEMAPHORE = threading.Semaphore(GEMINI_MAX_WORKERS)
GEMINI_API_KEY_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY"
)

PRODUCT_GROUPS = {
    "Piping & Fittings": ["FLANGE", "PIPE", "ELBOW", "TEE", "UNION", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING", "UPVC", "CPVC", "PVC"],
    "Valves & Actuators": ["BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "GLOBE VALVE", "CONTROL VALVE", "VALVE", "ACTUATOR", "COCK"],
    "Fasteners & Seals": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O RING", "MECHANICAL SEAL", "SEAL", "JOINT"],
    "Electrical & Instruments": ["TRANSMITTER", "CABLE", "WIRE", "GAUGE", "SENSOR", "CONNECTOR", "SWITCH", "TERMINAL", "INSTRUMENT", "CAMERA"],
    "Tools & Hardware": ["PLIER", "CUTTING PLIER", "STRIPPER", "WIRE STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER", "TAPE MEASURE", "MEASURING TAPE", "BIT", "DRILL BIT"],
    "Consumables & General": ["BRUSH", "PAINT BRUSH", "TAPE", "ADHESIVE", "HOSE", "SAFETY GLOVE", "GLOVE", "CLEANER", "PAINT", "CEMENT", "STICKER", "CHALK"],
    "Specialized Spares": ["FILTER", "BEARING", "PUMP", "MOTOR", "CARTRIDGE", "IMPELLER", "SPARE"]
}

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"]
}

# --- AI UTILITIES ---
def clean_description(text):
    text = str(text).upper().replace('"', ' ')
    text = text.replace("O-RING", "O RING")
    text = text.replace("MECH-SEAL", "MECHANICAL SEAL").replace("MECH SEAL", "MECHANICAL SEAL")
    text = re.sub(r'[^A-Z0-9\s./-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def token_pattern(token):
    return rf'(?<!\w){re.escape(token)}(?!\w)'

def get_tech_dna(text):
    text = clean_description(text)
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(token_pattern(k), text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

def intelligent_noun_extractor(text):
    text = clean_description(text)
    phrases = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "MECHANICAL SEAL", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER", "DRILL BIT"]
    for p in phrases:
        if re.search(token_pattern(p), text): return p
    all_nouns = [item for sublist in PRODUCT_GROUPS.values() for item in sublist]
    for n in all_nouns:
        if re.search(token_pattern(n), text): return n
    return text.split()[0] if text.split() else "MISC"

def map_product_group(noun):
    for group, keywords in PRODUCT_GROUPS.items():
        if noun in keywords:
            return group
    for group, keywords in PRODUCT_GROUPS.items():
        for keyword in keywords:
            if re.search(token_pattern(keyword), noun):
                return group
    return DEFAULT_PRODUCT_GROUP

def dominant_group(series):
    counts = series.value_counts()
    return counts.idxmax() if not counts.empty else "UNMAPPED"

def apply_distance_floor(distances, min_threshold=MIN_DISTANCE_THRESHOLD):
    max_dist = np.max(distances, axis=1)
    return np.where(max_dist == 0, min_threshold, max_dist)

def normalize_confidence_scores(scores):
    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores)
    if scores.empty:
        return scores
    max_score = scores.max()
    if max_score >= GEMINI_CONFIDENCE_MIN_THRESHOLD:
        return scores
    min_score = scores.min()
    if max_score == min_score:
        return pd.Series(np.full(len(scores), max(max_score, GEMINI_CONFIDENCE_MIN_TARGET)), index=scores.index)
    scaled = (scores - min_score) / (max_score - min_score)
    return (scaled * (GEMINI_CONFIDENCE_MAX_TARGET - GEMINI_CONFIDENCE_MIN_TARGET) + GEMINI_CONFIDENCE_MIN_TARGET).round(4)

def build_fuzzy_duplicates(df, id_col):
    fuzzy_list = []
    recs = df.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, min(i + COMPARISON_WINDOW_SIZE, len(recs))):
            r1, r2 = recs[i], recs[j]
            desc1 = r1.get('Standard_Desc') or ''
            desc2 = r2.get('Standard_Desc') or ''
            sim = SequenceMatcher(None, desc1, desc2).ratio()
            if sim > FUZZY_SIMILARITY_THRESHOLD:
                dna1 = r1.get('Tech_DNA') or {'numbers': set(), 'attributes': {}}
                dna2 = r2.get('Tech_DNA') or {'numbers': set(), 'attributes': {}}
                is_variant = (dna1['numbers'] != dna2['numbers']) or (dna1['attributes'] != dna2['attributes'])
                fuzzy_list.append({
                    'ID A': r1[id_col], 'ID B': r2[id_col],
                    'Desc A': desc1, 'Desc B': desc2,
                    'Match %': f"{sim:.1%}", 'Verdict': "🛠️ Variant" if is_variant else "🚨 Duplicate"
                })
    return fuzzy_list

def get_gemini_secret(key):
    secrets = get_streamlit_secrets()
    if key in secrets:
        return secrets[key]
    try:
        return st.secrets[key]
    except (AttributeError, KeyError, StreamlitSecretNotFoundError):
        return None

def get_gemini_api_key():
    for key in GEMINI_API_KEY_KEYS:
        api_key = os.getenv(key) or get_gemini_secret(key)
        if api_key:
            api_key = str(api_key).strip()
            if api_key:
                return api_key
    return None

def validate_gemini_api_key(api_key):
    """Validate that the API key looks like a valid Gemini API key."""
    if not api_key:
        return False
    api_key = str(api_key).strip()
    if len(api_key) < GEMINI_API_KEY_MIN_LENGTH:
        return False
    if any(ord(char) < 32 for char in api_key):
        return False
    return bool(GEMINI_API_KEY_PATTERN.search(api_key))

def check_dns_resolution(hostname):
    """Check if a hostname can be resolved via DNS."""
    try:
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, socket.herror, OSError):
        return False

def check_gemini_api_connectivity():
    """
    Check if Gemini API is accessible.
    Returns a tuple: (is_accessible, error_message)
    """
    hostname = GEMINI_API_HOSTNAME
    
    # First check DNS resolution
    if not check_dns_resolution(hostname):
        return False, "dns_resolution_failed"
    
    # Try to establish a connection
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((hostname, 443))
        return True, None
    except socket.timeout:
        return False, "connection_timeout"
    except (socket.gaierror, socket.herror):
        return False, "dns_resolution_failed"
    except ConnectionRefusedError:
        return False, "connection_refused"
    except OSError as e:
        return False, f"network_error: {str(e)}"
    finally:
        if sock:
            try:
                sock.close()
            except (OSError, socket.error):
                # Ignore errors during cleanup
                pass

def call_gemini_api(endpoint, payload, api_key, warning_message, show_warnings=True, retry_count=0):
    """
    Call Gemini API with improved error handling and retry logic.
    Returns None on failure, result on success.
    """
    if not api_key:
        return None
    api_key = str(api_key).strip()
    if not validate_gemini_api_key(api_key):
        if show_warnings:
            st.warning(f"{warning_message}: Invalid API key format")
        return None
    
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{GEMINI_API_BASE_URL}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key}
    )
    
    try:
        with request.urlopen(req, timeout=GEMINI_API_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
        
        # Check if API returned an error in the response
        if isinstance(result, dict) and result.get("error"):
            error_info = result.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            
            if show_warnings:
                st.warning(f"{warning_message}: API returned error - {error_msg}")
            return None
        
        return result
        
    except error.HTTPError as exc:
        # HTTP errors (4xx, 5xx)
        detail = f"HTTP {exc.code}"
        is_retryable = exc.code in {429, 500, 502, 503}
        
        if exc.code == 401:
            detail += " - Invalid or expired API key"
        elif exc.code == 403:
            detail += " - Access forbidden"
        elif exc.code == 429:
            detail += " - Rate limit exceeded"
        elif exc.code == 503:
            detail += " - Service temporarily unavailable"
        elif exc.code >= 500:
            detail += " - Server error"
        
        # Retry on transient errors with exponential backoff and cap
        if is_retryable and retry_count < GEMINI_MAX_RETRIES:
            delay = min(GEMINI_RETRY_DELAY * (2 ** retry_count), GEMINI_MAX_RETRY_DELAY)
            time.sleep(delay)
            return call_gemini_api(endpoint, payload, api_key, warning_message, show_warnings, retry_count + 1)
        
        if show_warnings:
            st.warning(f"{warning_message}: {detail}")
        return None
        
    except error.URLError as exc:
        # Network/DNS errors - typically not retryable
        error_reason = str(exc.reason)
        
        if isinstance(exc.reason, socket.gaierror):
            # DNS resolution failure
            detail = "DNS resolution failed - Gemini API may be blocked by firewall or network"
        elif isinstance(exc.reason, socket.timeout):
            detail = "Connection timeout - Network may be slow or API unreachable"
        elif "Connection refused" in error_reason:
            detail = "Connection refused - Service may be down"
        elif "Network is unreachable" in error_reason:
            detail = "Network unreachable - Check network connectivity"
        else:
            detail = f"Network error - {error_reason}"
        
        if show_warnings:
            st.warning(f"{warning_message}: {detail}")
        return None
        
    except socket.timeout:
        # Explicit timeout - could be transient
        if retry_count < GEMINI_MAX_RETRIES:
            delay = min(GEMINI_RETRY_DELAY * (2 ** retry_count), GEMINI_MAX_RETRY_DELAY)
            time.sleep(delay)
            return call_gemini_api(endpoint, payload, api_key, warning_message, show_warnings, retry_count + 1)
        
        detail = "Request timeout - API response took too long"
        if show_warnings:
            st.warning(f"{warning_message}: {detail}")
        return None
        
    except ValueError as exc:
        # JSON parsing error
        detail = f"Invalid response format - {exc}"
        if show_warnings:
            st.warning(f"{warning_message}: {detail}")
        return None

def extract_json_from_text(text):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        start = cleaned.index("{")
    except ValueError:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start:idx + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None

def normalize_gemini_label(label, labels):
    if not label:
        return None
    label_text = str(label).strip().lower()
    lowered_labels = [(candidate, candidate.lower()) for candidate in labels]
    best_match = None
    best_similarity = 0.0
    for candidate, candidate_text in lowered_labels:
        if label_text == candidate_text:
            return candidate
        similarity = SequenceMatcher(None, label_text, candidate_text).ratio()
        if similarity >= GEMINI_LABEL_SIMILARITY_THRESHOLD and similarity > best_similarity:
            best_similarity = similarity
            best_match = candidate
    return best_match

def parse_gemini_classification_response(response_text, labels):
    payload = extract_json_from_text(response_text)
    label = None
    confidence = 0.0
    if isinstance(payload, dict):
        label = payload.get("label") or payload.get("category") or payload.get("product_group")
        confidence = payload.get("confidence", payload.get("score", 0.0))
    elif response_text:
        label = response_text.strip()
    
    label = normalize_gemini_label(label, labels)
    if not label:
        return None
    
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return {"labels": [label], "scores": [confidence]}

def call_gemini_generate(prompt, api_key, warning_message, show_warnings=True):
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 128
        }
    }
    result = call_gemini_api(
        f"models/{GEMINI_CLASSIFICATION_MODEL}:generateContent",
        payload,
        api_key,
        warning_message,
        show_warnings=show_warnings
    )
    if not result:
        return None
    candidates = result.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    return text.strip() if text else None

def call_gemini_embedding(text, api_key, warning_message, show_warnings=True):
    payload = {"content": {"parts": [{"text": text}]}}
    result = call_gemini_api(
        f"models/{GEMINI_EMBEDDING_MODEL}:embedContent",
        payload,
        api_key,
        warning_message,
        show_warnings=show_warnings
    )
    if not result:
        return None
    embedding = result.get("embedding", {}).get("values")
    return embedding

def build_gemini_prompt(text, labels):
    labels_text = ", ".join(labels)
    return (
        "You are classifying an industrial inventory item. "
        f"Choose the single best category from this list: {labels_text}. "
        "Respond only with JSON in the form "
        '{"label": "<category>", "confidence": <number between 0 and 1>}. '
        f"Item: {text}"
    )

def run_gemini_classification(texts, labels):
    api_key = get_gemini_api_key()
    if not api_key:
        st.warning("Gemini API key missing; skipping hosted classification.")
        return None
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return None
    max_workers = min(GEMINI_MAX_WORKERS, GEMINI_BATCH_SIZE, len(texts))
    def classify_text(text):
        prompt = build_gemini_prompt(text, labels)
        with GEMINI_REQUEST_SEMAPHORE:
            time.sleep(GEMINI_REQUEST_DELAY)
            response_text = call_gemini_generate(
                prompt,
                api_key,
                "Gemini classification failed; using existing categories.",
                show_warnings=False
            )
        return parse_gemini_classification_response(response_text, labels)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(classify_text, texts))
        if any(result is None for result in results):
            st.warning("Gemini classification failed; using existing categories.")
            return None
        return results
    except (RuntimeError, ValueError) as exc:
        st.warning(f"Gemini classification failed; using existing categories. ({exc})")
        return None
    except Exception as exc:
        st.warning(f"Gemini classification failed; using existing categories. ({exc})")
        return None

def compute_embeddings(texts):
    api_key = get_gemini_api_key()
    if not api_key:
        st.warning("Gemini API key missing; skipping hosted embeddings.")
        return None
    if not texts:
        return None
    max_workers = min(GEMINI_MAX_WORKERS, GEMINI_BATCH_SIZE, len(texts))
    def embed_text(text):
        with GEMINI_REQUEST_SEMAPHORE:
            time.sleep(GEMINI_REQUEST_DELAY)
            return call_gemini_embedding(
                text,
                api_key,
                "Embedding generation failed; falling back to TF-IDF signals.",
                show_warnings=False
            )
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(embed_text, texts))
        if any(embedding is None for embedding in embeddings):
            st.warning("Embedding generation failed; falling back to TF-IDF signals.")
            return None
        embeddings = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    except (RuntimeError, ValueError) as exc:
        st.warning(f"Embedding generation failed; falling back to TF-IDF signals. ({exc})")
        return None
    except Exception as exc:
        st.warning(f"Embedding generation failed; falling back to TF-IDF signals. ({exc})")
        return None

def is_valid_classification_item(item):
    if not isinstance(item, dict):
        return False
    labels = item.get("labels")
    scores = item.get("scores")
    return isinstance(labels, list) and isinstance(scores, list) and bool(labels) and bool(scores)

def is_valid_classification_response(result):
    if isinstance(result, dict):
        return is_valid_classification_item(result)
    if isinstance(result, list) and result:
        return all(is_valid_classification_item(item) for item in result)
    return False

def is_valid_embedding_response(result):
    if not isinstance(result, list) or not result:
        return False
    if all(isinstance(x, (int, float)) for x in result):
        return True
    if isinstance(result[0], list) and result[0]:
        return all(isinstance(x, (int, float)) for x in result[0])
    return False

@st.cache_data(ttl=GEMINI_CONNECTION_CACHE_TTL)
def test_gemini_inference_connection(enable_gemini_models):
    """
    Test connection to Gemini API with comprehensive diagnostics.
    Returns a dict with connection status and detailed error information.
    """
    if not enable_gemini_models:
        return {
            "enabled": False,
            "classification": False,
            "embeddings": False,
            "reason": "disabled",
            "status": "disabled",
            "error_detail": None
        }
    
    # Check if API key exists
    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "enabled": False,
            "classification": False,
            "embeddings": False,
            "reason": "missing_key",
            "status": "missing_key",
            "error_detail": "No Gemini API key found in environment or secrets"
        }
    
    # Validate API key format
    if not validate_gemini_api_key(api_key):
        return {
            "enabled": False,
            "classification": False,
            "embeddings": False,
            "reason": "invalid_key",
            "status": "invalid_key",
            "error_detail": f"API key format is invalid (should be at least {GEMINI_API_KEY_MIN_LENGTH} characters)"
        }
    
    # Check network connectivity to Gemini API
    is_accessible, conn_error = check_gemini_api_connectivity()
    if not is_accessible:
        error_details = {
            "dns_resolution_failed": f"Cannot resolve {GEMINI_API_HOSTNAME} - May be blocked by firewall or network policy",
            "connection_timeout": "Connection timeout - Network may be slow or API unreachable",
            "connection_refused": "Connection refused - Service may be down or blocked",
        }
        error_detail = error_details.get(conn_error, f"Network connectivity issue: {conn_error}")
        
        return {
            "enabled": False,
            "classification": False,
            "embeddings": False,
            "reason": "network_unreachable",
            "status": "network_unreachable",
            "error_detail": error_detail,
            "connectivity_error": conn_error
        }
    
    # Test classification model
    test_text = GEMINI_CONNECTION_TEST_TEXT
    prompt = build_gemini_prompt(test_text, list(PRODUCT_GROUPS.keys()))
    classification_text = call_gemini_generate(
        prompt,
        api_key,
        "Gemini classification test failed",
        show_warnings=False
    )
    classification_result = parse_gemini_classification_response(classification_text, list(PRODUCT_GROUPS.keys()))
    classification_ok = is_valid_classification_response(classification_result)
    
    # Test embedding model
    embedding_result = call_gemini_embedding(
        test_text,
        api_key,
        "Gemini embedding test failed",
        show_warnings=False
    )
    embedding_ok = is_valid_embedding_response(embedding_result)
    
    # Determine overall status
    if classification_ok and embedding_ok:
        status = "full"
        error_detail = None
    elif classification_ok or embedding_ok:
        status = "partial"
        failed_models = []
        if not classification_ok:
            failed_models.append("classification")
        if not embedding_ok:
            failed_models.append("embeddings")
        error_detail = f"Some models unavailable: {', '.join(failed_models)}"
    else:
        status = "unavailable"
        error_detail = "Both classification and embedding models failed to respond"
    
    enabled = status in {"full", "partial"}
    reason = None if enabled else "inference_test_failed"
    
    return {
        "enabled": enabled,
        "classification": classification_ok,
        "embeddings": embedding_ok,
        "reason": reason,
        "status": status,
        "error_detail": error_detail
    }

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path, enable_gemini_classification=False, enable_gemini_embeddings=False):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    df['Standard_Desc'] = df[desc_col].apply(clean_description)
    df['Part_Noun'] = df['Standard_Desc'].apply(intelligent_noun_extractor)
    df['Product_Group'] = df['Part_Noun'].apply(map_product_group)

    # NLP & Topic Modeling
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    
    # Clustering for Confidence
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    max_tfidf_dist = apply_distance_floor(dists)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / max_tfidf_dist)).round(4)
    cluster_groups = df.groupby('Cluster_ID')['Product_Group'].agg(dominant_group)
    df['Cluster_Group'] = df['Cluster_ID'].map(cluster_groups)
    df['Cluster_Validated'] = df['Product_Group'] == df['Cluster_Group']
    
    # Anomaly
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(tfidf_matrix) # Using tfidf for complexity-based anomalies

    standard_desc = df['Standard_Desc'].tolist() if enable_gemini_embeddings else None
    gemini_inputs = (
        df['Part_Noun']
        .fillna('')
        .str.cat(df['Standard_Desc'].fillna(''), sep=' ')
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .tolist()
        if enable_gemini_classification else None
    )
    
    # Gemini Classification
    gemini_results = run_gemini_classification(gemini_inputs, list(PRODUCT_GROUPS.keys())) if enable_gemini_classification else None
    if gemini_results:
        df['Gemini_Product_Group'] = [res['labels'][0] for res in gemini_results]
        df['Gemini_Product_Confidence'] = [round(res['scores'][0], 4) for res in gemini_results]
    else:
        df['Gemini_Product_Group'] = df['Product_Group']
        df['Gemini_Product_Confidence'] = df['Confidence']
    df['Gemini_Product_Confidence'] = normalize_confidence_scores(df['Gemini_Product_Confidence'])
    
    # Gemini Embeddings for Clustering/Anomaly
    embeddings = compute_embeddings(standard_desc) if enable_gemini_embeddings else None
    if embeddings is not None:
        kmeans_gemini = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['Gemini_Cluster_ID'] = kmeans_gemini.fit_predict(embeddings)
        gemini_dists = kmeans_gemini.transform(embeddings)
        max_dist = apply_distance_floor(gemini_dists)
        df['Gemini_Cluster_Confidence'] = (1 - (np.min(gemini_dists, axis=1) / max_dist)).round(4)
        iso_gemini = IsolationForest(contamination=0.04, random_state=42)
        df['Gemini_Anomaly_Flag'] = iso_gemini.fit_predict(embeddings)
        df['Gemini_Embedding'] = list(embeddings)
    else:
        df['Gemini_Cluster_ID'] = df['Cluster_ID']
        df['Gemini_Cluster_Confidence'] = df['Confidence']
        df['Gemini_Anomaly_Flag'] = df['Anomaly_Flag']
        df['Gemini_Embedding'] = [None] * len(df)

    # Fuzzy & Tech DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)

    return df, id_col, desc_col

# --- DATA LOADING ---
gemini_status = test_gemini_inference_connection(ENABLE_GEMINI_MODELS)
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(
        target_file,
        enable_gemini_classification=gemini_status["classification"],
        enable_gemini_embeddings=gemini_status["embeddings"]
    )
else:
    st.error("Data file missing from repository. Please ensure 'raw_data.csv' is present.")
    st.stop()

# Filter defaults
group_options = list(PRODUCT_GROUPS.keys())

# --- HEADER & MODERN NAVIGATION ---
st.title("🛡️ AI Inventory Auditor Pro")
st.markdown("### Advanced Inventory Intelligence & Quality Management")

# Display Gemini connection status with detailed messaging
if gemini_status["enabled"]:
    enabled_features = []
    if gemini_status["classification"]:
        enabled_features.append("classification")
    if gemini_status["embeddings"]:
        enabled_features.append("embeddings")
    feature_label = ", ".join(enabled_features)
    status_label = gemini_status.get("status", "partial")
    if status_label not in {"full", "partial"}:
        status_label = "partial"
    st.success(f"✅ Gemini API connected ({status_label}: {feature_label}).")
    
    # Show warning if partial connectivity
    if status_label == "partial" and gemini_status.get("error_detail"):
        st.info(f"ℹ️ Note: {gemini_status['error_detail']}")

elif gemini_status["reason"] == "disabled":
    st.info("ℹ️ **Gemini models disabled.** Using local ML models for analysis.\n\n"
            "To enable hosted inference:\n"
            "- Set `ENABLE_GEMINI_MODELS=true` in environment or `.streamlit/secrets.toml`\n"
            "- Provide a key via `GEMINI_API_KEY` environment variable or secrets")

elif gemini_status["reason"] == "missing_key":
    st.warning("⚠️ **Gemini API key missing.** Using local signals instead of hosted inference.\n\n"
               "To enable hosted models:\n"
               "- Create a Gemini API key in Google AI Studio\n"
               "- Set via environment variable: `GEMINI_API_KEY=your_key`\n"
               "- Or add to `.streamlit/secrets.toml`: `GEMINI_API_KEY = \"your_key\"`")

elif gemini_status["reason"] == "invalid_key":
    st.warning("⚠️ **Gemini API key format is invalid.** Using local signals instead.\n\n"
               f"Details: {gemini_status.get('error_detail', 'API key validation failed')}\n\n"
               "Valid keys should:\n"
               f"- Be at least {GEMINI_API_KEY_MIN_LENGTH} characters long\n"
               "- Obtain from Google AI Studio")

elif gemini_status["reason"] == "network_unreachable":
    st.error("🚫 **Cannot reach Gemini API.** Using local signals instead.\n\n"
             f"**Issue:** {gemini_status.get('error_detail', 'Network connectivity problem')}\n\n"
             "**Possible causes:**\n"
             f"- Corporate firewall blocking {GEMINI_API_HOSTNAME}\n"
             "- Network policy restrictions\n"
             "- DNS resolution issues\n"
             "- Internet connectivity problems\n\n"
             "**Resolution:**\n"
             f"- Contact your network administrator to whitelist {GEMINI_API_HOSTNAME}\n"
             "- Check your network/firewall settings\n"
             "- Verify internet connectivity")

else:
    # Generic failure
    error_detail = gemini_status.get('error_detail', 'Connection test failed')
    st.warning(f"⚠️ **Gemini API connection test failed.** Using local signals instead.\n\n"
               f"Details: {error_detail}\n\n"
               "The application will continue using local ML models for analysis.")

# Modern horizontal tab navigation
page = st.tabs(["📈 Executive Dashboard", "📍 Categorization Audit", "🚨 Quality Hub (Anomalies/Dups)", "🧠 Technical Methodology", "🧭 My Approach"])

# --- PAGE: EXECUTIVE DASHBOARD ---
with page[0]:
    st.markdown("#### 📊 Inventory Health Overview")
    st.markdown("Get a bird's eye view of your inventory data quality and distribution.")
    
    # Filters at the top
    with st.container():
        st.markdown("##### 🔍 Filters")
        selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="dash_group")
    
    # Apply Filters
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    
    # KPI Row
    fuzzy_list = build_fuzzy_duplicates(df, id_col)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📦 SKUs Analyzed", len(df))
    kpi2.metric("🎯 Mean Gemini Confidence", f"{df['Gemini_Product_Confidence'].mean():.1%}")
    kpi3.metric("⚠️ Gemini Anomalies Found", len(df[df['Gemini_Anomaly_Flag'] == -1]))
    kpi4.metric("🔄 Duplicate Pairs", len(fuzzy_list))

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='Gemini_Product_Group', title="Inventory Distribution by Gemini Product Category", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        top_nouns = df['Part_Noun'].value_counts().head(10).reset_index()
        fig_bar = px.bar(top_nouns, x='Part_Noun', y='count', title="Top 10 Product Categories", labels={'Part_Noun':'Product', 'count':'Qty'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Health Gauge
    health_val = (len(df[df['Gemini_Anomaly_Flag'] == 1]) / len(df)) * 100
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_val,
        title = {'text': "Catalog Data Accuracy %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### 💼 Business Insights")
    insights = (
        df.groupby('Gemini_Product_Group', dropna=False)
        .agg(
            Items=(id_col, 'count'),
            Mean_Gemini_Confidence=('Gemini_Product_Confidence', 'mean'),
            Gemini_Anomaly_Rate=('Gemini_Anomaly_Flag', lambda x: (x == -1).mean())
        )
        .reset_index()
        .sort_values('Items', ascending=False)
    )
    insights['Mean_Gemini_Confidence'] = insights['Mean_Gemini_Confidence'].round(3)
    insights['Gemini_Anomaly_Rate'] = insights['Gemini_Anomaly_Rate'].round(3)
    st.dataframe(insights, use_container_width=True, height=260)
    fig_insights = px.bar(
        insights.head(10),
        x='Gemini_Product_Group',
        y='Mean_Gemini_Confidence',
        title="Top Gemini Categories by Confidence",
        labels={'Gemini_Product_Group': 'Gemini Category', 'Mean_Gemini_Confidence': 'Mean Confidence'}
    )
    st.plotly_chart(fig_insights, use_container_width=True)

# --- PAGE: CATEGORIZATION AUDIT ---
with page[1]:
    st.markdown("#### 📍 AI Categorization & Filtered Audit")
    st.markdown("Drill down into specific product categories with intelligent filtering.")
    
    # Filters at the top of the table
    with st.container():
        st.markdown("##### 🔍 Filters")
        selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="cat_group")
    
    # Apply Filters
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    st.markdown(f"**Showing {len(df)} items**")
    
    # Data Table with sorting
    st.dataframe(
        df[
            [
                id_col,
                'Standard_Desc',
                'Part_Noun',
                'Product_Group',
                'Gemini_Product_Group',
                'Gemini_Product_Confidence',
                'Confidence'
            ]
        ].sort_values('Gemini_Product_Confidence', ascending=False),
        use_container_width=True,
        height=400
    )
    
    summary = (
        df.groupby('Product_Group', dropna=False)
        .agg(
            Items=(id_col, 'count'),
            Mean_Confidence=('Confidence', 'mean'),
            Mean_Gemini_Confidence=('Gemini_Product_Confidence', 'mean'),
            Cluster_Match_Rate=('Cluster_Validated', 'mean')
        )
        .reset_index()
        .sort_values('Items', ascending=False)
    )
    summary['Mean_Confidence'] = summary['Mean_Confidence'].round(3)
    summary['Mean_Gemini_Confidence'] = summary['Mean_Gemini_Confidence'].round(3)
    summary['Cluster_Match_Rate'] = summary['Cluster_Match_Rate'].round(3)
    st.markdown("#### 📌 Category Distribution & Confidence")
    st.dataframe(summary, use_container_width=True, height=260)

    # Distribution of confidence
    fig_hist = px.histogram(df, x="Gemini_Product_Confidence", nbins=20, title="Gemini Confidence Score Distribution", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)

# --- PAGE: QUALITY HUB ---
with page[2]:
    st.markdown("#### 🚨 Anomaly & Duplicate Identification")
    st.markdown("Identify quality issues and potential duplicates in your inventory data.")
    
    # Filters at the top
    with st.container():
        st.markdown("##### 🔍 Filters")
        selected_group = st.multiselect("Product Category", options=group_options, default=group_options, key="qual_group")
    
    # Apply Filters
    df = df_raw[df_raw['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    
    t1, t2, t3 = st.tabs(["⚠️ Gemini Anomalies", "👯 Fuzzy Duplicates", "🧠 Semantic Duplicates"])
    
    with t1:
        st.subheader("Gemini Embedding Anomalies (Isolation Forest)")
        anoms = df[df['Gemini_Anomaly_Flag'] == -1]
        st.warning(f"Found {len(anoms)} anomalies in the current view.")
        st.dataframe(
            anoms[[id_col, desc_col, 'Part_Noun', 'Gemini_Product_Group', 'Gemini_Cluster_Confidence']],
            use_container_width=True,
            height=400
        )
        
    with t2:
        st.subheader("Fuzzy Duplicate Audit (Spec-Aware)")
        st.info("System identifies items with >85% text similarity but differentiates based on numeric specs (Size/Gender).")
        
        # Calculate fuzzy duplicates for the current view
        fuzzy_list = build_fuzzy_duplicates(df, id_col)
        
        if fuzzy_list:
            st.dataframe(pd.DataFrame(fuzzy_list), use_container_width=True, height=400)
        else:
            st.success("No fuzzy duplicates found in this filtered view.")

    with t3:
        st.subheader("Semantic Duplicate Audit (Gemini Embeddings)")
        if df['Gemini_Embedding'].apply(lambda x: x is None).all():
            st.info("Semantic duplicate detection unavailable (Gemini embeddings not loaded).")
        else:
            records = df.reset_index(drop=True)
            if records['Gemini_Embedding'].apply(lambda x: x is None).any():
                st.info("Semantic duplicate detection unavailable (Gemini embeddings incomplete).")
            else:
                sem_list = []
                recs = records.to_dict('records')
                embeddings = records['Gemini_Embedding'].tolist()
                window_size = COMPARISON_WINDOW_SIZE  # Keep comparisons lightweight for UI responsiveness.
                for i in range(len(recs)):
                    for j in range(i + 1, min(i + window_size, len(recs))):
                        sim = float(np.dot(embeddings[i], embeddings[j]))  # Cosine similarity on normalized embeddings.
                        if sim > SEMANTIC_SIMILARITY_THRESHOLD:
                            sem_list.append({
                                'ID A': recs[i][id_col],
                                'ID B': recs[j][id_col],
                                'Desc A': recs[i]['Standard_Desc'],
                                'Desc B': recs[j]['Standard_Desc'],
                                'Semantic Match %': f"{sim:.1%}"
                            })
                if sem_list:
                    st.dataframe(pd.DataFrame(sem_list), use_container_width=True, height=400)
                else:
                    st.success("No semantic duplicates found in this filtered view.")

# --- PAGE: METHODOLOGY ---
with page[3]:
    st.markdown("#### 🧠 Technical Methodology & AI Stack")
    st.markdown("Understand the advanced algorithms powering this inventory intelligence system.")
    
    # Add Connection Diagnostics Section
    with st.expander("🔍 Gemini Connection Diagnostics", expanded=False):
        st.markdown("### Connection Status")
        
        # Display status overview
        col1, col2, col3 = st.columns(3)
        with col1:
            if gemini_status["enabled"]:
                st.metric("Status", "✅ Connected", delta=None)
            else:
                st.metric("Status", "❌ Disconnected", delta=None)
        
        with col2:
            st.metric("Classification", "✅ OK" if gemini_status["classification"] else "❌ Failed")
        
        with col3:
            st.metric("Embeddings", "✅ OK" if gemini_status["embeddings"] else "❌ Failed")
        
        st.markdown("---")
        
        # Detailed diagnostics
        st.markdown("### Diagnostic Details")
        
        # Token status
        api_key = get_gemini_api_key()
        st.markdown(f"**API Key Status:** {'✅ Present' if api_key else '❌ Missing'}")
        if api_key:
            is_valid = validate_gemini_api_key(api_key)
            st.markdown(f"**API Key Validation:** {'✅ Valid format' if is_valid else '❌ Invalid format'}")
        
        # Network connectivity
        st.markdown("**Network Tests:**")
        dns_ok = check_dns_resolution(GEMINI_API_HOSTNAME)
        st.markdown(f"- DNS Resolution: {'✅ OK' if dns_ok else '❌ Failed'}")
        
        is_accessible, conn_error = check_gemini_api_connectivity()
        st.markdown(f"- TCP Connectivity: {'✅ OK' if is_accessible else '❌ Failed'}")
        if conn_error:
            st.markdown(f"  - Error: `{conn_error}`")
        
        # Overall reason
        if gemini_status.get("reason"):
            st.markdown(f"**Overall Status:** `{gemini_status['reason']}`")
        
        if gemini_status.get("error_detail"):
            st.markdown(f"**Error Details:** {gemini_status['error_detail']}")
        
        # Configuration info
        st.markdown("---")
        st.markdown("### Configuration")
        st.markdown(f"**ENABLE_GEMINI_MODELS:** `{ENABLE_GEMINI_MODELS}`")
        st.markdown(f"**GEMINI_API_BASE_URL:** `{GEMINI_API_BASE_URL}`")
        st.markdown(f"**GEMINI_API_TIMEOUT:** `{GEMINI_API_TIMEOUT}s`")
        st.markdown(f"**Models:**")
        st.markdown(f"- Classification: `{GEMINI_CLASSIFICATION_MODEL}`")
        st.markdown(f"- Embeddings: `{GEMINI_EMBEDDING_MODEL}`")
    
    st.markdown("""
    ### 1. Data Processing (ETL)
    We standardize the raw 543 rows by stripping quote artifacts, uppercasing, and cleaning symbols. We utilize **RegEx** to extract technical specifications (Numbers, Sizes, Genders) into a "Technical DNA" profile for every part.
    
    ### 2. Intelligent Categorization
    Instead of standard K-Means (which is biased by word frequency), we use a **Prioritized Knowledge Base** to anchor nouns to super-categories. We also run a cached Gemini **classification model** (gemini-1.5-flash) to assign *Gemini_Product_Group* labels with confidence scores.
    
    ### 3. Cluster Validation
    We validate the knowledge-anchored categories against **K-Means** clusters to ensure semantic consistency before scoring confidence. Gemini embeddings (text-embedding-004) power additional clustering confidence on semantic embeddings.
    
    ### 4. Anomaly Detection
    We use the **Isolation Forest** algorithm on both TF-IDF features and Gemini embeddings to flag unusual items with *Gemini_Anomaly_Flag*.
    
    ### 5. Fuzzy Match & Conflict Resolution
    We use the **Levenshtein Distance** algorithm. However, we've added a **Business Logic Layer**: if two items have similar text but conflicting 'Technical DNA' (e.g. one is Male, one is Female), the system overrides the AI and flags it as a **Variant**, not a duplicate. We also run a semantic duplicate check using cosine similarity on Gemini embeddings (text-embedding-004) within a small window.
    """)

# --- PAGE: MY APPROACH ---
with page[4]:
    st.markdown("#### 🧭 My Approach")
    st.markdown("A concise walkthrough of the full end-to-end workflow implemented across the app.")
    st.markdown("""
    <h2>🏛️ Architectural Philosophy: The Hybrid Intelligence Model</h2>
    <p>The core strength of this application lies in its <b>Hybrid AI Approach</b>. Rather than relying on a single algorithm, it combines three distinct layers of logic to ensure accuracy:</p>
    <ol>
        <li><p><b>Heuristic Layer:</b> Uses a predefined Knowledge Base and Regular Expressions (RegEx) for absolute technical accuracy.</p></li>
        <li><p><b>Statistical Layer (Classical ML):</b> Employs <b>TF-IDF</b>, <b>K-Means</b>, and <b>Isolation Forest</b> for pattern recognition and anomaly detection based on the specific dataset.</p></li>
        <li><p><b>Neural Layer (Deep Learning):</b> Leverages the <b>Gemini API</b> (Gemini 1.5 Flash and text-embedding-004) for semantic understanding and classification.</p></li>
    </ol>
    <hr>
    <h2>🛠️ Phase 1: Data Standardization &amp; "Tech DNA" Extraction</h2>
    <p>The system first cleanses the data to remove "noise" (special characters, case inconsistencies) that typically disrupts auditing.</p>
    <ul>
        <li><p><b>Standardization:</b> The <code>clean_description</code> function normalizes descriptions (e.g., converting "O-RING" to "O RING" and "MECH-SEAL" to "MECHANICAL SEAL").</p></li>
        <li><p><b>Feature Engineering (Tech DNA):</b> The <code>get_tech_dna</code> function is a specialized parser. It extracts "Genetic Markers" of an inventory item—specifically <b>numeric values</b> and <b>technical attributes</b> (Gender, Connection type, Pressure rating). This allows the AI to distinguish between a "Male Valve" and a "Female Valve" even if the text descriptions are 99% similar.</p></li>
    </ul>
    <hr>
    <h2>🏷️ Phase 2: Multi-Stage Categorization</h2>
    <p>To ensure items are placed in the correct <code>Product_Group</code>, the app runs a parallel classification process:</p>
    <h3>1. Rule-Based Noun Extraction</h3>
    <p>The <code>intelligent_noun_extractor</code> uses a prioritized list of phrases (e.g., "BALL VALVE" takes precedence over "VALVE") to identify the "Part Noun."</p>
    <h3>2. Gemini Classification (Deep Learning)</h3>
    <p>If enabled, the system calls the <code>gemini-1.5-flash</code> model. Unlike traditional models, this does not require training on your specific data; it uses its pre-trained "knowledge" of the English language to categorize items into labels like "Fasteners &amp; Seals" or "Piping &amp; Fittings."</p>
    <h3>3. Cluster Validation</h3>
    <p>The system uses <b>K-Means Clustering</b> to group items that are mathematically similar. It then checks if the "Human Logic" category matches the "Machine Logic" cluster. If they match, the <b>Confidence Score</b> increases.</p>
    <hr>
    <h2>🚨 Phase 3: The Quality &amp; Audit Hub</h2>
    <p>This is the engine's "Defense Layer," designed to catch errors that a human auditor might miss.</p>
    <h3>Anomaly Detection (Isolation Forest)</h3>
    <p>The <code>IsolationForest</code> algorithm treats the inventory list as a multi-dimensional map. Items that exist in "lonely" areas of this map (mathematical outliers) are flagged as anomalies. This is excellent for catching typos or items that simply don't belong in the catalog.</p>
    <h3>Fuzzy vs. Semantic Duplicates</h3>
    <ul>
        <li><p><b>Fuzzy Matching:</b> Uses Levenshtein distance to find text-based similarities.</p></li>
        <li><p><b>Semantic Matching:</b> Uses <b>Cosine Similarity</b> on high-dimensional vectors (Embeddings).</p></li>
        <li><p><b>The "Spec-Trap" Override:</b> Crucially, if two items have a high similarity score but different "Tech DNA" (e.g., one is 150# rating and the other is 300#), the system overrides the duplicate flag and labels it a <b>Variant</b>.</p></li>
    </ul>
    <hr>
    <h2>📈 Phase 4: Executive Insights (Streamlit UI)</h2>
    <p>The final layer translates complex data into actionable metrics using <b>Plotly</b>:</p>
    <ul>
        <li><p><b>Inventory Health Gauge:</b> A real-time calculation of data accuracy.</p></li>
        <li><p><b>Confidence Distribution:</b> A histogram showing the reliability of the AI's categorization.</p></li>
        <li><p><b>Duplicate Pairs:</b> A structured list of potential risks for procurement and warehouse teams.</p></li>
    </ul>
    <hr>
    <h2>🧰 Technical Stack Summary</h2>

    | Component | Technology |
    | - | - |
    | Frontend | Streamlit |
    | Data Processing | Pandas, NumPy, RegEx |
    | Machine Learning | Scikit-Learn (KMeans, Isolation Forest) |
    | Deep Learning | Gemini API (Gemini 1.5 Flash, text-embedding-004) |
    | Visualizations | Plotly Express &amp; Graph Objects |
    """, unsafe_allow_html=True)
