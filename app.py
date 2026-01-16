import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import socket
from difflib import SequenceMatcher
from urllib import request, error

# Advanced AI/ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor Pro", layout="wide", page_icon="🛡️")

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
DEFAULT_PRODUCT_GROUP = "Consumables & General"
MIN_DISTANCE_THRESHOLD = 1e-8  # Replace zero distances to avoid divide-by-zero in confidence calculations.
COMPARISON_WINDOW_SIZE = 50  # Windowed comparisons keep duplicate checks lightweight.
FUZZY_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_SIMILARITY_THRESHOLD = 0.9
HF_BATCH_SIZE = 16
HF_ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
HF_INFERENCE_TIMEOUT = 30
ENABLE_HF_MODELS = os.getenv("ENABLE_HF_MODELS", "false").lower() == "true"
HF_CONFIDENCE_MIN_THRESHOLD = 0.8
HF_CONFIDENCE_MIN_TARGET = 0.6
HF_CONFIDENCE_MAX_TARGET = 0.98

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
    if max_score >= HF_CONFIDENCE_MIN_THRESHOLD:
        return scores
    min_score = scores.min()
    if max_score == min_score:
        return pd.Series(np.full(len(scores), max(max_score, HF_CONFIDENCE_MIN_TARGET)), index=scores.index)
    scaled = (scores - min_score) / (max_score - min_score)
    return (scaled * (HF_CONFIDENCE_MAX_TARGET - HF_CONFIDENCE_MIN_TARGET) + HF_CONFIDENCE_MIN_TARGET).round(4)

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

def get_hf_secret(key):
    try:
        return st.secrets[key]
    except (AttributeError, KeyError):
        return None

def get_hf_token():
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or get_hf_secret("HF_TOKEN")
        or get_hf_secret("HUGGINGFACEHUB_API_TOKEN")
    )

def call_hf_inference(model, payload, token, warning_message):
    if not token:
        return None
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{HF_INFERENCE_API_URL}/{model}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    try:
        with request.urlopen(req, timeout=HF_INFERENCE_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
        if isinstance(result, dict) and result.get("error"):
            st.warning(f"{warning_message} ({result.get('error')})")
            return None
        return result
    except (error.HTTPError, error.URLError, socket.timeout, ValueError) as exc:
        if isinstance(exc, socket.timeout):
            detail = "timeout"
        elif isinstance(exc, error.HTTPError):
            detail = f"HTTP {exc.code}"
        elif isinstance(exc, error.URLError):
            detail = "network error"
        else:
            detail = "invalid response"
        st.warning(f"{warning_message} ({detail})")
        return None

def run_hf_zero_shot(texts, labels):
    token = get_hf_token()
    if not token:
        st.warning("Hugging Face token missing; skipping hosted classification.")
        return None
    if isinstance(texts, str):
        texts = [texts]
    try:
        results = []
        for start in range(0, len(texts), HF_BATCH_SIZE):
            batch = texts[start:start + HF_BATCH_SIZE]
            payload = {
                "inputs": batch,
                "parameters": {
                    "candidate_labels": labels,
                    "hypothesis_template": "This industrial inventory item is {}"
                },
                "options": {"wait_for_model": True}
            }
            batch_results = call_hf_inference(
                HF_ZERO_SHOT_MODEL,
                payload,
                token,
                "Hugging Face classification failed; using existing categories."
            )
            if not batch_results:
                return None
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            results.extend(batch_results)
        return results
    except (RuntimeError, ValueError):
        st.warning("Hugging Face classification failed; using existing categories.")
        return None

def compute_embeddings(texts):
    token = get_hf_token()
    if not token:
        st.warning("Hugging Face token missing; skipping hosted embeddings.")
        return None
    try:
        embeddings = []
        for start in range(0, len(texts), HF_BATCH_SIZE):
            batch = texts[start:start + HF_BATCH_SIZE]
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            batch_embeddings = call_hf_inference(
                HF_EMBEDDING_MODEL,
                payload,
                token,
                "Embedding generation failed; falling back to TF-IDF signals."
            )
            if not batch_embeddings:
                return None
            if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0 and isinstance(batch_embeddings[0], (int, float)):
                # Hugging Face returns a flat list for single inputs; wrap for consistent batching.
                batch_embeddings = [batch_embeddings]
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    except (RuntimeError, ValueError):
        st.warning("Embedding generation failed; falling back to TF-IDF signals.")
        return None

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path, enable_hf_models=False):
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

    standard_desc = df['Standard_Desc'].tolist() if enable_hf_models else None
    hf_inputs = (
        df['Part_Noun']
        .fillna('')
        .str.cat(df['Standard_Desc'].fillna(''), sep=' ')
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .tolist()
        if enable_hf_models else None
    )

    # Hugging Face Zero-Shot Classification
    hf_results = run_hf_zero_shot(hf_inputs, list(PRODUCT_GROUPS.keys())) if enable_hf_models else None
    if hf_results:
        df['HF_Product_Group'] = [res['labels'][0] for res in hf_results]
        df['HF_Product_Confidence'] = [round(res['scores'][0], 4) for res in hf_results]
    else:
        df['HF_Product_Group'] = df['Product_Group']
        df['HF_Product_Confidence'] = df['Confidence']
    df['HF_Product_Confidence'] = normalize_confidence_scores(df['HF_Product_Confidence'])

    # Hugging Face Embeddings for Clustering/Anomaly
    embeddings = compute_embeddings(standard_desc) if enable_hf_models else None
    if embeddings is not None:
        kmeans_hf = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['HF_Cluster_ID'] = kmeans_hf.fit_predict(embeddings)
        hf_dists = kmeans_hf.transform(embeddings)
        max_dist = apply_distance_floor(hf_dists)
        df['HF_Cluster_Confidence'] = (1 - (np.min(hf_dists, axis=1) / max_dist)).round(4)
        iso_hf = IsolationForest(contamination=0.04, random_state=42)
        df['HF_Anomaly_Flag'] = iso_hf.fit_predict(embeddings)
        df['HF_Embedding'] = list(embeddings)
    else:
        df['HF_Cluster_ID'] = df['Cluster_ID']
        df['HF_Cluster_Confidence'] = df['Confidence']
        df['HF_Anomaly_Flag'] = df['Anomaly_Flag']
        df['HF_Embedding'] = [None] * len(df)

    # Fuzzy & Tech DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)

    return df, id_col, desc_col

# --- DATA LOADING ---
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(target_file, enable_hf_models=ENABLE_HF_MODELS)
else:
    st.error("Data file missing from repository. Please ensure 'raw_data.csv' is present.")
    st.stop()

# Filter defaults
group_options = list(PRODUCT_GROUPS.keys())

# --- HEADER & MODERN NAVIGATION ---
st.title("🛡️ AI Inventory Auditor Pro")
st.markdown("### Advanced Inventory Intelligence & Quality Management")

# Modern horizontal tab navigation
page = st.tabs(["📈 Executive Dashboard", "📍 Categorization Audit", "🚨 Quality Hub (Anomalies/Dups)", "🧠 Technical Methodology"])

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
    kpi2.metric("🎯 Mean HF Confidence", f"{df['HF_Product_Confidence'].mean():.1%}")
    kpi3.metric("⚠️ HF Anomalies Found", len(df[df['HF_Anomaly_Flag'] == -1]))
    kpi4.metric("🔄 Duplicate Pairs", len(fuzzy_list))

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='HF_Product_Group', title="Inventory Distribution by HF Product Category", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        top_nouns = df['Part_Noun'].value_counts().head(10).reset_index()
        fig_bar = px.bar(top_nouns, x='Part_Noun', y='count', title="Top 10 Product Categories", labels={'Part_Noun':'Product', 'count':'Qty'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Health Gauge
    health_val = (len(df[df['HF_Anomaly_Flag'] == 1]) / len(df)) * 100
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_val,
        title = {'text': "Catalog Data Accuracy %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### 💼 Business Insights")
    insights = (
        df.groupby('HF_Product_Group', dropna=False)
        .agg(
            Items=(id_col, 'count'),
            Mean_HF_Confidence=('HF_Product_Confidence', 'mean'),
            HF_Anomaly_Rate=('HF_Anomaly_Flag', lambda x: (x == -1).mean())
        )
        .reset_index()
        .sort_values('Items', ascending=False)
    )
    insights['Mean_HF_Confidence'] = insights['Mean_HF_Confidence'].round(3)
    insights['HF_Anomaly_Rate'] = insights['HF_Anomaly_Rate'].round(3)
    st.dataframe(insights, use_container_width=True, height=260)
    fig_insights = px.bar(
        insights.head(10),
        x='HF_Product_Group',
        y='Mean_HF_Confidence',
        title="Top HF Categories by Confidence",
        labels={'HF_Product_Group': 'HF Category', 'Mean_HF_Confidence': 'Mean Confidence'}
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
                'HF_Product_Group',
                'HF_Product_Confidence',
                'Confidence'
            ]
        ].sort_values('HF_Product_Confidence', ascending=False),
        use_container_width=True,
        height=400
    )
    
    summary = (
        df.groupby('Product_Group', dropna=False)
        .agg(
            Items=(id_col, 'count'),
            Mean_Confidence=('Confidence', 'mean'),
            Mean_HF_Confidence=('HF_Product_Confidence', 'mean'),
            Cluster_Match_Rate=('Cluster_Validated', 'mean')
        )
        .reset_index()
        .sort_values('Items', ascending=False)
    )
    summary['Mean_Confidence'] = summary['Mean_Confidence'].round(3)
    summary['Mean_HF_Confidence'] = summary['Mean_HF_Confidence'].round(3)
    summary['Cluster_Match_Rate'] = summary['Cluster_Match_Rate'].round(3)
    st.markdown("#### 📌 Category Distribution & Confidence")
    st.dataframe(summary, use_container_width=True, height=260)

    # Distribution of confidence
    fig_hist = px.histogram(df, x="HF_Product_Confidence", nbins=20, title="HF Confidence Score Distribution", color_discrete_sequence=['#636EFA'])
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
    
    t1, t2, t3 = st.tabs(["⚠️ HF Anomalies", "👯 Fuzzy Duplicates", "🧠 Semantic Duplicates"])
    
    with t1:
        st.subheader("HF Embedding Anomalies (Isolation Forest)")
        anoms = df[df['HF_Anomaly_Flag'] == -1]
        st.warning(f"Found {len(anoms)} anomalies in the current view.")
        st.dataframe(
            anoms[[id_col, desc_col, 'Part_Noun', 'HF_Product_Group', 'HF_Cluster_Confidence']],
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
        st.subheader("Semantic Duplicate Audit (Sentence-Transformers)")
        if df['HF_Embedding'].apply(lambda x: x is None).all():
            st.info("Semantic duplicate detection unavailable (HF embeddings not loaded).")
        else:
            records = df.reset_index(drop=True)
            if records['HF_Embedding'].apply(lambda x: x is None).any():
                st.info("Semantic duplicate detection unavailable (HF embeddings incomplete).")
            else:
                sem_list = []
                recs = records.to_dict('records')
                embeddings = records['HF_Embedding'].tolist()
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
    
    st.markdown("""
    ### 1. Data Processing (ETL)
    We standardize the raw 543 rows by stripping quote artifacts, uppercasing, and cleaning symbols. We utilize **RegEx** to extract technical specifications (Numbers, Sizes, Genders) into a "Technical DNA" profile for every part.
    
    ### 2. Intelligent Categorization
    Instead of standard K-Means (which is biased by word frequency), we use a **Prioritized Knowledge Base** to anchor nouns to super-categories. We also run a cached Hugging Face **zero-shot classifier** (facebook/bart-large-mnli) to assign *HF_Product_Group* labels with confidence scores.
    
    ### 3. Cluster Validation
    We validate the knowledge-anchored categories against **K-Means** clusters to ensure semantic consistency before scoring confidence. A sentence-transformer model (all-MiniLM-L6-v2) powers additional HF clustering confidence on semantic embeddings.
    
    ### 4. Anomaly Detection
    We use the **Isolation Forest** algorithm on both TF-IDF features and Hugging Face embeddings to flag unusual items with *HF_Anomaly_Flag*.
    
    ### 5. Fuzzy Match & Conflict Resolution
    We use the **Levenshtein Distance** algorithm. However, we've added a **Business Logic Layer**: if two items have similar text but conflicting 'Technical DNA' (e.g. one is Male, one is Female), the system overrides the AI and flags it as a **Variant**, not a duplicate. We also run a semantic duplicate check using cosine similarity on sentence-transformer embeddings within a small window.
    """)
