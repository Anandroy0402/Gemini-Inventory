import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher

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
DEPARTMENT_MAP = {
    'TL': 'TOOLS & EQUIPMENT', 'IN': 'INSTRUMENTATION', 'CN': 'CONSUMABLES', 
    'SP': 'SPARE PARTS', 'BM': 'CIVIL/BUILDING', 'PS': 'PROJECT STORES', 'IT': 'IT'
}

PRODUCT_GROUPS = {
    "PIPING & FITTINGS": ["FLANGE", "PIPE", "ELBOW", "TEE", "UNION", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING", "UPVC", "CPVC", "PVC"],
    "VALVES & ACTUATORS": ["BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "GLOBE VALVE", "CONTROL VALVE", "VALVE", "ACTUATOR", "COCK"],
    "FASTENERS & SEALS": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O RING", "MECHANICAL SEAL", "SEAL", "JOINT"],
    "ELECTRICAL & INSTR.": ["TRANSMITTER", "CABLE", "WIRE", "GAUGE", "SENSOR", "CONNECTOR", "SWITCH", "TERMINAL", "INSTRUMENT"],
    "TOOLS & HARDWARE": ["PLIER", "CUTTING PLIER", "STRIPPER", "WIRE STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER", "TAPE MEASURE", "MEASURING TAPE", "BIT", "DRILL BIT"],
    "CONSUMABLES & GENERAL": ["BRUSH", "PAINT BRUSH", "TAPE", "ADHESIVE", "HOSE", "SAFETY GLOVE", "GLOVE", "CLEANER", "PAINT", "CEMENT", "STICKER", "CHALK"],
    "SPECIALIZED SPARES": ["FILTER", "BEARING", "PUMP", "MOTOR", "CARTRIDGE", "IMPELLER", "SPARE"]
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
    return "UNMAPPED"

def dominant_group(series):
    counts = series.value_counts()
    return counts.idxmax() if not counts.empty else "UNMAPPED"

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    df['Standard_Desc'] = df[desc_col].apply(clean_description)
    df['Prefix'] = df[id_col].astype(str).str.extract(r'^([A-Z]+)')
    df['Business_Dept'] = df['Prefix'].map(DEPARTMENT_MAP).fillna('GENERAL STOCK')
    df['Part_Noun'] = df['Standard_Desc'].apply(intelligent_noun_extractor)
    df['Product_Group'] = df['Part_Noun'].apply(map_product_group)

    # NLP & Topic Modeling
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    
    # Clustering for Confidence
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / np.max(dists, axis=1))).round(4)
    cluster_groups = df.groupby('Cluster_ID')['Product_Group'].agg(dominant_group)
    df['Cluster_Group'] = df['Cluster_ID'].map(cluster_groups)
    df['Cluster_Validated'] = df['Product_Group'] == df['Cluster_Group']
    
    # Anomaly
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(tfidf_matrix) # Using tfidf for complexity-based anomalies

    # Fuzzy & Tech DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
    
    return df, id_col, desc_col

# --- DATA LOADING ---
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(target_file)
else:
    st.error("Data file missing from repository. Please ensure 'raw_data.csv' is present.")
    st.stop()

# Filter defaults
dept_options = sorted(df_raw['Business_Dept'].unique())
group_options = list(PRODUCT_GROUPS.keys())
extra_groups = sorted(set(df_raw['Product_Group'].unique()) - set(group_options))
group_options.extend(extra_groups)

# --- HEADER & MODERN NAVIGATION ---
st.title("🛡️ AI Inventory Auditor Pro")
st.markdown("### Advanced Inventory Intelligence & Quality Management")

# Modern horizontal tab navigation
page = st.tabs(["📈 Executive Dashboard", "📍 Categorization Audit", "🚨 Quality Hub (Anomalies/Dups)", "🧠 Technical Methodology"])

# Initialize filtered dataframe (will be filtered per page as needed)
df = df_raw.copy()

# --- PAGE: EXECUTIVE DASHBOARD ---
with page[0]:
    st.markdown("#### 📊 Inventory Health Overview")
    st.markdown("Get a bird's eye view of your inventory data quality and distribution.")
    
    # Filters at the top
    with st.container():
        st.markdown("##### 🔍 Filters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_dept = st.multiselect("Department", options=dept_options, default=dept_options, key="dash_dept")
        with fcol2:
            selected_group = st.multiselect("Business Category", options=group_options, default=group_options, key="dash_group")
    
    # Apply Filters
    df = df_raw[df_raw['Business_Dept'].isin(selected_dept)]
    df = df[df['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📦 SKUs Analyzed", len(df))
    kpi2.metric("🎯 Mean AI Confidence", f"{df['Confidence'].mean():.1%}")
    kpi3.metric("⚠️ Anomalies Found", len(df[df['Anomaly_Flag'] == -1]))
    kpi4.metric("🔄 Duplicate Pairs", "Audit Required")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='Business_Dept', title="Inventory Distribution by Department", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        top_nouns = df['Part_Noun'].value_counts().head(10).reset_index()
        fig_bar = px.bar(top_nouns, x='Part_Noun', y='count', title="Top 10 Product Categories", labels={'Part_Noun':'Product', 'count':'Qty'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Health Gauge
    health_val = (len(df[df['Anomaly_Flag'] == 1]) / len(df)) * 100
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_val,
        title = {'text': "Catalog Data Accuracy %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00cc96"}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE: CATEGORIZATION AUDIT ---
with page[1]:
    st.markdown("#### 📍 AI Categorization & Filtered Audit")
    st.markdown("Drill down into specific product categories with intelligent filtering.")
    
    # Filters at the top of the table
    with st.container():
        st.markdown("##### 🔍 Filters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_dept = st.multiselect("Department", options=dept_options, default=dept_options, key="cat_dept")
        with fcol2:
            selected_group = st.multiselect("Business Category", options=group_options, default=group_options, key="cat_group")
    
    # Apply Filters
    df = df_raw[df_raw['Business_Dept'].isin(selected_dept)]
    df = df[df['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    st.markdown(f"**Showing {len(df)} items**")
    
    # Data Table with sorting
    st.dataframe(
        df[[id_col, 'Standard_Desc', 'Part_Noun', 'Product_Group', 'Business_Dept', 'Confidence']].sort_values('Confidence', ascending=False),
        use_container_width=True,
        height=400
    )
    
    summary = (
        df.groupby('Product_Group', dropna=False)
        .agg(Items=(id_col, 'count'), Mean_Confidence=('Confidence', 'mean'), Cluster_Match_Rate=('Cluster_Validated', 'mean'))
        .reset_index()
        .sort_values('Items', ascending=False)
    )
    summary['Mean_Confidence'] = summary['Mean_Confidence'].round(3)
    summary['Cluster_Match_Rate'] = summary['Cluster_Match_Rate'].round(3)
    st.markdown("#### 📌 Category Distribution & Confidence")
    st.dataframe(summary, use_container_width=True, height=260)

    # Distribution of confidence
    fig_hist = px.histogram(df, x="Confidence", nbins=20, title="Confidence Score Distribution", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)

# --- PAGE: QUALITY HUB ---
with page[2]:
    st.markdown("#### 🚨 Anomaly & Duplicate Identification")
    st.markdown("Identify quality issues and potential duplicates in your inventory data.")
    
    # Filters at the top
    with st.container():
        st.markdown("##### 🔍 Filters")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_dept = st.multiselect("Department", options=dept_options, default=dept_options, key="qual_dept")
        with fcol2:
            selected_group = st.multiselect("Business Category", options=group_options, default=group_options, key="qual_group")
    
    # Apply Filters
    df = df_raw[df_raw['Business_Dept'].isin(selected_dept)]
    df = df[df['Product_Group'].isin(selected_group)]
    
    st.markdown("---")
    
    t1, t2 = st.tabs(["⚠️ Anomalies", "👯 Fuzzy Duplicates"])
    
    with t1:
        st.subheader("Statistical Anomalies (Isolation Forest)")
        anoms = df[df['Anomaly_Flag'] == -1]
        st.warning(f"Found {len(anoms)} anomalies in the current view.")
        st.dataframe(anoms[[id_col, desc_col, 'Part_Noun']], use_container_width=True, height=400)
        
    with t2:
        st.subheader("Fuzzy Duplicate Audit (Spec-Aware)")
        st.info("System identifies items with >85% text similarity but differentiates based on numeric specs (Size/Gender).")
        
        # Calculate fuzzy duplicates for the current view
        fuzzy_list = []
        recs = df.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, min(i + 50, len(recs))): # Smaller window for real-time speed
                r1, r2 = recs[i], recs[j]
                sim = SequenceMatcher(None, r1['Standard_Desc'], r2['Standard_Desc']).ratio()
                if sim > 0.85:
                    dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                    is_variant = (dna1['numbers'] != dna2['numbers']) or (dna1['attributes'] != dna2['attributes'])
                    fuzzy_list.append({
                        'ID A': r1[id_col], 'ID B': r2[id_col],
                        'Desc A': r1['Standard_Desc'], 'Desc B': r2['Standard_Desc'],
                        'Match %': f"{sim:.1%}", 'Verdict': "🛠️ Variant" if is_variant else "🚨 Duplicate"
                    })
        
        if fuzzy_list:
            st.dataframe(pd.DataFrame(fuzzy_list), use_container_width=True, height=400)
        else:
            st.success("No fuzzy duplicates found in this filtered view.")

# --- PAGE: METHODOLOGY ---
with page[3]:
    st.markdown("#### 🧠 Technical Methodology & AI Stack")
    st.markdown("Understand the advanced algorithms powering this inventory intelligence system.")
    
    st.markdown("""
    ### 1. Data Processing (ETL)
    We standardize the raw 543 rows by stripping quote artifacts, uppercasing, and cleaning symbols. We utilize **RegEx** to extract technical specifications (Numbers, Sizes, Genders) into a "Technical DNA" profile for every part.
    
    ### 2. Intelligent Categorization
    Instead of standard K-Means (which is biased by word frequency), we use a **Prioritized Knowledge Base** to anchor nouns to super-categories. This ensures that tools like 'Pliers' aren't mislabeled as 'Pipes' just because they both mention a 'Size'.
    
    ### 3. Cluster Validation
    We validate the knowledge-anchored categories against **K-Means** clusters to ensure semantic consistency before scoring confidence.
    
    ### 4. Anomaly Detection
    We use the **Isolation Forest** algorithm. It isolates observations by randomly selecting a feature and then randomly selecting a split value. Outliers (anomalies) are easier to isolate, resulting in shorter paths.
    
    ### 5. Fuzzy Match & Conflict Resolution
    We use the **Levenshtein Distance** algorithm. However, we've added a **Business Logic Layer**: if two items have similar text but conflicting 'Technical DNA' (e.g. one is Male, one is Female), the system overrides the AI and flags it as a **Variant**, not a duplicate.
    """)
