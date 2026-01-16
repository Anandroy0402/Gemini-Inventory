import streamlit as st
import pandas as pd
import plotly.express as px

# Page Config
st.set_page_config(page_title="AI Inventory Analytics", layout="wide")

# Title & Intro
st.title("📊 AI-Driven Inventory Intelligence")
st.markdown("""
**Objective:** Analyze raw inventory data to detect anomalies, duplicates, and categorize products using Unsupervised Learning.
""")

# Load Data
@st.cache_data
def load_data():
    # In a real app, these would come from your cloud database or S3
    df = pd.read_csv('processed_data.csv')
    dups = pd.read_csv('duplicates.csv')
    return df, dups

try:
    df, dups = load_data()
except:
    st.error("Data not found! Please run the processing script first or upload 'processed_data.csv'.")
    st.stop()

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total SKU Count", len(df))
col2.metric("AI Categories Detected", df['AI_Category'].nunique())
col3.metric("Anomalies Detected", len(df[df['is_anomaly']=='Yes']))
col4.metric("Potential Duplicates", len(dups))

# --- TABBED VIEW ---
tab1, tab2, tab3 = st.tabs(["📈 Business Insights", "🚨 Anomaly Detection", "👯 Duplicate Management"])

with tab1:
    st.subheader("Automated Product Categorization")
    st.markdown("Categories were generated automatically using **TF-IDF Vectorization** and **K-Means Clustering**.")
    
    # Bar Chart
    cat_counts = df['AI_Category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    fig = px.bar(cat_counts, x='Count', y='Category', orientation='h', title="Inventory Distribution by AI Category", color='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Detailed Data"):
        st.dataframe(df[['Item No.', 'Description', 'AI_Category', 'Confidence_Score']])

with tab2:
    st.subheader("Data Anomaly Report")
    st.markdown("Items flagged by **Isolation Forest** algorithm due to unusual description patterns (e.g., length, character complexity).")
    
    anomalies = df[df['is_anomaly']=='Yes']
    st.dataframe(anomalies[['Item No.', 'Description', 'UoM', 'AI_Category']], use_container_width=True)
    
    # Scatter Plot
    fig2 = px.scatter(df, x='desc_len', y='digit_count', color='is_anomaly', 
                      title="Anomaly Detection: Description Length vs Complexity",
                      hover_data=['Description'])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Fuzzy Duplicate Identification")
    st.markdown("Potential duplicates identified using **Sequence Matching (Levenshtein Distance)** with >85% similarity.")
    st.dataframe(dups, use_container_width=True)