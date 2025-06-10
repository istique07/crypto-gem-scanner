
import streamlit as st
import pandas as pd
from scanner import run_multi_coin_ml_scan

st.set_page_config(page_title="AI Crypto GEM Scanner", layout="wide")
st.title("🚀 AI-Based Crypto Buy/Hold Scanner")

num_coins = st.slider("Select number of top coins", 1, 20, 5)
if st.button("🔍 Run Scan"):
    df = run_multi_coin_ml_scan(limit=num_coins)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "scan_report.csv", "text/csv")
