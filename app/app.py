"""
app.py  — Streamlit App
-----------------------------
Functionality:
  1. Lets the user upload a CSV (customer_360 format)
  2. Scores all customers using the saved pipelines
  3. Shows a scorecard table and key metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Make src/ importable from app folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))  

from train import (LeakageDropper, TypedImputer, OutlierClipper,EncoderTransformer, ColumnScaler,
                    VarianceFilter,CorrDropper, UnionFeatureSelector)

from predict import score_all_customers
  
# -------------------------------Page config ---------------------------------------------- 
st.set_page_config(page_title="Customer Analytics Dashboard",page_icon="📊",layout="wide")

col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/16139/16139721.png", width=120)
with col2:
    st.title("Ecommerce Customer Analytics Dashboard")
    st.caption("Predict High Spenders, Churners and Prioritize Customers")

st.markdown(""" This app scores customers using trained XGBoost models to predict:
- **High Spenders** — top 10% of revenue contributors
- **Churners** — customers who bought in period 1 but stopped in period 2 """)

# -------------------------------Slidebar---------------------------------------------- 
st.sidebar.header("Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Upload customer_360.csv (or any file with Customer360 columns)",
    type=["csv"])

MODEL_DIR_SPD = os.path.join(os.path.dirname(__file__), '..', 'models', 'high_spender')
MODEL_DIR_CHN = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn')

# -------------------------------Main Logic---------------------------------------------- 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(f"Loaded {len(df):,} customers")

    with st.spinner("Running predictions..."):
        try:
            scorecard = score_all_customers( df, MODEL_DIR_SPD, MODEL_DIR_CHN)

        except FileNotFoundError:
            st.error("⚠️ Model files not found. Please run `python -m streamlit run app/app.py` first.")
            st.stop()

# -------------------------------Summary KPIs---------------------------------------------- 

    priority_cutoff = scorecard['priority_score'].quantile(0.90)   # Top 10% priority customers
    high_priority_count = (scorecard['priority_score'] >= priority_cutoff).sum()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers",   f"{len(scorecard):,}")
    col2.metric("High Spenders",     f"{scorecard['high_spender_predicted'].sum():,}")
    col3.metric("At-Risk Churners",  f"{scorecard['churn_predicted'].sum():,}")
    col4.metric("Urgent (Both)",     f"{((scorecard['high_spender_predicted']==1) & (scorecard['churn_predicted']==1)).sum():,}")
    col5.metric('Top 10% Priority',f"{high_priority_count:,}")

    # -------------------------------Action Priority breakdown---------------------------------------------- 
    st.subheader("Action Priority Distribution")
    priority_counts = scorecard['action_priority'].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    fig = px.bar(priority_counts,x="Priority",y="Count",color="Priority",text="Count",template="plotly_dark",
                 color_discrete_map={ '🔴 URGENT - Retain High Spender': '#EF4444',
                                      '🟡 Win-back Campaign': '#F59E0B',
                                      '🟢 VIP Programme': '#10B981',
                                      '⚪ Standard': '#6B7280' })

    fig.update_traces(textposition='outside')
    fig.update_layout( showlegend=False, xaxis_title="Priority Level",yaxis_title="Customer Count")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------Scorecard table---------------------------------------------- 
    st.subheader("Customer Scorecard")
    display_cols = ['user_id', 'high_spender_probability', 'spend_tier',
                    'churn_probability', 'churn_risk','priority_score','action_priority']
    st.dataframe(scorecard[display_cols],use_container_width=True)
      
    # -------------------------------Top 10 Priority Customers---------------------------------------------- 
    st.subheader("Top 10 Priority Customers")
    top_customers = (scorecard[['user_id','priority_score','high_spender_probability','churn_probability',
        'action_priority']].sort_values('priority_score', ascending=False).head(10))
    st.dataframe(top_customers, use_container_width=True)

    # -------------------------------Download button---------------------------------------------- 
    csv = scorecard.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="⬇️ Download Scorecard CSV",data=csv,file_name="customer_scorecard.csv",
        mime="text/csv")

else:
    st.info("Upload a Customer360 CSV file in the sidebar to get started.")

    # ── Show example of what the output looks like ────────────────────────────
    st.subheader("Example Output Preview")
    example = pd.DataFrame({
        'user_id':                  [10001,10002,10003,10004],
        'high_spender_probability': [0.92, 0.41, 0.87, 0.12],
        'spend_tier':               ['Very High', 'Medium', 'High', 'Low'],
        'churn_probability':        [0.75, 0.22, 0.10, 0.60],
        'churn_risk':               ['High', 'Low', 'Low', 'High'],
        'priority_score':           [0.69, 0.09, 0.09, 0.07],
        'action_priority':          [
            '🔴 URGENT - Retain High Spender',
            '⚪ Standard',
            '🟢 VIP Programme',
            '🟡 Win-back Campaign'
        ]
    })
    st.dataframe(example, use_container_width=True)

st.markdown("---")
st.markdown("Built by **Vikas Nagar** | **Ecommerce Customer Analytics Project**")
