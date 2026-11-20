import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Parcl Real Estate Buyer Intelligence", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("clustered_buyers.csv")

df = load_data()

# --- SIDEBAR USER CONTROLS ---
st.sidebar.header("Filter Buyer Profiles")

sel_country = st.sidebar.multiselect("Country", sorted(df["country"].unique()), default=sorted(df["country"].unique()))
sel_region = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
sel_purpose = st.sidebar.multiselect("Acquisition Purpose", sorted(df["acquisition_purpose"].unique()), default=sorted(df["acquisition_purpose"].unique()))
sel_client = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()), default=sorted(df["client_type"].unique()))

filtered_df = df[
    (df["country"].isin(sel_country)) &
    (df["region"].isin(sel_region)) &
    (df["acquisition_purpose"].isin(sel_purpose)) &
    (df["client_type"].isin(sel_client))
]

# --- MAIN DASHBOARD HEADER & KPIS ---
st.title("Parcl AI-Driven Buyer Segmentation & Investment Profiling")
st.markdown("Interactive market intelligence dashboard identifying distinct real estate buyer personas and investment behaviors.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Buyers Analyzed", f"{len(filtered_df):,}")
c2.metric("Avg Investment (INR)", f"₹{filtered_df['investment_amount_inr'].mean()/1e7:.2f} Cr" if len(filtered_df) else "₹0")
c3.metric("Average Buyer Age", f"{filtered_df['age'].mean():.1f} yrs" if len(filtered_df) else "0")
loan_pct = (filtered_df["loan_applied"].eq("Yes").mean() * 100) if len(filtered_df) else 0
c4.metric("Loan Dependency Rate", f"{loan_pct:.1f}%")

st.divider()

# --- 1. BUYER SEGMENTATION OVERVIEW ---
st.subheader("1. Buyer Segmentation Overview")
col1, col2 = st.columns(2)

with col1:
    seg_counts = filtered_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_pie = px.pie(seg_counts, names="Segment", values="Count", hole=0.4, title="Cluster Distribution Share")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_ref = px.histogram(
        filtered_df, x="Segment", color="referral_channel", barmode="group",
        title="Customer Acquisition Channels by Segment"
    )
    st.plotly_chart(fig_ref, use_container_width=True)

# --- 2. INVESTOR BEHAVIOR DASHBOARD ---
st.subheader("2. Investor Behavior Dashboard")
col3, col4 = st.columns(2)

with col3:
    fig_purpose = px.histogram(
        filtered_df, x="Segment", color="acquisition_purpose", barmode="stack",
        title="Acquisition Purpose Breakdown by Segment"
    )
    st.plotly_chart(fig_purpose, use_container_width=True)

with col4:
    fig_box = px.box(
        filtered_df, x="Segment", y="investment_amount_inr", color="loan_applied",
        title="Investment Amount (INR) & Loan Usage across Segments"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- 3. GEOGRAPHIC BUYER ANALYSIS ---
st.subheader("3. Geographic Buyer Analysis")
col5, col6 = st.columns(2)

with col5:
    geo_country = filtered_df.groupby(["country", "Segment"]).size().reset_index(name="Buyers")
    fig_country = px.bar(
        geo_country, x="country", y="Buyers", color="Segment", barmode="group",
        title="Buyer Segments by Country"
    )
    st.plotly_chart(fig_country, use_container_width=True)

with col6:
    fig_region = px.histogram(
        filtered_df, x="region", color="Segment", barmode="stack",
        title="Regional Concentration of Buyer Segments"
    )
    st.plotly_chart(fig_region, use_container_width=True)

# --- 4. SEGMENT INSIGHTS PANEL ---
st.subheader("4. Segment Insights Panel (Descriptive Statistics)")
numeric_cols = ["age", "satisfaction_score", "investment_amount_inr", "property_count"]
if len(filtered_df) > 0:
    summary_stats = filtered_df.groupby("Segment")[numeric_cols].agg(["mean", "median", "std"]).round(2)
    st.dataframe(summary_stats, use_container_width=True)
st.dataframe(filtered_df.head(25), use_container_width=True)
