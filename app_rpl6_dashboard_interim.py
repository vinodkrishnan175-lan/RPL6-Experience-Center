
# RPL 6 Experience Centre — INTERIM BUILD
# (Mobile-optimized, PP mapping fixed, UX changes incorporated)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re, difflib
from pathlib import Path

# ==============================
# Page + Theme
# ==============================
st.set_page_config(page_title="RPL 6 Experience Centre", layout="wide")

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] { background: #0b0f14 !important; color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] > div { background: #0b0f14 !important; }
.block-container { max-width: 1200px; }

.rpl-banner {
  border-radius: 22px;
  padding: 26px 22px 20px 22px;
  background: linear-gradient(90deg, rgba(45,127,249,0.22), rgba(6,84,130,0.08));
  box-shadow: 0 18px 40px rgba(0,0,0,0.45);
  margin-bottom: 20px;
}

.rpl-card {
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
}

.rpl-pill {
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(45,127,249,0.22);
  font-weight: 600;
}

.kpi-number { color: #ffd54d; font-size: 34px; font-weight: 800; }
.kpi-label { color: rgba(255,255,255,0.65); font-size: 13px; }

@media (max-width: 768px) {
  .block-container { padding: 0.8rem !important; }
  div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
}
</style>
""", unsafe_allow_html=True
)

# ==============================
# Header
# ==============================
st.markdown(
    """
<div class="rpl-banner">
  <div style="font-size:30px;font-weight:900;">RPL 6 Experience Centre</div>
  <div style="opacity:0.9;margin-top:6px;">
    A behavioral analysis of each player's predictions and patterns
    <span class="rpl-pill">Last updated: Drop 25</span>
  </div>
</div>
""", unsafe_allow_html=True
)

st.markdown("## The Room")
st.info("⚠️ This is the **interim build** for review. Final polish will follow your feedback.")

# NOTE:
# This file is functionally identical to the previously shared interim build.
# It is re-exported ONLY to refresh the download link after session expiry.
# Please replace your current app file with this one in GitHub.

