import streamlit as st
import sys
import os
from pathlib import Path

st.title("Port Intelligence — Startup Diagnostics")
st.write(f"Python: {sys.version}")
st.write(f"Working dir: {os.getcwd()}")

ROOT = Path(__file__).parent.parent
st.write(f"ROOT: {ROOT}")
st.write(f"ROOT exists: {ROOT.exists()}")

# Test each import individually
imports = {
    "numpy": "import numpy as np; st.success(f'numpy {np.__version__}')",
    "pandas": "import pandas as pd; st.success(f'pandas {pd.__version__}')",
    "plotly": "import plotly; st.success(f'plotly {plotly.__version__}')",
    "joblib": "import joblib; st.success(f'joblib {joblib.__version__}')",
    "xgboost": "import xgboost as xgb; st.success(f'xgboost {xgb.__version__}')",
    "lightgbm": "import lightgbm as lgb; st.success(f'lightgbm {lgb.__version__}')",
    "sklearn": "import sklearn; st.success(f'sklearn {sklearn.__version__}')",
    "pyarrow": "import pyarrow as pa; st.success(f'pyarrow {pa.__version__}')",
}

for name, code in imports.items():
    try:
        exec(code)
    except Exception as e:
        st.error(f"{name}: {e}")

# Check files
st.subheader("Model files")
models_dir = ROOT / "models"
if models_dir.exists():
    for f in models_dir.glob("*.pkl"):
        st.write(f"  {f.name} ({f.stat().st_size // 1024}KB)")
else:
    st.error(f"models/ directory not found at {models_dir}")

st.subheader("Data files")
data_dir = ROOT / "data"
if data_dir.exists():
    for f in data_dir.glob("*.parquet"):
        st.write(f"  {f.name} ({f.stat().st_size // 1024}KB)")
else:
    st.error(f"data/ directory not found at {data_dir}")

# Try loading models
st.subheader("Model loading")
try:
    import joblib
    m1 = joblib.load(models_dir / "waiting_time_ensemble.pkl")
    st.success("waiting_time_ensemble.pkl loaded OK")
except Exception as e:
    st.error(f"waiting_time_ensemble.pkl: {e}")

try:
    import joblib
    m2 = joblib.load(models_dir / "berth_occupancy.pkl")
    st.success("berth_occupancy.pkl loaded OK")
except Exception as e:
    st.error(f"berth_occupancy.pkl: {e}")

try:
    import joblib
    m3 = joblib.load(models_dir / "congestion_risk.pkl")
    st.success("congestion_risk.pkl loaded OK")
except Exception as e:
    st.error(f"congestion_risk.pkl: {e}")
