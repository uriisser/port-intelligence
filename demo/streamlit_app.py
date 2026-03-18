"""
demo/streamlit_app.py — Port Intelligence Live Dashboard (Standalone)
Runs without the FastAPI backend — loads models directly.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "port_calls.parquet"
MODELS_DIR = ROOT / "models"
CARDS_DIR  = MODELS_DIR / "model_cards"

st.set_page_config(
    page_title="Port Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m1 = joblib.load(MODELS_DIR / "waiting_time_ensemble.pkl")
    m2 = joblib.load(MODELS_DIR / "berth_occupancy.pkl")
    m3 = joblib.load(MODELS_DIR / "congestion_risk.pkl")
    return m1, m2, m3

@st.cache_data
def load_history():
    try:
        df = pd.read_parquet(DATA_PATH)
        df['ata_actual'] = pd.to_datetime(df['ata_actual'])
        return df
    except Exception as e:
        return pd.DataFrame()

# ── Feature builder ───────────────────────────────────────────────────────────
PORT_MAP = {"Haifa": 0, "Ashdod": 1}
VESSEL_TYPE_MAP = {"CONTAINER": 0, "BULK": 1, "GENERAL_CARGO": 2, "RORO": 3, "TANKER": 4}

def build_features(vessel_type, teu_capacity, dwt, loa, draft,
                   port_name, berth_id, service_line,
                   eta_dt, cranes_used, cargo_tons,
                   teu_loaded, teu_discharged,
                   weather_wind, berth_comp,
                   arrivals_6h, arrivals_12h, arrivals_24h, queue_pos):

    hour = eta_dt.hour
    dow  = eta_dt.weekday()
    month = eta_dt.month

    f = np.zeros(46, dtype=np.float32)
    f[0]  = hour
    f[1]  = dow
    f[2]  = month
    f[3]  = np.sin(2 * np.pi * hour / 24)
    f[4]  = np.cos(2 * np.pi * hour / 24)
    f[5]  = np.sin(2 * np.pi * dow / 7)
    f[6]  = np.cos(2 * np.pi * dow / 7)
    f[7]  = np.sin(2 * np.pi * month / 12)
    f[8]  = np.cos(2 * np.pi * month / 12)
    f[9]  = int(dow >= 5)
    f[10] = int(hour < 6 or hour >= 22)
    f[11] = int(8 <= hour <= 18)
    f[12] = int(month in [6, 7, 8])
    f[13] = int(month in [12, 1, 2])
    f[14] = 0
    f[15] = arrivals_6h
    f[16] = arrivals_12h
    f[17] = teu_capacity
    f[18] = dwt
    f[19] = loa
    f[20] = draft
    f[21] = cargo_tons
    f[22] = teu_loaded
    f[23] = teu_discharged
    f[24] = teu_loaded + teu_discharged
    f[25] = cargo_tons / max(dwt, 1)
    f[26] = cranes_used
    f[27] = queue_pos
    f[28] = arrivals_24h
    f[29] = arrivals_12h / max(arrivals_24h, 1)
    f[30] = berth_comp
    f[31] = berth_comp * arrivals_12h
    f[32] = berth_comp ** 2
    f[33] = arrivals_6h / max(cranes_used, 1)
    f[34] = queue_pos * berth_comp
    f[35] = 0
    f[36] = 0
    f[37] = 0
    f[38] = 0
    f[39] = weather_wind
    f[40] = int(weather_wind > 30)
    f[41] = berth_comp
    f[42] = cargo_tons / max(teu_capacity or 1, 1)
    f[43] = teu_discharged / max(teu_loaded + teu_discharged + 1, 1)
    f[44] = VESSEL_TYPE_MAP.get(vessel_type, 0)
    f[45] = PORT_MAP.get(port_name, 0)

    return f.reshape(1, -1)

def run_predict(m1, m2, m3, X):
    xgb_p = float(m1['xgb_reg'].predict(X)[0])
    lgb_p  = float(m1['lgb_reg'].predict(X)[0])
    w      = m1['ensemble_weight']
    wait   = max(0.0, w * xgb_p + (1 - w) * lgb_p)
    mae    = m1['metrics']['mae']
    ci     = [round(max(0, wait - 1.5 * mae), 1), round(wait + 1.5 * mae, 1)]

    occ_idx   = int(m2['model'].predict(X)[0])
    occ_proba = m2['model'].predict_proba(X)[0]
    classes   = m2['label_names']
    occ_class = classes[occ_idx]
    occ_probs = {c: round(float(p), 3) for c, p in zip(classes, occ_proba)}

    cong_prob = float(m3['model'].predict_proba(X)[0][1])
    threshold = m3.get('decision_threshold', 0.5)
    cong_flag = cong_prob >= threshold

    return round(wait, 1), ci, occ_class, occ_probs, round(cong_prob, 3), cong_flag

# ── Load everything ───────────────────────────────────────────────────────────
m1, m2, m3 = load_models()
df_hist     = load_history()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚢 Port Intelligence")
    st.caption("Israeli Ports Authority · ML Forecasting System")
    st.markdown("---")
    st.success("Models loaded")
    st.metric("Waiting Time MAE", f"{m1['metrics']['mae']:.2f}h")
    st.metric("Congestion AUC",   f"{m3['metrics']['auc']:.3f}")
    st.metric("Features", "46")
    st.markdown("---")
    st.caption(f"Data: {len(df_hist):,} port calls" if not df_hist.empty else "Data not loaded")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚢 Live Prediction",
    "⚓ Berth Forecast",
    "📊 Historical KPIs",
    "🎯 Model Accuracy",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Live Vessel Prediction")
    st.caption("Submit a vessel arrival to get waiting time forecast, berth recommendation, and congestion risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Vessel Info")
        vessel_id    = st.text_input("Vessel ID / IMO", value="MSC-DIANA-2024")
        vessel_type  = st.selectbox("Vessel Type", ["CONTAINER", "BULK", "GENERAL_CARGO", "RORO", "TANKER"])
        teu_capacity = st.slider("TEU Capacity", 0, 24000, 8000, 500,
                                  disabled=(vessel_type != "CONTAINER"))
        dwt          = st.number_input("DWT (tons)", 1000, 500000, 80000, 1000)
        loa          = st.slider("LOA (m)", 50, 450, 250)
        draft        = st.slider("Draft (m)", 2.0, 18.0, 12.0, 0.1)

    with col2:
        st.subheader("Port & Schedule")
        port_name    = st.selectbox("Port", ["Haifa", "Ashdod"])
        berth_prefix = "H" if port_name == "Haifa" else "A"
        max_berths   = 20 if port_name == "Haifa" else 15
        berth_id     = st.selectbox("Requested Berth",
                                     [f"{berth_prefix}{i:02d}" for i in range(1, max_berths + 1)])
        service_line = st.selectbox("Service Line", [
            "Asia-EU", "Med-India", "Asia-Med", "Intra-Med", "Red-Sea-Med",
            "North-EU", "West-Africa", "East-Africa", "Americas", "Adriatic"])
        eta_date     = st.date_input("ETA Date", value=date.today() + timedelta(days=2))
        eta_hour     = st.slider("ETA Hour (UTC)", 0, 23, 10)
        eta_dt       = datetime(eta_date.year, eta_date.month, eta_date.day, eta_hour)
        teu_cap      = max(teu_capacity, 1)
        teu_loaded      = st.slider("TEU Loaded",     0, teu_cap, teu_cap // 2,
                                     disabled=vessel_type != "CONTAINER")
        teu_discharged  = st.slider("TEU Discharged", 0, teu_cap, teu_cap // 2,
                                     disabled=vessel_type != "CONTAINER")
        cargo_tons      = st.number_input("Cargo (tons)", 0, 500000, 80000, 1000)
        cranes_used     = st.slider("Cranes Assigned", 0, 8, 3)

    with col3:
        st.subheader("Conditions")
        weather_wind = st.slider("Wind (knots)", 0.0, 60.0, 8.0, 0.5)
        berth_comp   = st.slider("Berth Competition (0=empty, 5=severe)", 0.0, 5.0, 1.0, 0.1)
        arrivals_12h = st.slider("Arrivals in last 12h (same port)", 0, 50, 10)
        arrivals_6h  = arrivals_12h // 2
        arrivals_24h = arrivals_12h * 2
        queue_pos    = st.slider("Queue Position", 1, 30, 5)
        st.markdown("---")
        predict_btn  = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        X = build_features(
            vessel_type, teu_capacity, dwt, loa, draft,
            port_name, berth_id, service_line, eta_dt,
            cranes_used, float(cargo_tons),
            teu_loaded, teu_discharged,
            weather_wind, berth_comp,
            arrivals_6h, arrivals_12h, arrivals_24h, queue_pos
        )
        wait, ci, occ_class, occ_probs, cong_prob, cong_flag = run_predict(m1, m2, m3, X)
        rec_berth = f"{berth_prefix}{'03' if berth_comp < 1.5 else '07'}"

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Anchor Wait", f"{wait}h", f"CI [{ci[0]}–{ci[1]}h]")
        r2.metric("Congestion Risk", f"{cong_prob:.0%}",
                  delta="HIGH RISK" if cong_flag else "Normal",
                  delta_color="inverse" if cong_flag else "off")
        r3.metric("Recommended Berth", rec_berth)
        r4.metric("Berth Occupancy", occ_class)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_occ = go.Figure(go.Bar(
                x=list(occ_probs.keys()),
                y=list(occ_probs.values()),
                marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
            ))
            fig_occ.update_layout(title="Berth Occupancy Probability",
                                   yaxis_tickformat=".0%", height=300,
                                   margin=dict(t=40, b=20))
            st.plotly_chart(fig_occ, use_container_width=True)

        with col_b:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cong_prob * 100,
                title={"text": "Congestion Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if cong_flag else "#2ecc71"},
                    "steps": [
                        {"range": [0,  40], "color": "#d5f5e3"},
                        {"range": [40, 70], "color": "#fdebd0"},
                        {"range": [70, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                  "thickness": 0.75, "value": 80},
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("Raw Prediction"):
            st.json({
                "vessel_id": vessel_id,
                "port_name": port_name,
                "waiting_anchor_forecast": wait,
                "confidence_interval": ci,
                "recommended_berth": rec_berth,
                "congestion_risk": cong_prob,
                "congestion_flag": cong_flag,
                "occupancy_class": occ_class,
                "occupancy_probabilities": occ_probs,
            })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BERTH FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("24h Berth Utilization Forecast")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        port_b   = st.selectbox("Port", ["Haifa", "Ashdod"], key="port_b")
        prefix_b = "H" if port_b == "Haifa" else "A"
        max_b    = 20 if port_b == "Haifa" else 15
        berth_b  = st.selectbox("Berth", [f"{prefix_b}{i:02d}" for i in range(1, max_b + 1)], key="berth_b")
        fc_date  = st.date_input("Forecast Date", value=date.today() + timedelta(days=1), key="fc_date")
        bc_val   = st.slider("Expected Berth Competition", 0.0, 5.0, 1.5, 0.1, key="bc_fc")
        run_fc   = st.button("Get Forecast", type="primary", key="run_fc")

    with col_r:
        if run_fc:
            hours, utils, classes = [], [], []
            for h in range(24):
                eta_fc = datetime(fc_date.year, fc_date.month, fc_date.day, h)
                bc_h   = bc_val * (1.2 if 8 <= h <= 18 else 0.7)
                X_fc   = build_features(
                    "CONTAINER", 8000, 80000, 250, 12.0,
                    port_b, f"{prefix_b}01", "Asia-EU", eta_fc,
                    3, 80000.0, 4000, 4000, 8.0, bc_h,
                    5, 10, 20, 5
                )
                wait_h, _, occ_cls, occ_p, _, _ = run_predict(m1, m2, m3, X_fc)
                util = min(1.0, wait_h / 24.0 + occ_p.get("High", 0) * 0.5 + occ_p.get("Medium", 0) * 0.3)
                hours.append(h)
                utils.append(round(util, 2))
                classes.append(occ_cls)

            color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
            fig_fc = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in hours],
                y=utils,
                marker_color=[color_map.get(c, "#95a5a6") for c in classes],
                text=classes,
                textposition="outside",
            ))
            fig_fc.update_layout(
                title=f"Berth {berth_b} — {fc_date} — Hourly Utilization",
                xaxis_title="Hour (UTC)", yaxis_title="Utilization",
                yaxis_range=[0, 1.2], height=400,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.dataframe(pd.DataFrame({
                "Hour":        [f"{h:02d}:00" for h in hours],
                "Utilization": [f"{u:.0%}" for u in utils],
                "Class":       classes,
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Select a berth and date, then click **Get Forecast**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORICAL KPIs
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Historical Port KPIs (Jan 2024 – Dec 2025)")

    if df_hist.empty:
        st.warning("Historical data not available.")
    else:
        df_hist['month']     = df_hist['ata_actual'].dt.to_period('M').dt.to_timestamp()
        df_hist['total_teu'] = df_hist['teu_loaded'] + df_hist['teu_discharged']

        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Total Calls", f"{len(df_hist):,}")
        m2c.metric("Total TEU",   f"{df_hist['total_teu'].sum():,.0f}")
        m3c.metric("Median Wait", f"{df_hist['waiting_anchor_hours'].median():.1f}h")
        m4c.metric("P80 Wait",    f"{np.percentile(df_hist['waiting_anchor_hours'], 80):.1f}h")

        st.markdown("---")
        monthly_teu = df_hist.groupby(['month', 'port_name'])['total_teu'].sum().reset_index()
        fig_teu = px.bar(monthly_teu, x='month', y='total_teu', color='port_name',
                         title='Monthly TEU by Port',
                         labels={'total_teu': 'TEU', 'month': 'Month', 'port_name': 'Port'},
                         color_discrete_map={'Haifa': '#3498db', 'Ashdod': '#e67e22'},
                         barmode='group')
        fig_teu.update_layout(height=350)
        st.plotly_chart(fig_teu, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_wait = px.histogram(df_hist, x='waiting_anchor_hours', color='port_name',
                                    nbins=60, title='Waiting Time Distribution',
                                    labels={'waiting_anchor_hours': 'Anchor Wait (h)', 'port_name': 'Port'},
                                    color_discrete_map={'Haifa': '#3498db', 'Ashdod': '#e67e22'},
                                    opacity=0.75, barmode='overlay', range_x=[0, 50])
            p80 = np.percentile(df_hist['waiting_anchor_hours'], 80)
            p95 = np.percentile(df_hist['waiting_anchor_hours'], 95)
            fig_wait.add_vline(x=p80, line_dash="dash", line_color="orange",
                               annotation_text=f"P80={p80:.1f}h")
            fig_wait.add_vline(x=p95, line_dash="dash", line_color="red",
                               annotation_text=f"P95={p95:.1f}h")
            fig_wait.update_layout(height=350)
            st.plotly_chart(fig_wait, use_container_width=True)

        with col_b:
            vtype_cnt = df_hist.groupby(['vessel_type', 'port_name']).size().reset_index(name='calls')
            fig_vtype = px.bar(vtype_cnt, x='vessel_type', y='calls', color='port_name',
                               title='Vessel Type Mix',
                               labels={'calls': 'Calls', 'vessel_type': 'Type', 'port_name': 'Port'},
                               color_discrete_map={'Haifa': '#3498db', 'Ashdod': '#e67e22'},
                               barmode='group')
            fig_vtype.update_layout(height=350)
            st.plotly_chart(fig_vtype, use_container_width=True)

        df_hist['dow']  = df_hist['ata_actual'].dt.dayofweek
        df_hist['hour'] = df_hist['ata_actual'].dt.hour
        heatmap_data  = df_hist.groupby(['dow', 'hour']).size().reset_index(name='calls')
        heatmap_pivot = heatmap_data.pivot(index='dow', columns='hour', values='calls').fillna(0)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=[days[i] for i in heatmap_pivot.index],
            colorscale='Blues', colorbar_title='Calls',
        ))
        fig_heat.update_layout(title='Arrival Pattern (Day × Hour)', height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Performance Summary")
    st.caption("Test-set metrics from Phase 2 training (Jan–Aug 2024 train / Nov–Dec 2025 test)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Model 1: Waiting Time")
        st.metric("MAE",  f"{m1['metrics']['mae']:.2f}h",  "< 4h target")
        st.metric("MAPE", f"{m1['metrics']['mape']:.1f}%", "< 25% target")
        st.metric("R²",   f"{m1['metrics']['r2']:.3f}",    "> 0.75 target")
    with col2:
        st.subheader("Model 2: Berth Occupancy")
        st.metric("Accuracy", f"{m2['metrics']['accuracy']:.3f}")
        st.metric("Macro F1", f"{m2['metrics']['macro_f1']:.3f}")
    with col3:
        st.subheader("Model 3: Congestion Risk")
        st.metric("AUC-ROC",   f"{m3['metrics']['auc']:.3f}")
        st.metric("Precision", f"{m3['metrics']['precision']:.3f}", "> 0.80 target")
        st.metric("Recall",    f"{m3['metrics']['recall']:.3f}",    ">= 0.80 target")

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    shap_files = {
        "Waiting Time":    CARDS_DIR / "shap_waiting_time.png",
        "Berth Occupancy": CARDS_DIR / "shap_berth_occupancy.png",
        "Congestion Risk": CARDS_DIR / "shap_congestion_risk.png",
    }
    selected = st.selectbox("Select Model", list(shap_files.keys()))
    shap_path = shap_files[selected]
    if shap_path.exists():
        st.image(str(shap_path), use_column_width=True)
    else:
        st.info("SHAP plots available after running train_models.py locally.")

    pr_path  = CARDS_DIR / "pr_curve_congestion.png"
    res_path = CARDS_DIR / "residuals_waiting_time.png"
    if pr_path.exists():
        st.subheader("Congestion Risk — Precision-Recall Curve")
        st.image(str(pr_path), width=600)
    if res_path.exists():
        st.subheader("Waiting Time — Residual Plot")
        st.image(str(res_path), use_column_width=True)
