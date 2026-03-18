"""
demo/streamlit_app.py — Port Intelligence Live Dashboard

Tabs:
  1. Live Prediction    — submit a vessel and get instant forecast
  2. Berth Forecast     — 24h hourly occupancy heat map
  3. Historical KPIs    — monthly TEU, waiting time distributions
  4. Model Accuracy     — live MAE tracking from prediction log
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "port_calls.parquet")

st.set_page_config(
    page_title="Port Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


@st.cache_data(ttl=300)
def load_history():
    try:
        df = pd.read_parquet(DATA_PATH)
        df['ata_actual'] = pd.to_datetime(df['ata_actual'])
        return df
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def api_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.json()
    except Exception:
        return {}


def post_predict(payload: dict) -> dict:
    r = requests.post(f"{API_BASE}/predict_vessel", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def get_berth_forecast(berth_id: str, forecast_date: str, port_name: str) -> dict:
    r = requests.get(
        f"{API_BASE}/berth_forecast/{berth_id}/{forecast_date}",
        params={"port_name": port_name},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Container_ship_Yorkville.jpg/320px-Container_ship_Yorkville.jpg",
             use_column_width=True)
    st.title("Port Intelligence")
    st.caption("Israeli Ports Authority · ML Forecasting System")

    health = api_health()
    status_color = "green" if health.get("status") == "ok" else "red"
    st.markdown(f"**API Status:** :{status_color}[{health.get('status','unknown').upper()}]")
    st.markdown(f"**Cache:** {'On' if health.get('cache_enabled') else 'Off'}")

    if health.get("status") == "ok":
        m = api_metrics()
        if m:
            st.markdown("---")
            st.markdown("**Model Performance (Test Set)**")
            st.metric("Waiting Time MAE", f"{m['model1_waiting_time'].get('mae', 0):.2f}h")
            st.metric("Congestion AUC", f"{m['model3_congestion'].get('auc', 0):.3f}")
            st.metric("Features", m.get("feature_count", 0))

    st.markdown("---")
    st.markdown("[API Docs](http://localhost:8000/docs) | [ReDoc](http://localhost:8000/redoc)")


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
        vessel_type  = st.selectbox("Vessel Type", ["CONTAINER","BULK","GENERAL_CARGO","RORO","TANKER"])
        teu_capacity = st.slider("TEU Capacity", 0, 24000, 8000, 500,
                                  disabled=(vessel_type != "CONTAINER"))
        dwt          = st.number_input("DWT (tons)", 1000, 500000, 80000, 1000)
        loa          = st.slider("LOA (m)", 50, 450, 250)
        draft        = st.slider("Draft (m)", 2.0, 18.0, 12.0, 0.1)
        company_name = st.selectbox("Shipping Company", [
            "Maersk","MSC","CMA CGM","COSCO","Hapag-Lloyd","ONE",
            "Evergreen","ZIM","Yang Ming","Other"])

    with col2:
        st.subheader("Port & Schedule")
        port_name    = st.selectbox("Port", ["Haifa", "Ashdod"])
        berth_prefix = "H" if port_name == "Haifa" else "A"
        max_berths   = 20 if port_name == "Haifa" else 15
        berth_id     = st.selectbox("Requested Berth",
                                     [f"{berth_prefix}{i:02d}" for i in range(1, max_berths+1)])
        service_line = st.selectbox("Service Line", [
            "Asia-EU","Med-India","Asia-Med","Intra-Med","Red-Sea-Med",
            "North-EU","West-Africa","East-Africa","Americas","Adriatic"])
        eta_date     = st.date_input("ETA Date", value=date.today() + timedelta(days=2))
        eta_hour     = st.slider("ETA Hour (UTC)", 0, 23, 10)
        eta_planned  = datetime(eta_date.year, eta_date.month, eta_date.day, eta_hour)

        teu_loaded      = st.slider("TEU Loaded",      0, teu_capacity or 1, (teu_capacity or 1)//2,
                                     disabled=vessel_type != "CONTAINER")
        teu_discharged  = st.slider("TEU Discharged",  0, teu_capacity or 1, (teu_capacity or 1)//2,
                                     disabled=vessel_type != "CONTAINER")
        cargo_tons      = st.number_input("Cargo (tons)", 0, 500000, 80000, 1000)
        cranes_used     = st.slider("Cranes Assigned", 0, 8, 3)

    with col3:
        st.subheader("Conditions")
        weather_wind    = st.slider("Wind (knots)", 0.0, 60.0, 8.0, 0.5)
        berth_comp      = st.slider("Berth Competition (0=empty, 5=severe)", 0.0, 5.0, 1.0, 0.1)
        arrivals_12h    = st.slider("Arrivals in last 12h (same port)", 0, 50, 10)
        arrivals_6h     = arrivals_12h // 2
        arrivals_24h    = arrivals_12h * 2
        queue_pos       = st.slider("Queue Position", 1, 30, 5)

        st.markdown("---")
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        if health.get("status") != "ok":
            st.error("API is not reachable. Start it with: `make up` or `uvicorn api.main:app`")
        else:
            payload = {
                "vessel_id": vessel_id,
                "port_name": port_name,
                "eta_planned": eta_planned.isoformat(),
                "vessel_type": vessel_type,
                "teu_capacity": teu_capacity,
                "dwt": dwt,
                "loa": loa,
                "draft": draft,
                "company_name": company_name,
                "service_line": service_line,
                "berth_id": berth_id,
                "cranes_used": cranes_used,
                "cargo_tons": float(cargo_tons),
                "teu_loaded": teu_loaded,
                "teu_discharged": teu_discharged,
                "weather_wind_knots": weather_wind,
                "berth_competition": berth_comp,
                "arrivals_6h": arrivals_6h,
                "arrivals_12h": arrivals_12h,
                "arrivals_24h": arrivals_24h,
                "queue_position": queue_pos,
            }
            with st.spinner("Running ML inference..."):
                try:
                    result = post_predict(payload)

                    # ── Result cards ──────────────────────────────────────────
                    st.markdown("---")
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric(
                            "Anchor Wait",
                            f"{result['waiting_anchor_forecast']}h",
                            delta=f"CI [{result['confidence_interval'][0]}–{result['confidence_interval'][1]}h]",
                        )
                    with r2:
                        cong = result['congestion_risk']
                        flag = result['congestion_flag']
                        st.metric(
                            "Congestion Risk",
                            f"{cong:.0%}",
                            delta="HIGH RISK" if flag else "Normal",
                            delta_color="inverse" if flag else "off",
                        )
                    with r3:
                        st.metric("Recommended Berth", result['recommended_berth'])
                    with r4:
                        st.metric("Berth Occupancy", result['occupancy_class'])

                    # ── Occupancy probabilities ───────────────────────────────
                    occ_prob = result['occupancy_probabilities']
                    fig_occ = go.Figure(go.Bar(
                        x=list(occ_prob.keys()),
                        y=list(occ_prob.values()),
                        marker_color=['#2ecc71','#f39c12','#e74c3c'],
                    ))
                    fig_occ.update_layout(
                        title="Berth Occupancy Probability",
                        yaxis_tickformat=".0%",
                        height=300,
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_occ, use_container_width=True)

                    # ── Gauge chart for congestion ────────────────────────────
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=cong * 100,
                        title={"text": "Congestion Risk (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e74c3c" if flag else "#2ecc71"},
                            "steps": [
                                {"range": [0,  40], "color": "#d5f5e3"},
                                {"range": [40, 70], "color": "#fdebd0"},
                                {"range": [70,100], "color": "#fadbd8"},
                            ],
                            "threshold": {"line": {"color": "red", "width": 4},
                                          "thickness": 0.75, "value": 80},
                        },
                        delta={"reference": 50},
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(t=40,b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # ── Raw JSON ──────────────────────────────────────────────
                    with st.expander("Raw API Response"):
                        st.json(result)

                except requests.HTTPError as e:
                    st.error(f"API Error {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BERTH FORECAST (24h heat map)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("24h Berth Utilization Forecast")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        port_b     = st.selectbox("Port", ["Haifa", "Ashdod"], key="port_b")
        prefix_b   = "H" if port_b == "Haifa" else "A"
        max_b      = 20 if port_b == "Haifa" else 15
        berth_b    = st.selectbox("Berth", [f"{prefix_b}{i:02d}" for i in range(1, max_b+1)],
                                   key="berth_b")
        fc_date    = st.date_input("Forecast Date", value=date.today() + timedelta(days=1),
                                    key="fc_date")
        run_fc     = st.button("Get Forecast", type="primary", key="run_fc")

    with col_r:
        if run_fc:
            if health.get("status") != "ok":
                st.error("API not reachable.")
            else:
                with st.spinner("Generating 24h forecast..."):
                    try:
                        fc = get_berth_forecast(berth_b, fc_date.isoformat(), port_b)
                        preds = fc["predictions"]
                        hours = [p["hour"] for p in preds]
                        utils = [p["utilization"] for p in preds]
                        classes = [p["occupancy_class"] for p in preds]

                        color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                        colors = [color_map.get(c, "#95a5a6") for c in classes]

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Bar(
                            x=[f"{h:02d}:00" for h in hours],
                            y=utils,
                            marker_color=colors,
                            text=classes,
                            textposition="outside",
                        ))
                        fig_fc.update_layout(
                            title=f"Berth {berth_b} — {fc_date} — Hourly Utilization",
                            xaxis_title="Hour (UTC)",
                            yaxis_title="Utilization",
                            yaxis_range=[0, 1.1],
                            height=400,
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        # Table
                        tbl = pd.DataFrame([{
                            "Hour": f"{p['hour']:02d}:00",
                            "Utilization": f"{p['utilization']:.1%}",
                            "Class": p["occupancy_class"],
                            "P(Low)": f"{p['probabilities'].get('Low',0):.1%}",
                            "P(Med)": f"{p['probabilities'].get('Medium',0):.1%}",
                            "P(High)": f"{p['probabilities'].get('High',0):.1%}",
                        } for p in preds])
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

                    except requests.HTTPError as e:
                        st.error(f"API Error: {e.response.text}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.info("Select a berth and date, then click **Get Forecast**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORICAL KPIs
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Historical Port KPIs (Jan 2024 – Dec 2025)")

    df_hist = load_history()
    if df_hist.empty:
        st.warning("Historical data not available.")
    else:
        df_hist['month'] = df_hist['ata_actual'].dt.to_period('M').dt.to_timestamp()
        df_hist['total_teu'] = df_hist['teu_loaded'] + df_hist['teu_discharged']

        # Top-level metrics
        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Total Calls",  f"{len(df_hist):,}")
        m2c.metric("Total TEU",    f"{df_hist['total_teu'].sum():,.0f}")
        m3c.metric("Median Wait",  f"{df_hist['waiting_anchor_hours'].median():.1f}h")
        m4c.metric("P80 Wait",     f"{np.percentile(df_hist['waiting_anchor_hours'], 80):.1f}h")

        st.markdown("---")

        # Monthly TEU by port
        monthly_teu = df_hist.groupby(['month','port_name'])['total_teu'].sum().reset_index()
        fig_teu = px.bar(monthly_teu, x='month', y='total_teu', color='port_name',
                         title='Monthly TEU by Port',
                         labels={'total_teu':'TEU','month':'Month','port_name':'Port'},
                         color_discrete_map={'Haifa':'#3498db','Ashdod':'#e67e22'},
                         barmode='group')
        fig_teu.update_layout(height=350)
        st.plotly_chart(fig_teu, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Waiting time distribution
            fig_wait = px.histogram(
                df_hist, x='waiting_anchor_hours', color='port_name',
                nbins=60, title='Waiting Time Distribution',
                labels={'waiting_anchor_hours':'Anchor Wait (h)','port_name':'Port'},
                color_discrete_map={'Haifa':'#3498db','Ashdod':'#e67e22'},
                opacity=0.75, barmode='overlay',
                range_x=[0, 50],
            )
            # Add P80/P95 lines
            p80 = np.percentile(df_hist['waiting_anchor_hours'], 80)
            p95 = np.percentile(df_hist['waiting_anchor_hours'], 95)
            fig_wait.add_vline(x=p80, line_dash="dash", line_color="orange",
                               annotation_text=f"P80={p80:.1f}h")
            fig_wait.add_vline(x=p95, line_dash="dash", line_color="red",
                               annotation_text=f"P95={p95:.1f}h")
            fig_wait.update_layout(height=350)
            st.plotly_chart(fig_wait, use_container_width=True)

        with col_b:
            # Vessel type breakdown
            vtype_cnt = df_hist.groupby(['vessel_type','port_name']).size().reset_index(name='calls')
            fig_vtype = px.bar(vtype_cnt, x='vessel_type', y='calls', color='port_name',
                                title='Vessel Type Mix',
                                labels={'calls':'Calls','vessel_type':'Type','port_name':'Port'},
                                color_discrete_map={'Haifa':'#3498db','Ashdod':'#e67e22'},
                                barmode='group')
            fig_vtype.update_layout(height=350)
            st.plotly_chart(fig_vtype, use_container_width=True)

        # Weekly pattern heatmap
        df_hist['dow']  = df_hist['ata_actual'].dt.dayofweek
        df_hist['hour'] = df_hist['ata_actual'].dt.hour
        heatmap_data = df_hist.groupby(['dow','hour'])['id'].count().reset_index(name='calls')
        heatmap_pivot = heatmap_data.pivot(index='dow', columns='hour', values='calls').fillna(0)
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=[days[i] for i in heatmap_pivot.index],
            colorscale='Blues',
            colorbar_title='Calls',
        ))
        fig_heat.update_layout(title='Arrival Pattern (Day × Hour)', height=300)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Model Performance Summary")
    st.caption("Test-set metrics from Phase 2 training (Jan–Aug 2024 train / Nov–Dec 2025 test)")

    m = api_metrics()
    if m:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Model 1: Waiting Time")
            st.metric("MAE",  f"{m['model1_waiting_time'].get('mae', 0):.2f}h", "< 4h target")
            st.metric("MAPE", f"{m['model1_waiting_time'].get('mape', 0):.1f}%", "< 25% target")
            st.metric("R²",   f"{m['model1_waiting_time'].get('r2', 0):.3f}",   "> 0.75 target")

        with col2:
            st.subheader("Model 2: Berth Occupancy")
            st.metric("Accuracy", f"{m['model2_occupancy'].get('accuracy', 0):.3f}")
            st.metric("Macro F1", f"{m['model2_occupancy'].get('macro_f1', 0):.3f}")

        with col3:
            st.subheader("Model 3: Congestion Risk")
            st.metric("AUC-ROC",  f"{m['model3_congestion'].get('auc', 0):.3f}")
            st.metric("Precision",f"{m['model3_congestion'].get('precision', 0):.3f}", "> 0.80 target")
            st.metric("Recall",   f"{m['model3_congestion'].get('recall', 0):.3f}",    ">= 0.80 target")

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    shap_files = {
        "Waiting Time": "../models/model_cards/shap_waiting_time.png",
        "Berth Occupancy": "../models/model_cards/shap_berth_occupancy.png",
        "Congestion Risk": "../models/model_cards/shap_congestion_risk.png",
    }
    selected_shap = st.selectbox("Select Model", list(shap_files.keys()))
    shap_path = os.path.join(os.path.dirname(__file__), shap_files[selected_shap])
    if os.path.exists(shap_path):
        st.image(shap_path, use_column_width=True)
    else:
        st.info("SHAP plots available after running train_models.py")

    # PR curve
    pr_path = os.path.join(os.path.dirname(__file__), "../models/model_cards/pr_curve_congestion.png")
    if os.path.exists(pr_path):
        st.subheader("Congestion Risk — Precision-Recall Curve")
        st.image(pr_path, width=600)

    # Residual plot
    res_path = os.path.join(os.path.dirname(__file__), "../models/model_cards/residuals_waiting_time.png")
    if os.path.exists(res_path):
        st.subheader("Waiting Time — Residual Plot")
        st.image(res_path, use_column_width=True)
