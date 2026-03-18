# Model Card: Congestion Risk Classifier
**Generated:** 2026-03-15 09:37 UTC
**Task:** Binary classification — Congestion risk (top 20% waits)
**Algorithm:** XGBoost Classifier with class weighting
**Input:** 46 features
**Congestion threshold:** ≥ 11.1h waiting time

## Performance (Test Set)
| Metric | Value | Target |
|--------|-------|--------|
| AUC-ROC | 0.991 | — |
| Precision @ recall≥0.80 | 0.969 | > 0.80 |
| Recall | 0.802 | ≥ 0.80 |
| Decision threshold | 0.892 | — |

## Top Features (XGBoost importance)
| Feature | Importance |
|---------|-----------|
| berth_competition | 0.3797 |
| weather_wind_knots | 0.0595 |
| weather_storm_flag | 0.0175 |
| port_name_enc | 0.0453 |
| day_of_week | 0.0419 |
| port_haifa | 0.0378 |
| week_of_year | 0.0186 |
| month_sin | 0.0172 |

## Files
- `models/congestion_risk.pkl`
- `models/model_cards/shap_congestion_risk.png`
- `models/model_cards/pr_curve_congestion.png`
