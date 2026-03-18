# Model Card: Berth Occupancy Classifier
**Generated:** 2026-03-15 09:37 UTC
**Task:** 3-class classification — Low / Medium / High occupancy
**Algorithm:** XGBoost Classifier
**Input:** 46 features

## Performance (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 1.000 |
| Macro F1 | 1.000 |

## Class Definitions
| Class | Definition |
|-------|-----------|
| Low (0) | berth_competition_ratio < 1.0 |
| Medium (1) | 1.0 ≤ ratio < 2.0 |
| High (2) | ratio ≥ 2.0 |

## Top Features (XGBoost importance)
| Feature | Importance |
|---------|-----------|
| port_name_enc | 0.4538 |
| berth_competition_ratio | 0.2403 |
| arrivals_12h | 0.0934 |
| service_frequency | 0.0797 |
| port_haifa | 0.0495 |
| queue_position | 0.0269 |
| day_of_week | 0.0139 |
| berth_zone_enc | 0.0139 |
| crane_sharing_risk | 0.0092 |
| dow_cos | 0.0036 |

## Files
- `models/berth_occupancy.pkl`
- `models/model_cards/shap_berth_occupancy.png`
