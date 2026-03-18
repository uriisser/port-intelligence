# Model Card: Waiting Time Regression Ensemble
**Generated:** 2026-03-15 09:37 UTC
**Task:** Regression — Predict anchorage waiting time (0-96h)
**Algorithm:** XGBoost + LightGBM weighted ensemble (w=0.30)
**Input:** 46 features
**Training rows:** 60,801

## Performance (Test Set)
| Metric | Value | Target |
|--------|-------|--------|
| MAE | 1.22h | < 4h |
| MAPE | 17.7% | < 25% |
| R² | 0.938 | > 0.75 |

## Top Features (XGBoost importance)
| Feature | Importance |
|---------|-----------|
| berth_competition | 0.4821 |
| quarter | 0.1527 |
| weather_storm_flag | 0.0624 |
| weather_wind_knots | 0.0384 |
| month | 0.0368 |
| day_of_week | 0.0308 |
| day_of_month | 0.0276 |
| days_since_holiday | 0.0139 |
| dow_sin | 0.0128 |

## Files
- `models/waiting_time_ensemble.pkl`
- `models/model_cards/shap_waiting_time.png`
- `models/model_cards/residuals_waiting_time.png`

## Notes
- Target clipped to [0, 96]h at training time
- Ensemble weight optimized on validation set
