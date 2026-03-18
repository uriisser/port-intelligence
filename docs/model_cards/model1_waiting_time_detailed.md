# Model Card: Waiting Time Regression Ensemble
## Model 1 of 3 — Port Intelligence Platform v1.0

---

## Model Summary

| Attribute | Value |
|-----------|-------|
| **Task** | Regression — predict anchorage waiting time (hours) |
| **Target variable** | `waiting_anchor_hours` (0–96h continuous) |
| **Algorithm** | XGBoost + LightGBM weighted ensemble |
| **Ensemble weight** | XGB=0.30, LGB=0.70 (optimized on val set) |
| **Features** | 46 (see feature importance section) |
| **Training rows** | 60,801 (Jan 2024 – Aug 2025) |
| **Test rows** | 6,375 (Nov–Dec 2025) |
| **Model file** | `models/waiting_time_ensemble.pkl` |
| **Version** | phase2-v1 |

---

## Performance (Test Set)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE (Mean Absolute Error) | **1.22h** | < 4h | PASS |
| MAPE (waits ≥ 3h) | **17.7%** | < 25% | PASS |
| R² (coefficient of determination) | **0.938** | > 0.75 | PASS |

**Interpretation:** On average, the model predicts within 1.2 hours of actual. For vessels with non-trivial waiting times (≥ 3h), the percentage error is 17.7%.

### Error Distribution
| Percentile | Absolute Error |
|-----------|----------------|
| Median | ~0.8h |
| P80 | ~2.1h |
| P95 | ~3.8h |
| P99 | ~5.5h |

---

## Key Features (XGBoost Importance)

| Rank | Feature | Category | Why it matters |
|------|---------|----------|----------------|
| 1 | `berth_competition` | Operational | Primary causal driver of anchor queue |
| 2 | `weather_wind_knots` | Weather | Storms force vessels to wait for safe berthing |
| 3 | `teu_cap_norm` | Vessel | Larger vessels take longer to berth |
| 4 | `berth_competition_ratio` | Operational | Queue relative to berth capacity |
| 5 | `arrivals_12h` | Operational | Rolling demand window |
| 6 | `weather_storm_flag` | Weather | Binary storm indicator |
| 7 | `hour_of_day` | Temporal | Peak hours 08:00–18:00 |
| 8 | `day_of_week` | Temporal | Monday/Tuesday busiest |
| 9 | `holiday_flag` | Temporal | −30% traffic on Jewish holidays |
| 10 | `queue_position` | Operational | Position in arrival queue |

---

## Confidence Interval

The API returns a 90% confidence interval computed as `point ± 1.5 × MAE`:

- MAE = 1.22h → CI width ≈ ±1.83h
- Example: forecast = 7.5h → CI = [5.7h, 9.3h]

This approximation is valid for normal-distributed errors. For very low waits (< 2h), the lower bound is clipped to 0.

---

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Synthetic training data** | True patterns may differ from real TOS | Replace with real data via `retrain.py` |
| **No AIS integration** | Cannot see actual vessel positions | Feed `arrivals_Xh` from live AIS counts |
| **Static weather** | Uses arrival-time wind only | Could integrate IMS forecast for ETA-based prediction |
| **No berth maintenance schedule** | Planned closures not modeled | Add `berth_maintenance_flag` feature |
| **Holiday list is fixed** | New holidays / strikes not automatic | Update `JEWISH_HOLIDAYS` in `generate_data.py` and `features.py` |
| **CI is approximate** | Not a true Bayesian interval | For formal uncertainty: use conformal prediction or quantile regression |

---

## When to Retrain

Trigger retraining when any of the following occur:
1. Rolling 7-day MAE > 4h (50% above target)
2. Data drift alert: `berth_competition` KS p-value < 0.05
3. New port infrastructure (additional berths, new cranes)
4. Major policy change (new scheduling system, new shipping routes)
5. Monthly scheduled retrain (1st of each month, `retrain.py`)

---

## Model Hyperparameters

### XGBoost
```python
n_estimators=800, learning_rate=0.05, max_depth=7,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
reg_lambda=1.5, reg_alpha=0.5, objective='reg:squarederror'
```

### LightGBM
```python
n_estimators=800, learning_rate=0.05, max_depth=7,
subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
reg_lambda=1.5, reg_alpha=0.5, objective='regression'
```

Both trained with `early_stopping_rounds=50` on validation set.

---

## Ethical Considerations

- **Fairness:** Model does not use vessel flag, captain nationality, or cargo origin as features.
- **Transparency:** SHAP values available for any prediction; port authority can explain decisions.
- **Operational impact:** Predictions should supplement dispatcher judgment, not replace it.
