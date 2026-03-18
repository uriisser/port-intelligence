# Model Card: Congestion Risk Classifier
## Model 3 of 3 — Port Intelligence Platform v1.0

---

## Model Summary

| Attribute | Value |
|-----------|-------|
| **Task** | Binary classification — congestion risk flag |
| **Target variable** | `congestion_flag` (top 20% waiting times = 1) |
| **Algorithm** | XGBoost Classifier with class weighting |
| **Class balance** | 20% positive (congestion), 80% negative |
| **Decision threshold** | 0.892 (optimized for precision ≥ 0.80 @ recall ≥ 0.80) |
| **Features** | 46 |
| **Training rows** | 60,801 |
| **Test rows** | 6,375 |
| **Model file** | `models/congestion_risk.pkl` |

---

## Congestion Definition

A vessel call is labelled **congested** if its `waiting_anchor_hours` exceeds the **P80 of the training distribution** (currently 11.1 hours).

This means:
- In a typical week: ~20% of arrivals will be flagged as potentially congested
- The threshold is recalculated on every retrain (changes as port patterns change)

---

## Performance (Test Set)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUC-ROC | **0.991** | — | Excellent |
| Precision @ recall ≥ 0.80 | **0.969** | > 0.80 | PASS |
| Recall | **0.802** | ≥ 0.80 | PASS |
| Decision threshold | **0.892** | — | |

**Reading the operating point:**
- Out of 100 vessels the model flags as congested: **97 will actually be congested** (precision)
- Out of 100 actually congested vessels: **80 are correctly flagged** (recall)
- 20% of truly congested vessels are missed (false negatives)
- 3% of alerted vessels are false alarms (high-precision design)

---

## Threshold Tuning Philosophy

The threshold (0.892) was chosen to **maximize precision while maintaining recall ≥ 0.80**.

**Why this matters for port operations:**
- High precision = duty managers trust the alerts (low false alarm fatigue)
- Recall ≥ 0.80 = most genuine congestion events are caught
- The 20% miss rate is acceptable because Model 1 (waiting time regression) provides a second line of detection

**Alternative operating points (if needed):**

| Goal | Threshold | Precision | Recall |
|------|-----------|-----------|--------|
| Maximum recall | ~0.30 | ~0.70 | ~0.95 |
| Balanced (current) | 0.892 | 0.969 | 0.802 |
| Maximum precision | ~0.98 | ~0.99 | ~0.50 |

To change threshold without retraining:
```python
bundle = joblib.load('models/congestion_risk.pkl')
bundle['decision_threshold'] = 0.50  # new threshold
joblib.dump(bundle, 'models/congestion_risk.pkl')
# Restart API to reload
```

---

## Alert Workflow

```
Model 3 score ≥ 0.892
       │
       ▼
congestion_flag = TRUE returned in API response
       │
       ├─► Duty manager notification (email/SMS/dashboard alert)
       │
       ├─► Check Model 1 waiting time forecast:
       │    • If waiting_anchor_forecast > 12h: HIGH PRIORITY alert
       │    • If waiting_anchor_forecast 6–12h: MEDIUM PRIORITY
       │    • If waiting_anchor_forecast < 6h: may be false positive, monitor
       │
       └─► Response options:
            • Redirect vessel to alternative berth
            • Delay arrival by 4–6h (coordinate with agent)
            • Activate emergency crane allocation
            • Notify downstream logistics (rail, trucks, warehouses)
```

---

## Key Features (Importance)

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `berth_competition` | Primary congestion driver |
| 2 | `weather_wind_knots` | Storm conditions force all vessels to wait |
| 3 | `arrivals_12h` | Rolling arrival pressure |
| 4 | `berth_competition_ratio` | Demand vs capacity |
| 5 | `weather_storm_flag` | Hard binary storm threshold |
| 6 | `teu_cap_norm` | Large vessels disproportionately block berths |
| 7 | `day_of_week` | Monday/Tuesday consistently higher congestion |
| 8 | `hour_sin` / `hour_cos` | Morning peak arrival window |

---

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Binary output loses nuance | Can't distinguish "slightly congested" from "crisis" | Combine with Model 1 waiting time |
| 20% miss rate | Some congestion events go undetected | Use Model 1 MAE as backup signal |
| Fixed training threshold | P80 changes as port grows | Recalculate on each retrain |
| No supply-chain context | Strike action, rail disruption not modeled | Add `external_disruption_flag` feature |
| Weather is point-in-time | Cannot forecast congestion 48h ahead | Integrate NWP (numerical weather prediction) |

---

## Hyperparameters

```python
n_estimators=700, learning_rate=0.05, max_depth=7,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
reg_lambda=2.0, eval_metric='auc',
scale_pos_weight=4.0  # compensates 80/20 class imbalance
```

---

## Retraining Notes

- Congestion threshold (P80) is recalculated automatically on each retrain
- Decision threshold is re-optimized from the new PR curve
- If port expands (new berths), congestion rate may drop — model will adapt
- Monitor false positive rate monthly via `monitoring/model_performance.py`
