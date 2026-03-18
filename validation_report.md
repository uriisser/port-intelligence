# Validation Report — Port Calls Synthetic Dataset
**Generated:** 2026-03-15 09:34 UTC
**Script:** generate_data.py (seed=42)
**Period:** January 2024 – December 2025

---

## ✅ All Constraints Met

| Check | Target | Actual | Status |
|-------|--------|--------|--------|
| Total TEU | 6,280,000 | 6,280,000 | ✅ |
| Haifa port calls | 46,000 | 46,000 | ✅ |
| Ashdod port calls | 30,000 | 30,000 | ✅ |
| Total port calls | 76,000 | 76,000 | ✅ |
| Container vessel mix | 60–70% | 63.7% | ✅ |
| Waiting time P80 | < 12h | 11.1h | ✅ |
| Waiting time P95 | < 36h | 18.5h | ✅ |
| Waiting time P99 | < 72h | 26.5h | ✅ |

## Vessel Mix

| vessel_type   |   count |   pct |
|:--------------|--------:|------:|
| BULK          |   15140 |  19.9 |
| CONTAINER     |   48412 |  63.7 |
| GENERAL_CARGO |    7357 |   9.7 |
| RORO          |    3487 |   4.6 |
| TANKER        |    1604 |   2.1 |

## Output Files

| File | Description |
|------|-------------|
| `data/port_calls.parquet` | ML-ready dataset (76,000 rows × 27 columns) |
| `data/port_calls.sql` | PostgreSQL COPY dump |
| `data/data_summary_stats.csv` | Monthly TEU breakdown by port |
| `schema.sql` | DDL with all constraints |
| `generate_data.py` | Reproducible generator (seed=42) |
| `validation_report.md` | This file |

## Shape Check
```
df.shape == (76000, 27)
```

## Waiting Time Distribution
| Percentile | Hours |
|-----------|-------|
| P50 | 5.9h |
| P80 | 11.1h |
| P95 | 18.5h |
| P99 | 26.5h |
| Max | 59.8h |

---
**All 6.28M TEU ✓ | 76,000 calls ✓ | Waiting time percentiles ✓**
