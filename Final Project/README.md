# Conflict Situation Monitoring for Maritime Shipping
### CE49X — Introduction to Data Science for Civil Engineering
**Boğaziçi University, Spring 2026 · Dr. Eyuphan Koç**

**Team:** Ahmet Emre Bingöl (2021403162) · Ayda Tuna (2022403264)

---

## Project Overview

This project builds a situation-monitoring pipeline that correlates **NASA FIRMS satellite thermal anomalies** with **war/conflict news** and an independent **ground-truth conflict database (ACLED)** to assess armed-conflict activity in regions critical to global shipping and energy markets.

**Regions:** Ukraine, Yemen, Iraq  
**Period:** January 2024 – June 2024  
**Core question:** Can satellite-detected thermal anomalies serve as indicators of armed conflict?

A key, honestly-reported finding is that the thermal signal **alone** cannot separate conflict fires from industrial gas flaring — which is exactly why the pipeline fuses thermal + news + ACLED ground truth.

---

## Repository Structure

```
Final Project/
├── conflict_monitoring.ipynb     # Main notebook — all code, analysis & discussion
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for API keys (copy to .env — see Setup)
├── Final_Project.pdf             # Project specification
│
├── acled_events_raw.csv          # Cached ACLED ground-truth events (so the notebook runs without creds)
├── brent_2024.csv                # Cached Brent crude prices (yfinance BZ=F)
│
├── dashboard.png                 # Multi-panel summary dashboard (300 DPI)
├── temporal_analysis.png         # Monthly event counts & FRP trends
├── daynight_split.png            # Day vs. night detection split
├── spatial_analysis.png          # Thermal events & top hotspots
├── news_coverage.png             # News volume & conflict-association rate
├── keyword_heatmap.png           # Conflict-keyword frequency by region
├── satellite_news_delay.png      # Offset to nearest news article
├── acled_validation.png          # News heuristic vs ACLED (confusion matrix)
├── acled_overlay.png             # ACLED vs thermal spatial overlay
├── lead_lag_ccf.png              # Thermal–conflict cross-correlation
├── flare_masking.png             # Industrial flare masking map
├── spatial_significance.png      # Moran's I / Getis-Ord Gi* hotspots
├── confusion_matrices.png        # ML classifier confusion matrices
├── feature_importance.png        # Decision-tree feature importances
├── ablation_comparison.png       # Full vs physical-only features (leakage check)
├── cost_overlay.png              # Yemen thermal activity vs Brent crude
├── rerouting_alerts.png          # Red Sea rerouting alert days
├── hedge_signal.png              # Basra FRP anomalies vs Brent
└── interactive_map.html          # Standalone interactive operator map (open in a browser)
```

---

## Prerequisites

- Python 3.8+ (developed on 3.10)
- Docker Desktop (running)
- Anaconda or pip

---

## Setup

### 1. PostgreSQL with Docker

```powershell
docker run --name ce49x-postgres `
  -e POSTGRES_USER=ce49x `
  -e POSTGRES_HOST_AUTH_METHOD=trust `
  -e POSTGRES_DB=conflict_monitoring `
  -p 5432:5432 `
  -d postgres:16
```

Verify / stop / restart:

```powershell
docker ps
docker stop ce49x-postgres
docker start ce49x-postgres
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. API keys (`.env`)

The notebook reads API keys from a `.env` file (kept out of the repo via `.gitignore`). Copy the template and fill in your own keys:

```bash
cp .env.example .env   # then edit .env
```

`.env` keys used:

| Variable | Used for | Get a key |
|---|---|---|
| `FIRMS_MAP_KEY` | NASA FIRMS thermal data | https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| `GUARDIAN_API_KEY` | The Guardian news | https://open-platform.theguardian.com/access/ |
| `ACLED_USER` / `ACLED_PASS` | ACLED ground-truth events (optional) | https://acleddata.com/register/ |

> **Runs without keys too:** GDELT needs no key, and ACLED transparently falls back to the committed `acled_events_raw.csv` cache if `ACLED_USER`/`ACLED_PASS` are unset — so a grader can reproduce the analysis without registering. Only the live FIRMS/Guardian fetch requires keys.

> **Reproducibility note:** FIRMS, GDELT and Guardian are fetched live, so exact counts vary slightly run-to-run. GDELT's public endpoint is rate-limited and on a typical run only one of the three region queries returns data (which one varies). The Guardian (stable, ~1,500 articles) carries the bulk of the news corpus, and ACLED/Brent are cached to CSV, so the conclusions are stable.

### 4. Run the analysis

```bash
cd "Final Project"
jupyter notebook conflict_monitoring.ipynb
```

Then **Kernel → Restart & Run All** (~10 min, dominated by the live FIRMS fetch of 140k+ records). The notebook runs top-to-bottom without errors.

---

## What the Notebook Does

**Task 1 — Data Collection & Assembly:** FIRMS thermal data + GDELT/Guardian news → cleaned → stored in PostgreSQL.
**Task 2 — Spatial & Temporal Analysis:** DBSCAN thermal-event clustering, monthly/FRP trends, day–night split, hotspot mapping.
**Task 3 — Correlation & Classification:** news–thermal matching, ACLED ground-truth validation (Cohen's κ), lead/lag cross-correlation, industrial flare masking, spatial significance (Moran's I, Getis-Ord Gi*), ML classifiers, and a physical-vs-geographic **ablation** that exposes target leakage.
**Task 4 — Dashboard, Insights & Reflection:** 300 DPI summary dashboard, business-impact ($) quantification, limitations, and an interactive operator map.

---

## Database Tables

| Table | Contents |
|---|---|
| `firms_detections` | Raw FIRMS thermal detection records (cleaned) |
| `news_articles` | Collected conflict news articles with metadata |
| `thermal_events` | DBSCAN-clustered thermal events with computed features |
| `event_matches` | Thermal-event ↔ news-article matching results (incl. confirming article) |
| `acled_events` | ACLED ground-truth verified conflict events |
| `thermal_events_masked` | Thermal events with the industrial-flare mask flag |

Verify tables:

```bash
docker exec -it ce49x-postgres psql -U ce49x -d conflict_monitoring -c "\dt"
```

---

## Data Sources

All sources are cited in the notebook with full endpoints and access dates.

| Source | Type | URL | Access |
|---|---|---|---|
| NASA FIRMS (VIIRS_SNPP_SP) | Satellite thermal anomalies | https://firms.modaps.eosdis.nasa.gov/api/area/ | May 2025 |
| GDELT v2 DOC API | News articles | https://api.gdeltproject.org/api/v2/doc/doc | May 2025 |
| The Guardian API | News articles | https://content.guardianapis.com/search | May 2025 |
| ACLED | Ground-truth conflict events | https://acleddata.com/api/acled/read | May 2025 |
| Brent crude (ICE, `BZ=F`) | Oil price (via `yfinance`) | https://finance.yahoo.com/quote/BZ=F | May 2025 |

---

## Pipeline Architecture

```
NASA FIRMS API  ──►  firms_detections (DB)  ──►  DBSCAN clustering ──►  thermal_events (DB)
News APIs       ──►  news_articles (DB)                                      │
ACLED API/cache ──►  acled_events (DB)                                       ▼
                                                       news matching ──►  event_matches (DB)
                                                                             │
                                  flare masking ──►  thermal_events_masked   ▼
                                                                     ML classification + ablation
                                                                             │
                                                                             ▼
                                                       dashboard.png · interactive_map.html
```
