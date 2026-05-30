# Conflict Situation Monitoring for Maritime Shipping
### CE49X — Introduction to Data Science for Civil Engineering
**Boğaziçi University, Spring 2026 · Dr. Eyuphan Koç**

---

## Project Overview

This project builds a situation monitoring pipeline that correlates **NASA FIRMS satellite thermal anomalies** with **war and conflict news** to assess armed conflict activity in regions critical to global shipping and energy markets.

**Regions:** Ukraine, Yemen, Iraq  
**Period:** January 2024 – June 2024  
**Core question:** Can satellite-detected thermal anomalies serve as early indicators of armed conflict?

---

## Repository Structure

```
Final Project/
├── conflict_monitoring.ipynb   # Main notebook (all tasks)
├── dashboard.png               # Multi-panel summary dashboard (300 DPI)
├── temporal_analysis.png       # Task 2 temporal visualizations
├── spatial_analysis.png        # Task 2 spatial visualizations
├── news_coverage.png           # Task 3 news coverage charts
├── confusion_matrices.png      # Task 3 ML confusion matrices
├── feature_importance.png      # Task 3 decision tree feature importances
├── Final_Project.pdf           # Project specification
└── README.md                   # This file
```

---

## Prerequisites

- Python 3.8+
- Docker Desktop (running)
- Anaconda or pip

---

## 1. Set Up PostgreSQL with Docker

```powershell
docker run --name ce49x-postgres `
  -e POSTGRES_USER=ce49x `
  -e POSTGRES_HOST_AUTH_METHOD=trust `
  -e POSTGRES_DB=conflict_monitoring `
  -p 5432:5432 `
  -d postgres:16
```

Verify the container is running:

```powershell
docker ps
```

To stop and restart later:

```powershell
docker stop ce49x-postgres
docker start ce49x-postgres
```

---

## 2. Install Python Dependencies

```bash
pip install requests pandas numpy matplotlib seaborn scipy scikit-learn sqlalchemy psycopg2-binary jupyter
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## 3. Run the Analysis

Open the notebook:

```bash
cd "Final Project"
jupyter notebook conflict_monitoring.ipynb
```

Then run **Kernel → Restart & Run All**.

The notebook will:
1. Collect FIRMS thermal data from NASA API (~10 min, 144k+ records)
2. Collect conflict news from GDELT and The Guardian APIs
3. Save all data to PostgreSQL (`firms_detections`, `news_articles` tables)
4. Read data back from DB, cluster thermal events with DBSCAN
5. Save thermal events to DB (`thermal_events` table)
6. Match events with news, run ML classification
7. Save matches to DB (`event_matches` table)
8. Generate all visualizations and `dashboard.png`

---

## 4. Database Tables

| Table | Contents |
|---|---|
| `firms_detections` | Raw FIRMS thermal detection records (cleaned) |
| `news_articles` | Collected conflict news articles with metadata |
| `thermal_events` | Clustered thermal events with computed features |
| `event_matches` | Thermal event–news article matching results |

Verify tables:

```bash
docker exec -it ce49x-postgres psql -U ce49x -d conflict_monitoring -c "\dt"
```

---

## 5. Data Sources

| Source | Type | API |
|---|---|---|
| NASA FIRMS (VIIRS_SNPP_SP) | Satellite thermal data | https://firms.modaps.eosdis.nasa.gov/api/area/ |
| GDELT v2 DOC API | News articles | https://api.gdeltproject.org/api/v2/doc/doc |
| The Guardian API | News articles | https://content.guardianapis.com/search |

---

## 6. Pipeline Architecture

```
NASA FIRMS API  ──►  firms_detections (DB)  ──►  DBSCAN Clustering
                                                        │
News APIs       ──►  news_articles (DB)                ▼
                              │              thermal_events (DB)
                              └──────────►  Conflict Matching
                                                        │
                                                        ▼
                                           event_matches (DB)
                                                        │
                                                        ▼
                                           ML Classification
                                           Dashboard (dashboard.png)
```
