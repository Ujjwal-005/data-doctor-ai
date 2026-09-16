# Data Doctor AI

A universal tabular business analytics and machine-learning workspace that adapts to the data a user uploads.

## What it does
- Automatic field discovery and schema understanding
- Data quality and cleaning audit
- Power BI-style business dashboards
- Cross-dataset relationship detection and SQL model suggestions
- Multi-model ML comparison for supported prediction problems
- Time-aware demand/activity forecasting
- Anomaly detection and clustering
- Business story and action recommendations
- Business Owner and Data / Power BI Analyst modes

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deployment
Designed for Streamlit Community Cloud. The app is a Streamlit-only deployment and does not require the previous live API layer.
