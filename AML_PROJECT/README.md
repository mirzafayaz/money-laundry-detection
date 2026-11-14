# TRACEFIN AI

Graph-based financial transaction analysis prototype for AML investigation workflows.

## Features
- Interactive command center with transaction KPIs and activity trends
- Transaction network explorer with account focus and risk-coloured nodes
- Rule-based detection: fan-in, fan-out, circular flow, and rapid pass-through
- Account-level risk scoring and explainable investigation queue
- Searchable transaction ledger and CSV exports
- Optional Random Forest model insights when labelled data is supplied
- CSV upload and date / amount filters

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV schema
Required columns: `transaction_id`, `timestamp`, `sender_id`, `receiver_id`, `amount`.
Optional column: `label` (0 = normal, 1 = suspicious) for exploratory supervised model analysis.

## Important limitations
This is a hackathon prototype. Fan-in and fan-out are network-shape indicators and can describe legitimate merchants, marketplaces, payroll, payment aggregators, and collection accounts. Risk scores are rule-based triage signals, not proof of financial crime. Model metrics from small or synthetic datasets are not evidence of production performance. Real deployments require calibrated thresholds, customer/business context, robust validation, privacy controls, audit trails, and human review.
