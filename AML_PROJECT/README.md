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


## Two-stage verification and behavioral baseline
- **Stage 1:** graph typologies (fan-in, fan-out, circular transfers, rapid pass-through).
- **Stage 2:** account-specific behavioral comparison using daily transaction count, total amount, and unique counterparties.
- The baseline compares the latest observed day against earlier active days for the same account. At least three prior active days are required.
- The baseline anomaly score is a transparent heuristic triage score, not a probability of financial crime.
- The included demonstration CSV contains only one calendar day, so baseline results correctly show `INSUFFICIENT_HISTORY`. Upload multi-day transaction history to activate account-specific comparisons.

Run the baseline unit tests with:
```bash
python -m pytest tests/test_behavioral_baseline.py
```

## Flexible CSV uploads

TRACEFIN AI accepts transaction CSV exports with different header names. It automatically recognizes common variants (for example `txn_ref`, `from_account`, `to_account`, `transaction_value`, `transaction_date`) and provides a **Review / map CSV columns** panel in the sidebar for unfamiliar headers.

Minimum required transaction information:
- Date or timestamp
- Sender/source account identifier
- Receiver/destination account identifier
- Positive transaction amount

Transaction ID is optional (TRACEFIN generates one when absent). A fraud/suspicious label is optional; graph and rule-based analysis still work without labels, while supervised ML evaluation is skipped without usable labels.

This is flexible ingestion for **transaction datasets**, not arbitrary unrelated CSV files. If a file has no transaction date, source account, destination account, or amount, it cannot support the current money-flow and temporal analysis. CSV files with thousands separators in amounts should quote those values according to standard CSV format (for example `"5,000.00"`).

## Final Command Center dashboard refresh
The Command Center now follows the supplied one-page reference's warm charcoal / bronze visual direction and information hierarchy:
- Command Center overview with concise workflow panels and monitoring KPIs
- Transaction-to-case funnel and High / Medium / Low account-risk distribution
- Ranked account case queue and recent transaction stream
- Selected-account explanation, related transactions and mini network replay
- Model, CSV safeguards and accessibility notes
- Animated page/card/chart entrances, hover feedback, responsive layout and reduced-motion support
- Sidebar dashboard window scale control (Compact / Standard / Large)

The other workspaces remain available in the navigation tabs: Network Explorer, Alert Investigation, Transactions & Reports, Model Insights, and Two-Stage Verification. The dashboard's displayed counts and amounts are calculated from the currently loaded dataset and filters.
