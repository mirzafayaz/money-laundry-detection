"""Flexible CSV ingestion for TRACEFIN AI.

Canonical transaction schema used internally:
transaction_id, timestamp, sender_id, receiver_id, amount, [label]
"""
import re
import pandas as pd

ALIASES = {
    "transaction_id": ["transaction_id", "transactionid", "txn_id", "txnid", "txn_ref", "transaction_ref", "reference", "reference_id", "ref", "payment_id", "transfer_id", "id"],
    "timestamp": ["timestamp", "datetime", "date_time", "transaction_date", "transaction_datetime", "txn_date", "txn_time", "created_at", "date", "time", "value_date"],
    "sender_id": ["sender_id", "sender", "from_account", "from_account_id", "source_account", "source", "origin_account", "payer_id", "debit_account", "account_from", "from_id"],
    "receiver_id": ["receiver_id", "receiver", "to_account", "to_account_id", "destination_account", "destination", "beneficiary_account", "payee_id", "credit_account", "account_to", "to_id"],
    "amount": ["amount", "transaction_amount", "transaction_value", "transfer_amount", "value", "txn_amount", "payment_amount", "amount_inr", "amount_usd", "debit_amount", "credit_amount"],
    "label": ["label", "is_fraud", "fraud", "fraud_label", "is_suspicious", "suspicious", "target", "class", "is_illicit", "aml_label"],
}
REQUIRED = ["timestamp", "sender_id", "receiver_id", "amount"]


def _norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def detect_column_mapping(columns):
    """Return best-effort source-column -> canonical-column mapping."""
    normalized = {_norm(c): c for c in columns}
    mapping = {}
    used = set()
    for canonical, aliases in ALIASES.items():
        candidates = [_norm(a) for a in aliases]
        match = next((normalized[a] for a in candidates if a in normalized and normalized[a] not in used), None)
        if match is None:
            # Conservative fuzzy matching for common decorated headers.
            for col in columns:
                n = _norm(col)
                if col in used:
                    continue
                if canonical == "timestamp" and any(x in n for x in ("timestamp", "datetime", "transactiondate")):
                    match = col; break
                if canonical == "sender_id" and any(x in n for x in ("sender", "fromaccount", "sourceaccount")):
                    match = col; break
                if canonical == "receiver_id" and any(x in n for x in ("receiver", "toaccount", "destinationaccount", "beneficiary")):
                    match = col; break
                if canonical == "amount" and ("amount" in n or "value" in n) and not any(x in n for x in ("balance", "fee")):
                    match = col; break
            
        if match is not None:
            mapping[canonical] = match
            used.add(match)
    return mapping


def load_transactions(file, column_mapping=None):
    """Read a transaction CSV and normalize varying source headers."""
    try:
        df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, sep=None, engine="python", encoding="latin-1")

    if df.empty:
        raise ValueError("The CSV contains no transaction rows.")

    mapping = detect_column_mapping(df.columns)
    if column_mapping:
        mapping.update({k: v for k, v in column_mapping.items() if v in df.columns})

    missing = [c for c in REQUIRED if c not in mapping]
    if missing:
        raise ValueError("Could not identify: " + ", ".join(missing) + ". Use the CSV column-mapping controls to select the correct columns. Required transaction fields are timestamp/date, sender account, receiver account and amount.")

    rename = {source: canonical for canonical, source in mapping.items() if source in df.columns}
    # Avoid collisions if a file already contains canonical names plus alternate aliases.
    df = df.rename(columns=rename)

    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"TXN-{i+1:07d}" for i in range(len(df))]
    if "label" in df.columns:
        raw_label = df["label"].astype(str).str.strip().str.lower()
        df["label"] = raw_label.map({"1": 1, "true": 1, "yes": 1, "fraud": 1, "suspicious": 1, "2": 1,
                                      "0": 0, "false": 0, "no": 0, "normal": 0, "legitimate": 0})
        # Keep numeric binary labels where possible; unknown values remain missing.
        numeric = pd.to_numeric(raw_label, errors="coerce")
        df["label"] = df["label"].fillna(numeric)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)
    amount_text = df["amount"].astype(str).str.replace(r"[₹$€£,\s]", "", regex=True)
    df["amount"] = pd.to_numeric(amount_text, errors="coerce")
    for col in ("sender_id", "receiver_id"):
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA

    before = len(df)
    df = df.dropna(subset=REQUIRED)
    df = df[(df["amount"] > 0) & (df["sender_id"] != df["receiver_id"])].copy()
    if df.empty:
        raise ValueError(f"No valid transactions remain after validation (read {before} rows). Check date, account and amount mappings/values.")
    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)
    return df.sort_values("timestamp").reset_index(drop=True)
