
import pandas as pd

def load_transactions(file):

    df = pd.read_csv(file)

    required = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    df = df.dropna(subset=required)

    df = df[
        (df["amount"] > 0) &
        (df["sender_id"] != df["receiver_id"])
    ].copy()

    return df.sort_values("timestamp")
