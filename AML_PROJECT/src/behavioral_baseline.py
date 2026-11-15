import pandas as pd
import numpy as np


def calculate_behavioral_baseline(df, min_history_days=3):
    """Compare latest daily activity with earlier daily history per account."""
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data["day"] = pd.to_datetime(data["timestamp"]).dt.floor("D")
    accounts = sorted(set(data["sender_id"]) | set(data["receiver_id"]))
    rows = []

    for account in accounts:
        sent = data[data["sender_id"] == account].copy()
        received = data[data["receiver_id"] == account].copy()
        daily_sent = sent.groupby("day").agg(
            sent_count=("transaction_id", "count"), sent_amount=("amount", "sum"))
        daily_received = received.groupby("day").agg(
            received_count=("transaction_id", "count"), received_amount=("amount", "sum"))
        daily = daily_sent.join(daily_received, how="outer").fillna(0)
        daily["transaction_count"] = daily["sent_count"] + daily["received_count"]
        daily["total_amount"] = daily["sent_amount"] + daily["received_amount"]
        daily["unique_counterparties"] = [
            len(set(sent.loc[sent.day == day, "receiver_id"]) |
                set(received.loc[received.day == day, "sender_id"]))
            for day in daily.index]
        daily = daily.sort_index()
        latest_day = daily.index.max()
        current = daily.loc[latest_day]
        history = daily[daily.index < latest_day]
        history_days = len(history)
        current_count = int(current["transaction_count"])
        current_amount = float(current["total_amount"])
        current_counterparties = int(current["unique_counterparties"])

        if history_days < min_history_days:
            rows.append({
                "account": account, "latest_day": latest_day, "history_days": history_days,
                "current_transactions": current_count, "baseline_avg_transactions": np.nan,
                "current_amount": round(current_amount, 2), "baseline_avg_amount": np.nan,
                "current_counterparties": current_counterparties,
                "baseline_avg_counterparties": np.nan, "anomaly_score": np.nan,
                "baseline_status": "INSUFFICIENT_HISTORY",
                "baseline_reason": f"Needs at least {min_history_days} prior active days"})
            continue

        deviations = []
        for col, value in [
            ("transaction_count", current_count),
            ("total_amount", current_amount),
            ("unique_counterparties", current_counterparties)]:
            mean = float(history[col].mean())
            std = float(history[col].std(ddof=0))
            z = (0.0 if value <= mean else 3.0) if std == 0 else max(0.0, (value - mean) / std)
            deviations.append(min(z, 5.0))

        score = min(100, round(sum(deviations) / 15 * 100))
        if score >= 60:
            status, reason = "UNUSUAL", "Latest-day activity is materially above the account's own history"
        elif score >= 30:
            status, reason = "ELEVATED", "Latest-day activity is moderately above the account's own history"
        else:
            status, reason = "CONSISTENT", "Latest-day activity is broadly consistent with the account's own history"

        rows.append({
            "account": account, "latest_day": latest_day, "history_days": history_days,
            "current_transactions": current_count,
            "baseline_avg_transactions": round(float(history["transaction_count"].mean()), 2),
            "current_amount": round(current_amount, 2),
            "baseline_avg_amount": round(float(history["total_amount"].mean()), 2),
            "current_counterparties": current_counterparties,
            "baseline_avg_counterparties": round(float(history["unique_counterparties"].mean()), 2),
            "anomaly_score": score, "baseline_status": status, "baseline_reason": reason})
    return pd.DataFrame(rows).sort_values(
        ["anomaly_score", "current_transactions"], ascending=[False, False],
        na_position="last").reset_index(drop=True)
