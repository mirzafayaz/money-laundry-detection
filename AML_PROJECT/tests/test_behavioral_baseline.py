import pandas as pd
from src.behavioral_baseline import calculate_behavioral_baseline


def test_baseline_flags_account_specific_spike():
    rows = []
    tx = 0
    # Account B has a stable one-transaction-per-day history for 4 days.
    for day in range(1, 5):
        tx += 1
        rows.append({
            "transaction_id": f"T{tx}", "timestamp": f"2026-11-{day:02d} 10:00:00",
            "sender_id": "B", "receiver_id": f"R{day}", "amount": 100.0
        })
    # On day five it suddenly transacts with many counterparties.
    for i in range(10):
        tx += 1
        rows.append({
            "transaction_id": f"T{tx}", "timestamp": f"2026-11-05 10:{i:02d}:00",
            "sender_id": "B", "receiver_id": f"X{i}", "amount": 100.0
        })
    result = calculate_behavioral_baseline(pd.DataFrame(rows))
    account = result[result.account == "B"].iloc[0]
    assert account["baseline_status"] == "UNUSUAL"
    assert account["history_days"] == 4


def test_single_day_is_insufficient_history():
    df = pd.DataFrame([{
        "transaction_id": "T1", "timestamp": "2026-11-14 10:00:00",
        "sender_id": "A", "receiver_id": "B", "amount": 500.0
    }])
    result = calculate_behavioral_baseline(df)
    assert set(result["baseline_status"]) == {"INSUFFICIENT_HISTORY"}
