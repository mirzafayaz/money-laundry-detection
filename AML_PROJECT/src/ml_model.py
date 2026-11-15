
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def extract_account_features(df):

    accounts = set(df["sender_id"]) | set(df["receiver_id"])

    features = []

    for account in accounts:

        sent = df[df["sender_id"] == account]
        received = df[df["receiver_id"] == account]

        total_sent = sent["amount"].sum()
        total_received = received["amount"].sum()

        features.append({
            "account": account,
            "total_sent": total_sent,
            "total_received": total_received,
            "transactions_sent": len(sent),
            "transactions_received": len(received),
            "unique_senders": received["sender_id"].nunique(),
            "unique_receivers": sent["receiver_id"].nunique(),
            "avg_sent": sent["amount"].mean() if len(sent) else 0,
            "avg_received": received["amount"].mean() if len(received) else 0,
            "label": int(
                ((sent["label"] == 1).any()) or
                ((received["label"] == 1).any())
            )
        })

    return pd.DataFrame(features)


def train_model(df):

    account_data = extract_account_features(df)

    X = account_data.drop(columns=["account", "label"])
    y = account_data["label"]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain both normal and suspicious accounts.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
        # Threshold-based baseline
    baseline_predictions = (
        (X_test["transactions_sent"] +
         X_test["transactions_received"]) >= 3
    ).astype(int)

    baseline_results = {
        "precision": precision_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, baseline_predictions, zero_division=0
        )
    }

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "report": classification_report(
            y_test, predictions, zero_division=0
        )
    }
    
        # Predict risk for all accounts for dashboard display
    account_data["ml_prediction"] = model.predict(X)
    account_data["suspicious_probability"] = (
        model.predict_proba(X)[:, 1] * 100
    ).round(2)
    
        # Predict all account risks
    account_data["ml_prediction"] = model.predict(X)

    account_data["suspicious_probability"] = (
        model.predict_proba(X)[:, 1] * 100
    ).round(2)

    return model, account_data, results, baseline_results
