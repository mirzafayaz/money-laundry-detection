
def generate_explanation(result):

    account = result["account"]
    patterns = result["patterns"]

    descriptions = {
        "FAN_IN": "Receives funds from multiple accounts.",
        "FAN_OUT": "Distributes funds to multiple accounts.",
        "CIRCULAR_ROUND_TRIPPING":
            "Participates in a circular transaction network.",
        "RAPID_PASS_THROUGH":
            "Transfers substantial funds shortly after receiving them."
    }

    reasons = [
        descriptions[p]
        for p in patterns
        if p in descriptions
    ]

    return (
        f"Account {account} has a risk score of "
        f"{result['risk_score']}/100. "
        + " ".join(reasons)
    )
