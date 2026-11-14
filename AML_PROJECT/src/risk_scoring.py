
PATTERN_WEIGHTS = {
    "FAN_IN": 20,
    "FAN_OUT": 20,
    "CIRCULAR_ROUND_TRIPPING": 35,
    "RAPID_PASS_THROUGH": 25
}


def calculate_risk(alerts):

    account_scores = {}
    account_patterns = {}

    for alert in alerts:

        account = alert["account"]
        pattern = alert["pattern"]

        if account not in account_scores:
            account_scores[account] = 0
            account_patterns[account] = set()

        # Avoid counting the same pattern twice
        if pattern not in account_patterns[account]:

            account_scores[account] += PATTERN_WEIGHTS.get(
                pattern, 0
            )

            account_patterns[account].add(pattern)

    results = []

    for account, score in account_scores.items():

        score = min(score, 100)

        if score >= 70:
            level = "HIGH"
        elif score >= 40:
            level = "MEDIUM"
        else:
            level = "LOW"

        results.append({
            "account": account,
            "risk_score": score,
            "risk_level": level,
            "patterns": list(account_patterns[account])
        })

    return sorted(
        results,
        key=lambda x: x["risk_score"],
        reverse=True
    )
