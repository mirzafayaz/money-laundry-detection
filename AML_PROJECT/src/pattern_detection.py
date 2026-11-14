
def detect_fan_in(G, min_accounts=2):

    alerts = []

    for account in G.nodes():

        senders = list(G.predecessors(account))

        if len(senders) >= min_accounts:

            alerts.append({
                "account": account,
                "pattern": "FAN_IN",
                "connected_accounts": len(senders),
                "risk": "HIGH"
            })

    return alerts


def detect_fan_out(G, min_accounts=2):

    alerts = []

    for account in G.nodes():

        receivers = list(G.successors(account))

        if len(receivers) >= min_accounts:

            alerts.append({
                "account": account,
                "pattern": "FAN_OUT",
                "connected_accounts": len(receivers),
                "risk": "HIGH"
            })

    return alerts





#detection for circular transaction


def detect_circular(G, min_length=3, max_length=6):

    import networkx as nx

    alerts = []

    cycles = nx.simple_cycles(G)

    for cycle in cycles:

        if not (min_length <= len(cycle) <= max_length):
            continue

        transactions = []
        previous_time = None
        valid = True

        for i in range(len(cycle)):

            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]

            edges = G.get_edge_data(u, v)

            if not edges:
                valid = False
                break

            candidates = sorted(
                edges.values(),
                key=lambda x: x["timestamp"]
            )

            selected = next(
                (
                    edge for edge in candidates
                    if previous_time is None
                    or edge["timestamp"] > previous_time
                ),
                None
            )

            if selected is None:
                valid = False
                break

            transactions.append(selected)
            previous_time = selected["timestamp"]

        if valid:
            alerts.append({
                "account": cycle[0],
                "pattern": "CIRCULAR_ROUND_TRIPPING",
                "cycle": cycle + [cycle[0]],
                "connected_accounts": len(cycle),
                "risk": "HIGH"
            })

    return alerts


def detect_rapid_pass_through(
    G,
    max_minutes=10,
    min_ratio=0.70,
    max_ratio=1.05
):

    alerts = []

    for account in G.nodes():

        incoming = list(
            G.in_edges(account, keys=True, data=True)
        )

        outgoing = list(
            G.out_edges(account, keys=True, data=True)
        )

        if not incoming or not outgoing:
            continue

        latest_in = max(
            incoming,
            key=lambda x: x[3]["timestamp"]
        )

        earliest_out = min(
            outgoing,
            key=lambda x: x[3]["timestamp"]
        )

        time_gap = (
            earliest_out[3]["timestamp"]
            - latest_in[3]["timestamp"]
        ).total_seconds() / 60

        if time_gap < 0 or time_gap > max_minutes:
            continue

        total_in = sum(
            edge[3]["amount"] for edge in incoming
        )

        total_out = sum(
            edge[3]["amount"] for edge in outgoing
        )

        ratio = total_out / total_in if total_in else 0

        if min_ratio <= ratio <= max_ratio:

            alerts.append({
                "account": account,
                "pattern": "RAPID_PASS_THROUGH",
                "incoming_amount": total_in,
                "outgoing_amount": total_out,
                "amount_ratio": round(ratio, 3),
                "time_gap_minutes": round(time_gap, 2),
                "risk": "HIGH"
            })

    return alerts




def detect_patterns(G):

    fan_in = detect_fan_in(G)
    fan_out = detect_fan_out(G)
    circular = detect_circular(G)
    rapid = detect_rapid_pass_through(G)

    return fan_in + fan_out + circular + rapid


