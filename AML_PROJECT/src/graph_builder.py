
import networkx as nx

def build_graph(df):

    G = nx.MultiDiGraph()

    for _, row in df.iterrows():

        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            transaction_id=row["transaction_id"],
            amount=float(row["amount"]),
            timestamp=row["timestamp"]
        )

    return G
