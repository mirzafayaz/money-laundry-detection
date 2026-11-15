
from src.data_loader import load_transactions
from src.graph_builder import build_graph

# Load transaction dataset
df = load_transactions("data/raw/transactions.csv")

# Build transaction graph
G = build_graph(df)

# Display results
print("TRACEFIN AI - GRAPH TEST")
print("-------------------------")

print("Total transactions:", len(df))
print("Total accounts:", G.number_of_nodes())
print("Total graph edges:", G.number_of_edges())

print("\nAccounts:")
print(list(G.nodes()))

print("\nTransaction Network:")

for u, v, data in G.edges(data=True):
    print(u, "->", v, "| Amount:", data["amount"])
