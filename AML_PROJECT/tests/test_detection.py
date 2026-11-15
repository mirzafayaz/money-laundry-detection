
from src.data_loader import load_transactions
from src.graph_builder import build_graph
from src.pattern_detection import detect_patterns

df = load_transactions("data/raw/transactions.csv")

G = build_graph(df)

alerts = detect_patterns(G)

print("\nTRACEFIN AI - DETECTION RESULTS")
print("--------------------------------")

for alert in alerts:
    print(alert)
