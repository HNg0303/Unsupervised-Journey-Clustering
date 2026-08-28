import pickle 
from pathlib import Path

print(Path.cwd())
pkl_path = Path.cwd() / "output/partitioned_runs/pca48_svd48_ngrams12_500k/latest/ios/ios_journey_scorer.pkl"
print(f"Pickle Path: {pkl_path}")

with open(pkl_path, "rb") as file:
    data = pickle.load(file)

print(data)