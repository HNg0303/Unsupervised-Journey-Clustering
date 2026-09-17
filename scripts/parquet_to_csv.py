from pathlib import Path
import pandas as pd

source = Path("output/scores/july/android/model_version=latest/platform=android")
target = source.parent / "android_scores.csv"

first = True
for file in sorted(source.glob("*.parquet")):
    frame = pd.read_parquet(file)
    frame.to_csv(target, mode="w" if first else "a", header=first, index=False)
    first = False

print(target)