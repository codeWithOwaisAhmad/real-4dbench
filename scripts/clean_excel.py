import pandas as pd

# 1. Load the benchmark log
df_log = pd.read_excel("idea6_benchmark_log.xlsx")

# 2. Filter out all empty spacer rows
df_log = df_log.dropna(subset=["Clip ID"]).reset_index(drop=True)

print(f"Loaded {len(df_log)} valid clips.")  # Exactly 99 clips