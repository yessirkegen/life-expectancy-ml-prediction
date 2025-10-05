import pandas as pd
df = pd.read_csv("data/processed/master.csv")
print(sorted(df["year"].unique()))
print(df["year"].value_counts())
