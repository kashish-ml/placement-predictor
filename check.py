import pandas as pd

df = pd.read_csv("placement.csv")
print(df.head())
print(df.columns.tolist())
print(df.shape)