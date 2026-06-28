import pandas as pd

df = pd.read_csv("data/loan.csv.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)
import pandas as pd

df = pd.read_csv("data/loan.csv.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
import pandas as pd

df = pd.read_csv("data/loan.csv.csv")

print(df.columns.tolist())