import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

data = {
    "Product": ["A", "B", "C", "D", "E", "F", "G"],
    "Price": [250, 300, 275, 4200, 260, 290, np.nan]
}
df = pd.DataFrame(data)

print("Part 1: Descriptive Statistics with Pandas")
print(df.describe())
print("Mean:", df["Price"].mean())
print("Median:", df["Price"].median())
print("Mode:", df["Price"].mode()[0])
print("Variance:", df["Price"].var())
print("Std Dev:", df["Price"].std())

print("\nPart 2: Skewness & Kurtosis with SciPy")
prices = df["Price"].dropna()
print("Skewness:", stats.skew(prices))
print("Kurtosis:", stats.kurtosis(prices))

print("\nPart 3: Detecting Missing Values")
print(df.isnull())
print(df.isnull().sum())

print("\nPart 4: Handling Missing Values")
df_dropped = df.dropna()
print("After dropna():")
print(df_dropped)

df_filled = df.copy()
df_filled["Price"] = df_filled["Price"].fillna(df_filled["Price"].median())
print("After fillna(median):")
print(df_filled)

print("\nPart 5: Min-Max Normalization")
prices = df_filled["Price"]
min_max = (prices - prices.min()) / (prices.max() - prices.min())
df_filled["Price_MinMax"] = min_max
print(df_filled[["Product", "Price", "Price_MinMax"]])

print("\nPart 6: Z-score Standardization")
df_filled["Price_Zscore"] = zscore(df_filled["Price"])
print(df_filled[["Product", "Price", "Price_Zscore"]])

print("\nPart 7: Outlier Detection - IQR Method")
Q1 = df_filled["Price"].quantile(0.25)
Q3 = df_filled["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df_filled[(df_filled["Price"] < lower_bound) | (df_filled["Price"] > upper_bound)]
print("IQR bounds:", lower_bound, "-", upper_bound)
print("Outliers (IQR method):")
print(outliers_iqr)

print("\nPart 8: Outlier Detection - Z-score Method")
df_filled["abs_zscore"] = df_filled["Price_Zscore"].abs()
outliers_z = df_filled[df_filled["abs_zscore"] > 3]
print("Outliers (Z-score method, threshold=3):")
print(outliers_z[["Product", "Price", "Price_Zscore"]])
print("Note: with threshold=3 this small sample flags none;")
print("a threshold of 2 would flag:")
print(df_filled[df_filled["abs_zscore"] > 2][["Product", "Price", "Price_Zscore"]])