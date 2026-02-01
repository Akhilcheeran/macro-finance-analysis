
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

START = "2015-01-01"

# -------------------------
# 1) Market data (S&P 500)
# -------------------------
sp500_raw = yf.download("^GSPC", start=START, progress=False)

# Month-end close (single clean column)
# -------------------------
# 1) Market data (S&P 500)
# -------------------------
sp500_raw = yf.download("^GSPC", start=START, progress=False)

# Make sure we extract a Series of closes (not a DataFrame)
close = sp500_raw["Close"]
if isinstance(close, pd.DataFrame):
    # If it's a DataFrame (e.g., multiindex/columns), take the first column
    close = close.iloc[:, 0]

# Month-end close (as DataFrame with a clean name)
sp500_m = close.resample("ME").last().to_frame()
sp500_m.columns = ["sp500_close"]

# Monthly returns
sp500_m["sp500_ret_m"] = sp500_m["sp500_close"].pct_change(fill_method=None)

# -------------------------
# 2) Macro data (FRED)
# -------------------------
cpi = pdr.DataReader("CPIAUCSL", "fred", START).rename(columns={"CPIAUCSL": "cpi"})
unrate = pdr.DataReader("UNRATE", "fred", START).rename(columns={"UNRATE": "unrate"})
fedfunds = pdr.DataReader("FEDFUNDS", "fred", START).rename(columns={"FEDFUNDS": "fedfunds"})

# Convert macro series to month-end to match sp500_m
cpi = cpi.resample("ME").last()
unrate = unrate.resample("ME").last()
fedfunds = fedfunds.resample("ME").last()

# CPI YoY inflation %
cpi["infl_yoy"] = cpi["cpi"].pct_change(12, fill_method=None) * 100

# -------------------------
# 3) Merge (month-end aligned)
# -------------------------
df = sp500_m.join([cpi, unrate, fedfunds], how="inner").dropna()

print("Merged dataset shape:", df.shape)
print(df.head())

# -------------------------
# 4) Plots
# -------------------------
df["sp500_close"].plot(title="S&P 500 (Month-end Close)")
plt.show()

df["infl_yoy"].plot(title="Inflation (CPI YoY %)")
plt.show()

df["unrate"].plot(title="Unemployment Rate (%)")
plt.show()

df["fedfunds"].plot(title="Fed Funds Rate (%)")
plt.show()

# Save for next steps
df.to_csv("merged_monthly_macro_market.csv")
print("Saved: merged_monthly_macro_market.csv")

# =========================
# 6) CORRELATION ANALYSIS
# =========================
corr_cols = ["sp500_ret_m", "infl_yoy", "unrate", "fedfunds"]
corr_matrix = df[corr_cols].corr()

print("\nCorrelation matrix:")
print(corr_matrix)
# =========================
# 7) REGRESSION ANALYSIS
# =========================
import statsmodels.api as sm

# Define dependent and independent variables
y = df["sp500_ret_m"]
X = df[["infl_yoy", "unrate", "fedfunds"]]

# Add constant (intercept)
X = sm.add_constant(X)

# Fit OLS regression
model = sm.OLS(y, X).fit()

print("\nOLS Regression Results:")
print(model.summary())

# =========================
# 8) REGIME SPLIT: PRE vs POST 2022
# =========================

import statsmodels.api as sm

# Split the dataset
df_pre = df[df.index < "2022-01-01"]
df_post = df[df.index >= "2022-01-01"]

def run_regression(data, label):
    y = data["sp500_ret_m"]
    X = data[["infl_yoy", "unrate", "fedfunds"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(f"\n--- Regression results: {label} ---")
    print(model.summary())

# Run regressions
run_regression(df_pre, "PRE-2022")
run_regression(df_post, "POST-2022")