"""
Exploratory Data Analysis (EDA) using NumPy, Pandas & YData Profiling
Dataset: Retail Sales Dataset
"""

# ── 1. Install dependencies (run this once in your terminal) ──────────────────
# pip install numpy pandas ydata-profiling

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# ── 2. Generate a realistic Retail Sales Dataset ─────────────────────────────
np.random.seed(42)
n = 1000  # number of records

categories    = ["Electronics", "Clothing", "Groceries", "Furniture", "Sports"]
regions       = ["North", "South", "East", "West"]
payment_modes = ["Cash", "Credit Card", "UPI", "Net Banking"]

data = {
    "order_id":      np.arange(1001, 1001 + n),
    "customer_age":  np.random.randint(18, 70, n),
    "gender":        np.random.choice(["Male", "Female"], n),
    "category":      np.random.choice(categories, n),
    "quantity":      np.random.randint(1, 10, n),
    "unit_price":    np.round(np.random.uniform(10, 5000, n), 2),
    "discount_pct":  np.random.choice([0, 5, 10, 15, 20], n),
    "region":        np.random.choice(regions, n),
    "payment_mode":  np.random.choice(payment_modes, n),
    "rating":        np.round(np.random.uniform(1, 5, n), 1),
    "return_flag":   np.random.choice([0, 1], n, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)

# Derived columns
df["total_price"]    = np.round(df["quantity"] * df["unit_price"], 2)
df["discount_amt"]   = np.round(df["total_price"] * df["discount_pct"] / 100, 2)
df["final_amount"]   = np.round(df["total_price"] - df["discount_amt"], 2)
df["order_date"]     = pd.date_range(start="2023-01-01", periods=n, freq="8h")

# Introduce ~2% missing values to make the EDA more realistic
for col in ["customer_age", "rating", "discount_pct"]:
    mask = np.random.rand(n) < 0.02
    df.loc[mask, col] = np.nan

# Save raw dataset
df.to_csv("retail_sales.csv", index=False)
print("✅  Dataset saved → retail_sales.csv")
print(f"    Shape : {df.shape}")

# ── 3. Basic EDA with NumPy & Pandas ─────────────────────────────────────────
print("\n" + "="*55)
print("  BASIC EDA — NumPy & Pandas")
print("="*55)

# --- Shape & dtypes ---
print("\n📌 Shape:", df.shape)
print("\n📌 Data Types:\n", df.dtypes)

# --- Missing values ---
print("\n📌 Missing Values:\n", df.isnull().sum())

# --- Descriptive Statistics ---
print("\n📌 Descriptive Statistics (numeric):\n", df.describe().round(2))

# --- NumPy-based stats on key columns ---
num_cols = ["final_amount", "quantity", "unit_price", "customer_age", "rating"]
print("\n📌 NumPy Statistics:")
for col in num_cols:
    arr = df[col].dropna().values
    print(f"\n  [{col}]")
    print(f"    Mean     : {np.mean(arr):.2f}")
    print(f"    Median   : {np.median(arr):.2f}")
    print(f"    Std Dev  : {np.std(arr):.2f}")
    print(f"    Min/Max  : {np.min(arr):.2f} / {np.max(arr):.2f}")
    print(f"    25th pct : {np.percentile(arr, 25):.2f}")
    print(f"    75th pct : {np.percentile(arr, 75):.2f}")

# --- Categorical counts ---
print("\n📌 Category Distribution:\n", df["category"].value_counts())
print("\n📌 Region Distribution:\n",   df["region"].value_counts())
print("\n📌 Payment Mode:\n",          df["payment_mode"].value_counts())

# --- Correlation ---
print("\n📌 Correlation Matrix (numeric):\n",
      df[["final_amount", "quantity", "unit_price",
          "discount_pct", "rating"]].corr().round(3))

# --- Groupby insights ---
print("\n📌 Avg Final Amount by Category:\n",
      df.groupby("category")["final_amount"].mean().round(2).sort_values(ascending=False))

print("\n📌 Return Rate by Region:\n",
      df.groupby("region")["return_flag"].mean().round(3))

print("\n📌 Avg Rating by Gender:\n",
      df.groupby("gender")["rating"].mean().round(2))

# ── 4. YData Profiling Report ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  GENERATING YDATA PROFILING REPORT …")
print("="*55)

profile = ProfileReport(
    df,
    title="Retail Sales — EDA Report",
    explorative=True,          # enables correlations, interactions
    missing_diagrams=True,
    correlations={
        "pearson":  {"calculate": True},
        "spearman": {"calculate": True},
        "kendall":  {"calculate": False},
    },
)

report_path = "retail_sales_eda_report.html"
profile.to_file(report_path)
print(f"\n✅  YData Profiling report saved → {report_path}")
print("    Open it in any browser for an interactive deep-dive.\n")
