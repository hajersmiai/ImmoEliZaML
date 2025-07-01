import pandas as pd

# Load cleaned data
df = pd.read_csv("ImmoEliZaML/data/cleaned_data.csv")

# IQR filtering on 'price'
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Optionally clamp further
lower_bound = max(lower_bound, 50000)     # Minimum acceptable price
upper_bound = min(upper_bound, 5000000)   # Maximum acceptable price

# Filter out the outliers
df_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print(f"✅ Rows before filtering: {df.shape[0]}")
print(f"✅ Rows after filtering: {df_filtered.shape[0]}")
print(f"✅ Outliers removed: {df.shape[0] - df_filtered.shape[0]}")

# Optionally save cleaned data before log
df_filtered.to_csv("ImmoEliZaML/data/cleaned_data_no_outliers.csv", index=False)
print("[SUCCESS] Cleaned data without outliers saved.")
