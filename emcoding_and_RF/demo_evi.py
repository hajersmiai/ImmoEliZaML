import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

csv_path = Path(__file__).parent / "data" / "immoweb_real_estate.csv"

try:
    df = pd.read_csv(csv_path)
except pd.errors.ParserError as e:
    print("❌ ParserError occurred:", e)
    # Try again by skipping bad lines
    df = pd.read_csv(csv_path, on_bad_lines='skip')

original_columns = df.columns.tolist()
print("Raw shape:", df.shape)

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "id", "url"], inplace=True, errors="ignore")


# Define column types
binary_cols = [
    "hasAttic", "hasBasement", "hasDressingRoom", "hasDiningRoom",
    "hasLift", "hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels",
    "hasLivingRoom", "hasBalcony", "hasGarden", "hasAirConditioning",
    "hasArmoredDoor", "hasVisiophone", "hasOffice", "hasSwimmingPool",
    "hasFireplace", "hasTerrace", "accessibleDisabledPeople"
]

multi_cat_cols = [
    "type", "subtype", "province", "buildingCondition", "floodZoneType",
    "heatingType", "kitchenType", "gardenOrientation", "terraceOrientation", "epcScore"
]

# Fill missing categorical values with 'UNKNOWN' and clean formatting
for col in multi_cat_cols:
    df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

for col in binary_cols:
    if col in df.columns:
        print(f"{col} - dtype: {df[col].dtype}, NaNs: {df[col].isna().sum()}")

# Show all columns with at least one missing value
missing_cols = df.isnull().sum()
missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)

print("Columns with missing values:\n")
print(missing_cols)

drop_cols = [
    "monthlyCost",           
    "hasBalcony",            
    "hasAirConditioning",    
    "hasDressingRoom",       
    "hasThermicPanels",      
    "hasArmoredDoor",        
    "hasHeatPump",           
    "hasPhotovoltaicPanels"  
]
df.drop(columns=drop_cols, inplace=True)

# Drop columns with extreme missing values 
df.drop(columns=["hasOffice", "hasSwimmingPool", "hasFireplace", "accessibleDisabledPeople"], inplace=True)

# Fill only garden and parking related missing values
df["hasGarden"] = df["hasGarden"].fillna(False).astype(int)
df["gardenSurface"] = df["gardenSurface"].fillna(0)

df["parkingCountOutdoor"] = df["parkingCountOutdoor"].fillna(0)
df["parkingCountIndoor"] = df["parkingCountIndoor"].fillna(0)
print("Lift missing %:", df['hasLift'].isna().mean())
print("Facade missing %:", df['streetFacadeWidth'].isna().mean())


df["locality_encoded"] = df["locality"].astype("category").cat.codes
df.drop(columns=["locality"], inplace=True)


# Encode binary columns: map True/False, Yes/No to 1/0

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"True": 1, "False": 0, "Yes": 1, "No": 0}).astype("Int64")


# Encode multi-category columns into numerical values
for col in multi_cat_cols:
    df[col + "_encoded"] = df[col].astype("category").cat.codes

# Optionally print mapping before dropping (for debugging or feature understanding)
for col in multi_cat_cols:
    print(f"\n{col} mapping:")
    print(dict(enumerate(df[col].astype("category").cat.categories)))

# Drop original multi-category columns
df.drop(columns=multi_cat_cols, inplace=True)


# Check if any non-numeric columns remain
non_numeric = df.select_dtypes(include=["object", "category"]).columns.tolist()
print("Non-numeric columns still remaining:", non_numeric)

# Ensure all data is numeric
df = df.select_dtypes(include=["number"])
print(df.dtypes[df.dtypes != "float64"])

#Check how many missing values are in the key binary columns:
binary_cols_to_check = [
    'hasDiningRoom', 'hasLift', 'hasLivingRoom', 'hasVisiophone'
]
missing_binary = df[binary_cols_to_check].isna().sum()
print("Still missing (binary columns):\n", missing_binary)

#Calculate the % of missing values:
total_rows = len(df)
missing_binary_pct = (missing_binary / total_rows) * 100
print("Missing %:\n", missing_binary_pct)

# Drop completely missing columns
df.drop(columns=['hasDiningRoom', 'hasLift', 'hasLivingRoom', 'hasVisiophone'], inplace=True)

# Drop columns with more than 70% missing values
threshold = 0.7
missing_ratio = df.isna().sum() / len(df)
cols_to_drop = missing_ratio[missing_ratio > threshold].index
df.drop(columns=cols_to_drop, inplace=True)

# Step 1: Threshold for missing values (e.g. 75%)
missing_threshold = 0.75
total_rows = len(df)

# Step 2: Drop columns with too many NaNs (but keep selected important ones)
keep_columns = {
    'postCode', 'province_encoded', 'buildingCondition_encoded', 'epcScore_encoded',
    'price'  # Always keep target
}
nan_percent = df.isna().sum() / total_rows
high_nan_cols = nan_percent[nan_percent > missing_threshold].index
cols_to_drop_nan = [col for col in high_nan_cols if col not in keep_columns]
df = df.drop(columns=cols_to_drop_nan)

# Step 3: Drop columns with very low correlation to price
correlation = df.corr(numeric_only=True)['price'].dropna()
low_correlation = correlation[correlation.abs() < 0.03].index
cols_to_drop_corr = [col for col in low_correlation if col not in keep_columns]
df = df.drop(columns=cols_to_drop_corr)

# Step 4: Print final shape and remaining columns
print(" Cleaned DataFrame shape:", df.shape)
print(" Remaining columns:\n", df.columns.tolist())

# Show missing value percentage for each column
missing_percentage = df.isna().mean() * 100
print("\n Missing values percentage per column:\n")
print(missing_percentage.sort_values(ascending=False).round(2))

missing_percentage.to_csv("missing_report.csv")

# Drop high-missing columns (optional)
# ⚠️ Dropped columns due to high missing values (even though correlated with price):
# - terraceSurface (64.4% missing)
# - livingRoomSurface (64.0% missing)
# - landSurface (50.8% missing)
# While these features might be relevant to price prediction, 
# the amount of missing data made them unreliable without imputation.
# To keep the model clean and robust, we excluded them from this first baseline.




high_missing = ['terraceSurface', 'livingRoomSurface', 'landSurface']

df_cleaned = df.drop(columns=high_missing)

# Start cleaning rows from the already column-cleaned DataFrame
df_cleaned = df_cleaned.dropna(subset=['price'])

important_cols = ['bedroomCount', 'bathroomCount']
df_cleaned = df_cleaned.dropna(subset=important_cols)

row_nan_ratio = df_cleaned.isna().mean(axis=1)
df_cleaned = df_cleaned[row_nan_ratio < 0.3]

df_cleaned = df_cleaned[df_cleaned['habitableSurface'] > 0]

df_cleaned.reset_index(drop=True, inplace=True)

print("✅ Final cleaned shape:", df_cleaned.shape)

# Step: Compare correlation with price
correlations = df_cleaned.corr(numeric_only=True)['price'].sort_values(ascending=False)
print("\n🔍 Top correlations with price:\n")
print(correlations.head(10))

print("\n Lowest correlations with price:\n")
print(correlations.tail(10))


# Drop rows with missing 'price' or critical columns
df_cleaned = df.dropna(subset=['price', 'bedroomCount', 'bathroomCount'])

from pathlib import Path

# 1. Drop rows missing key columns
df_cleaned = df.dropna(subset=['price', 'bedroomCount', 'bathroomCount'])

# 2. Remove top 1% most expensive
price_threshold = df_cleaned['price'].quantile(0.99)
df_filtered = df_cleaned[df_cleaned['price'] <= price_threshold]

# 3. Save filtered dataset
output_path = Path(__file__).parent / "data" / "cleaned_immoweb_filtered.csv"
df_filtered.to_csv(output_path, index=False)
#  Columns:
# - bedroomCount
# - bathroomCount
# - postCode
# - habitableSurface
# - buildingConstructionYear
# - facedeCount
# - landSurface
# - livingRoomSurface
# - gardenSurface
# - toiletCount
# - terraceSurface
# - price
# - type_encoded
# - subtype_encoded
# - province_encoded
# - buildingCondition_encoded
# - floodZoneType_encoded
# - heatingType_encoded
# - kitchenType_encoded
# - epcScore_encoded

#New info to increase accuracy and tell a story

#Fiest price per m2
df["price_per_m2"] = df["price"] / df["habitableSurface"]
df = df[df["habitableSurface"] > 10]  # exclude unrealistic surfaces
df = df[~df["price_per_m2"].isna()]
df = df[df["price_per_m2"] != float("inf")]

#plot with location
df["price_per_m2"] = df["price"] / df["habitableSurface"]


# Optional: remove extreme outliers
df = df[(df["price_per_m2"] > 100) & (df["price_per_m2"] < 10000)]

print(df["price_per_m2"].describe())

df["price_per_m2"] = df["price"] / df["habitableSurface"]

# Only numeric columns for correlation
correlation_matrix = df.corr(numeric_only=True)

# Focus on correlation with price per m²
corr_with_price_m2 = correlation_matrix["price_per_m2"].drop("price_per_m2")


# Most positive and negative correlations
print("🔼 Top positive correlations:")
print(corr_with_price_m2.sort_values(ascending=False).head(10))

print("\n🔽 Top negative correlations:")
print(corr_with_price_m2.sort_values().head(10))

import seaborn as sns
import matplotlib.pyplot as plt

top_features = corr_with_price_m2.abs().sort_values(ascending=False).head(10).index.tolist()
top_features.append("price_per_m2")

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Combine positive + negative correlations for a single plot
# corr_series = df.corr(numeric_only=True)["price_per_m2"].drop("price_per_m2")
# corr_sorted = corr_series.sort_values()

# plt.figure(figsize=(10, 7))
# sns.barplot(x=corr_sorted.values, y=corr_sorted.index, palette="coolwarm")
# plt.title(" Correlation with Price per m²")
# plt.xlabel("Correlation Coefficient")
# plt.ylabel("Feature")
# plt.axvline(0, color='gray', linestyle='--')
# plt.tight_layout()
# plt.show()

# corr_sorted.to_csv("price_per_m2_correlations.csv")
# Calculate price per square meter if not already in the dataframe
if "price_per_m2" not in df_cleaned.columns:
    df_cleaned["price_per_m2"] = df_cleaned["price"] / df_cleaned["habitableSurface"]

print(df_cleaned["bedroomCount"].value_counts().sort_index())



df_subset = df_cleaned[df_cleaned["bedroomCount"].between(1, 6)]

sns.barplot(
    data=df_subset,
    x="bedroomCount",
    y="price_per_m2",
    estimator="median",
    ci=None,
    palette="Blues"
)
plt.title("Median Price per m² by Bedroom Count (1–6)")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Median Price per m² (€)")
plt.tight_layout()
plt.show()

avg_price = df.groupby("bedroomCount")["price"].mean()
bedroom_counts = avg_price.index
avg_values = avg_price.values

# Filter to exclude outliers (optional, e.g., > 6 bedrooms)
mask = bedroom_counts <= 6

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=bedroom_counts[mask], y=avg_values[mask], palette="Blues_d")
plt.title("Average Total Price vs Bedroom Count")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Total Price (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import numpy as np

# Compute deltas between successive average prices
avg_price_diff = avg_price.diff().fillna(0)

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_price.index, y=avg_price_diff.values, marker='o')
plt.title(" Jump in Price Between Bedroom Counts")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price Increase Compared to Previous (€)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Price per bedroom
price_per_bedroom = avg_price / avg_price.index

plt.figure(figsize=(10, 6))
sns.lineplot(x=price_per_bedroom.index, y=price_per_bedroom.values, marker='o')
plt.title("Average Price per Bedroom")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Price per Bedroom (€)")
plt.grid(True)
plt.tight_layout()
plt.show()


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_absolute_error

# # Features and target
# X = df_filtered.drop(columns=['price'])
# y = df_filtered['price']

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Predict & evaluate
# y_pred = model.predict(X_test)
# print("🎯 R² Score:", r2_score(y_test, y_pred))
# print("💸 MAE:", mean_absolute_error(y_test, y_pred))







#============================

# #======================================

# # Show top 5 unique values for each object column
# for col in df.select_dtypes(include="object"):
#     print(f"\n🔍 {col}:")
#     print(df[col].value_counts(dropna=False).head())



# # Display dataset shape: number of rows and columns
# # print("Dataset shape:", df.shape)

# # # Display column names
# # print("\nColumn names:")
# # print(df.columns.tolist())

# # # Display data types
# # print("\nData types:")
# # print(df.dtypes)

# # # Display the first 5 rows
# # df.head()

# #Remove duplicates
# df = df.drop_duplicates()
# print(f"Remaining rows after duplicate removal: {len(df)}")

# # Strip whitespace from column names and string values
# df.columns = df.columns.str.strip()
# for col in df.select_dtypes(include="object"):
#     df[col] = df[col].astype(str).str.strip()

# # # Replace missing values as NA
# # df.replace("", pd.NA, inplace=True)
# # df.replace(" ", pd.NA, inplace=True)

# for col in df.columns:
#     print(f"{col}: {df[col].astype(str).value_counts(dropna=False).head(5)}")


# # Drop rows where price is missing (either NaN or empty string or space)
# df = df[df["price"].notna() & (df["price"] != "") & (df["price"] != " ")]

# # Drop rows with less than 40% valid (non-null) data
# min_valid = int(len(df.columns) * 0.4)
# df = df.dropna(thresh=min_valid)


# # Drop columns with >50% missing values
# missing_percent = (df.isna().sum() / len(df)) * 100
# cols_dropped = missing_percent[missing_percent > 30].index

# df = df.drop(columns=cols_dropped)

# print("Dropped columns (missing > 30%):", cols_dropped.tolist())
# print("Remaining columns:", df.columns.tolist())
# print(f"Remaining columns: {df.shape[1]}")




