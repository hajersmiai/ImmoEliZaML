# filename: evaluate_dataset_size_impact.py

"""
Evaluate the impact of dataset size on ML performance using LightGBM
for ImmoEliza real estate price prediction.

✅ Measures MAE, RMSE, R² for 1k, 3k, 5k, 10k, 20k, and full dataset.
✅ Uses consistent LightGBM hyperparameters for fair comparison.
✅ Saves results CSV and shows plots for clear analysis.

"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("ImmoElizaML/ImmoEliZaML/data/cleaned_data_no_outliers.csv")

# Define feature set
features = [
    "bedroomCount", "bathroomCount", "habitableSurface", "roomCount",
    "buildingConstructionYear", "facedeCount", "floorCount", "toiletCount","postalCode",
    "diningRoomSurface", "kitchenSurface", "terraceSurface", "livingRoomSurface",
    "landSurface", "gardenSurface", "parkingCountIndoor", "parkingCountOutdoor",
    "buildingCondition_enc", "epcScore_enc", "heatingType_enc", "floodZoneType_enc", 
    "kitchenType_enc","subtype_enc","type_enc","gardenOrientation_enc","terraceOrientation_enc"
]
target = "price"
features = [col for col in features if col in df.columns]

X_full = df[features]
y_full = df[target]

# Sample sizes for testing
sample_sizes = [1000, 3000, 5000, 10000, 20000, len(df)]
results = []

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 8,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

for size in sample_sizes:
    print(f"\n🚀 Training with sample size: {size}")
    X_sample = X_full.sample(n=size, random_state=42)
    y_sample = y_full.loc[X_sample.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=200)
        ]
    )

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Sample Size": size,
        "MAE (EUR)": round(mae, 2),
        "RMSE (EUR)": round(rmse, 2),
        "R² Score": round(r2, 4),
        "Best Iteration": gbm.best_iteration
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\n✅ Results:\n", results_df)

# Save results
results_df.to_csv("ImmoElizaML/ImmoEliZaML/data/size_vs_performance_lightgbm.csv", index=False)
print("\n📁 Results saved to ImmoElizaML/ImmoEliZaML/data/size_vs_performance_lightgbm.csv")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results_df["Sample Size"], results_df["MAE (EUR)"], marker='o', label="MAE")
plt.plot(results_df["Sample Size"], results_df["RMSE (EUR)"], marker='s', label="RMSE")
plt.title("Impact of Dataset Size on MAE and RMSE (LightGBM - ImmoEliza)")
plt.xlabel("Sample Size")
plt.ylabel("Error (EUR)")
plt.legend()
plt.grid(True)
plt.savefig("ImmoElizaML/ImmoEliZaML/data/size_vs_performance_lightgbm.png", dpi=300)
plt.show()
print("📊 Graph saved to outputs/size_vs_performance_lightgbm.png")
