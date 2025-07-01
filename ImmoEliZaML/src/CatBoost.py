import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ImmoEliZaML/data/cleaned_data_no_outliers.csv")

features = [
    "bedroomCount", "bathroomCount", "habitableSurface", "roomCount",
    "buildingConstructionYear", "facedeCount", "floorCount", "toiletCount",
    "diningRoomSurface", "kitchenSurface", "terraceSurface", "livingRoomSurface",
    "landSurface", "gardenSurface", "parkingCountIndoor", "parkingCountOutdoor",
    "buildingCondition_enc", "epcScore_enc", "heatingType_enc", "floodZoneType_enc", "kitchenType_enc"
]
target = "price"
features = [col for col in features if col in df.columns]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize CatBoost
model = CatBoostRegressor(
    iterations=2000,            # CatBoostRegressor with:2000 iterations max.
    learning_rate=0.02,         # Low learning_rate for stable accuracy.
    depth=8,                    # Depth=8 ➔ Tree depth.
    loss_function='RMSE',       # Use RMSE as loss.
    early_stopping_rounds=100,  # Early stopping after 100 iterations without improvement.
    verbose=200,                # Displays the log every 200 iterations.
    random_seed=42              # Random_seed for reproducibility.
)

# Fit model
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ CatBoost MAE: {mae:.2f} EUR")
print(f"✅ CatBoost RMSE: {rmse:.2f} EUR")
print(f"✅ CatBoost R^2: {r2:.4f}")

#  Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("CatBoost: Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()