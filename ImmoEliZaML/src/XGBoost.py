import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("ImmoElizaML/ImmoEliZaML/data/cleaned_data_no_outliers.csvcd ")

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

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=100,
    verbose_eval=200
)

# Predict
y_pred = model.predict(dtest)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ XGBoost MAE: {mae:.2f} EUR")
print(f"✅ XGBoost RMSE: {rmse:.2f} EUR")
print(f"✅ XGBoost R^2: {r2:.4f}")
#  Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("XGBoost: Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()