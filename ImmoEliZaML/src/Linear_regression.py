import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1️⃣ Load cleaned data
data = pd.read_csv("ImmoEliZaML/data/cleaned_data_no_outliers.csv")
df = pd.DataFrame(data)

# 2️⃣ Select features and target
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

# 3️⃣ Split data
X = df[features]
y = df[target]
###############################################################################################################
# y = df[target] :                                     y = np.log(df[target]):                                #
#         MAE: 99663.61 EUR                                      MAE: 0.31 EUR                                #
#         R^2 Score: 0.3360                                      R^2 Score: 0.3422                            #
#         RMSE: 131978.19 EUR                                    RMSE: 0.40 EUR                               #                                                                           #
#                                                                                                             #
###############################################################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Create pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# 5️⃣ Train pipeline
pipeline.fit(X_train, y_train)

# 6️⃣ Predict
y_pred = pipeline.predict(X_test)

# 7️⃣ Evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(mean_absolute_error(y_test, y_pred))
print(f"MAE: {mae:.2f} EUR")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} EUR")

# 8️⃣ Plot predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Actual vs Predicted Prices (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Summary of current limitations

✅ LinearRegression is limited:

Unable to model complex nonlinearities between price and features.

The very architecture of the real estate market (location, rarity, feature interactions) is not linear.

✅ The log helps stabilize variance, but is not sufficient on its own.

"""