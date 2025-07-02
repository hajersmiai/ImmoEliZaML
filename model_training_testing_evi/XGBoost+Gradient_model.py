
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


df = pd.read_csv("cleaned_data_after_imputation .csv")
postal_df = pd.read_csv("code-postaux-belge.csv", delimiter=";")

postal_df.columns = postal_df.columns.str.strip().str.lower().str.replace(" ", "_")
df.columns = df.columns.str.strip().str.lower()

#Thinking of restricting the prediction afetr excluding prices on the highest 1% spectrum 

price_99th = df['price'].quantile(0.99)
print(f"99th percentile of price: {price_99th:.2f}")

df = df[df['price'] <= price_99th]
print(df.shape)


print(df.columns.tolist())
print(postal_df.columns.tolist())


#enrichment with columns based on postcode ['code', 'localite', 'longitude', 'latitude']
postal_df = postal_df.rename(columns={"code": "postcode"})

postal_df = postal_df.drop_duplicates(subset="postcode")
postal_df = postal_df.groupby("postcode").first().reset_index()

columns_to_keep = ['postcode', 'longitude', 'latitude']
postal_df = postal_df[columns_to_keep]

df["postcode"] = df["postcode"].astype(str)
postal_df["postcode"] = postal_df["postcode"].astype(str)

df = df.merge(postal_df, how="left", on="postcode")
df = df.dropna(subset=['longitude', 'latitude']) #drops only one row


print(df.columns.tolist())
print("Dataset shape:", df.shape)


#Now that the database is ready we proceed to change all categorical to numericalcategorical_cols = df.select_dtypes(include='object').columns.tolist()
#Check first:

# categorical_cols = df.select_dtypes(include='object').columns.tolist()
# print("Categorical columns:", categorical_cols)

# for col in categorical_cols:
#     print(f"\nColumn: {col}")
#     print(f"Unique values ({df[col].nunique()}):")
#     print(df[col].value_counts())

# print(df.isna().sum())

#First Type to 0/1
df['type'] = df['type'].map({'HOUSE': 0, 'APARTMENT': 1})

#epc and building condition => Ordinal encoding
epc_map = {'missing': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['epcscore'] = df['epcscore'].map(epc_map)

condition_map = {
    'missing': 0,
    'TO_RESTORE': 1,
    'TO_BE_DONE_UP': 2,
    'TO_RENOVATE': 3,
    'JUST_RENOVATED': 4,
    'GOOD': 5,
    'AS_NEW': 6
}
df['buildingcondition'] = df['buildingcondition'].map(condition_map)

#province 

df = pd.get_dummies(df, columns=['province'], drop_first=True)

from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer(transformers=[
#     ('onehot', OneHotEncoder(drop='first'), ['province'])
# ], remainder='passthrough')

# X_transformed = ct.fit_transform(df)

#For localities they are so many is going to bring noice to the model, I checked with bar plot and most listing are in the top 50 localities.(Forest can handle even 100, but linear models no(30 is ok).
#Plus in this occasion it wasnt worth it from 50-100 top localities). I grouped the rest in Other

top_50 = df['locality'].value_counts().nlargest(50).index
df['locality_grouped'] = df['locality'].apply(lambda x: x if x in top_50 else 'Other')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
locality_encoded = ohe.fit_transform(df[['locality_grouped']])
locality_feature_names = ohe.get_feature_names_out(['locality_grouped'])
locality_df = pd.DataFrame(locality_encoded, columns=locality_feature_names, index=df.index)
df = pd.concat([df, locality_df], axis=1)

df.drop('locality_grouped', axis=1, inplace=True)
df.drop('locality', axis=1, inplace=True)


#For subtype I had the idea to use label in an order that has to do with price from apartment to castle, but this would cause major data leakage,
# I decided to go safer with the median surface and group them (this is kinda ok for Forest but not for linear regression).
# # Second try I want to use one hot encoder + linear regression? distance based models like linear regression and KNN sensitive to labels/enumaration

ordered_subtypes = df.groupby("subtype")["habitablesurface"].median().sort_values().index
subtype_mapping = {subtype: i for i, subtype in enumerate(ordered_subtypes)}
df["subtype_encoded"] = df["subtype"].map(subtype_mapping)
df.drop(columns=['subtype'], inplace=True) #drop the original column



df.drop(columns=['postcode'], inplace=True)


df.to_csv("processed_data_for_modeling.csv", index=False)
print("Final cleaned dataset saved as 'processed_data_for_modeling.csv'")


target_name = "price"

X = df.drop(columns=[target_name])
y = df[target_name]

# ========== 2. Split Data ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ========== 3. Evaluation Function ==========
def evaluate_model(name, model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    r2 = r2_score(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))  # Proper definition

    print(f"\n{name} Performance:")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")  # Now rmse is defined

    return y_pred, r2, mae, rmse


def plot_preds(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()

# ========== 4. Gradient Boosting ==========
gbr = GradientBoostingRegressor(random_state=42)
param_grid_gbr = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}
gbr_search = RandomizedSearchCV(
    gbr, param_distributions=param_grid_gbr, n_iter=15, cv=3,
    scoring='r2', verbose=1, random_state=42, n_jobs=-1
)
gbr_search.fit(X_train, y_train)
gbr_best = gbr_search.best_estimator_
print("✅ Best GBR Params:", gbr_search.best_params_)

# ========== 5. XGBoost ==========
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_search = RandomizedSearchCV(
    xgb, param_distributions=param_grid_xgb, n_iter=20, cv=3,
    scoring='r2', verbose=1, random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
print("✅ Best XGB Params:", xgb_search.best_params_)

# ========== 6. Evaluation ==========



print("\n[Gradient Boosting Results]")
y_pred_gbr_test, r2_gbr_test, mae_gbr_test, rmse_gbr_test = evaluate_model(
    "GBR - Test", gbr_best, X_train, y_train, X_test, y_test
)


# ========== 7. Overfitting Gap ==========
print(f"\nOverfitting Gap GBR: {r2_score(y_train, gbr_best.predict(X_train)) - r2_score(y_test, gbr_best.predict(X_test)):.3f}")

# ========== 8. Plots ==========
plot_preds(y_test, y_pred_gbr_test, "GBR: Predicted vs Actual")
