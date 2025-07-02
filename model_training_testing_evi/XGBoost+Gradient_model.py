
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


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
data_columns = df.columns.drop([target_name], errors='ignore')


X = df[data_columns]
y = df[target_name]

# SPlit to train and a temporary group which is gonna be split in validation adn testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) #70%

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 15%  (calidation group to tune the model and then 15% for testing

print(X_train.describe())



from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"\n{name} Results:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    return model, r2, mae, rmse

# Gradient Boosting Regressor (Sklearn)

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=42)

param_grid_gbr = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

gbr_search = RandomizedSearchCV(
    gbr,
    param_distributions=param_grid_gbr,
    n_iter=15,
    cv=3,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

gbr_search.fit(X_train, y_train)
print("Best GBR Params:", gbr_search.best_params_)


# XGBoost

xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_grid_xgb,
    n_iter=20,
    cv=3,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
print("Best XGB Params:", xgb_search.best_params_)
import joblib



# Best models from search
xgb_best = xgb_search.best_estimator_
gbr_best = gbr_search.best_estimator_
# Evaluate best models on test set
evaluate_model("XGBoost (Test)", xgb_best, X_train, y_train, X_test, y_test)
evaluate_model("Gradient Boosting (Test)", gbr_best, X_train, y_train, X_test, y_test)


# Predictions
y_pred_xgb = xgb_best.predict(X_test)
y_pred_gbr = gbr_best.predict(X_test)

# Save the model you choose
joblib.dump(xgb_best, "xgb_model_price_predictor.joblib")

print("\n[Train vs Test Evaluation for XGBoost]")

# TRAIN performance
evaluate_model("XGBoost (Train)", xgb_best, X_train, y_train, X_train, y_train)

# TEST performance (again, just for side-by-side comparison)
evaluate_model("XGBoost (Test)", xgb_best, X_train, y_train, X_test, y_test)

train_r2 = r2_score(y_train, xgb_best.predict(X_train))
test_r2 = r2_score(y_test, xgb_best.predict(X_test))
print(f"\nOverfitting gap (Train R² - Test R²): {train_r2 - test_r2:.3f}")



def plot_preds(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()

# Plots
plot_preds(y_test, y_pred_xgb, "XGBoost Predictions vs Actual")
plot_preds(y_test, y_pred_gbr, "Gradient Boosting Predictions vs Actual")



