import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("cleanned_data_after_imputation.csv")

# Make sure there's no '...' in data
df.replace('...', pd.NA, inplace=True)

#Replace -1 with na
df.replace(-1, np.nan, inplace=True)

#drop columns and rows with over 75% missingness
threshold_col = len(df) * 0.90
df.dropna(axis=1, thresh=threshold_col, inplace=True)

threshold_row = df.shape[1] * 0.90
df.dropna(axis=0, thresh=threshold_row, inplace=True)

# drop rows with no price
df.dropna(subset=['price'], inplace=True)

df.drop_duplicates


df = df[
    (df['habitableSurface'] >= 10) & (df['habitableSurface'] <= 500) &
    (df['price'] >= 50000) & (df['price'] <= 2_000_000) &
    (df['bathroomCount'] <= 10) &
    (df['bedroomCount'] <= 10)
]

print("Remaining columns:", df.columns.tolist())
print("Remaining rows:", len(df))

filtered_df = df.copy()

numerical_columns = ["bedroomCount","bathroomCount", "habitableSurface", "buildingConditionNormalize", "epcScoreNormalize", "postCode"]

numerical_data = filtered_df[numerical_columns]


target_name = "price"

# X = pd.get_dummies(df.drop(columns=[target_name]))
X = numerical_data
y= filtered_df[target_name]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#Normalizzation and cleaning
# print(df.shape)
# print(df.dtypes)

# print(df['type'].unique())
# print(df['type'].value_counts())
# df['type'].isna().sum()

# print(df['subtype'].unique())
# print(df['subtype'].value_counts())
# df['subtype'].isna().sum()

# print(df.head)


# SPlit to train and a temporary group which is gonna be split in validation adn testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) #70%

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 15%  (calidation group to tune the model and then 15% for testing

print(X_train.describe)
print(X['epcScoreNormalize'].isna().sum())
print(X['epcScoreNormalize'].notna().sum())


model = LinearRegression()

# 2. Fit (train) the model
model.fit(X_train, y_train)

# 3. Predict on validation set
y_pred = model.predict(X_val)

# 4. Evaluate
r2 = r2_score(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)

print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")


# model = KNeighborsRegressor(n_neighbors=50)
# model.fit(X, y)

# y_pred = model.predict(X)
# accuracy = model.score(X, y)
# model_name = model.__class__.__name__

# print(f"{model_name} accuracy: {accuracy:.2f}")