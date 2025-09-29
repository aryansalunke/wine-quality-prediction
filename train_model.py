import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb  # Import XGBoost

# Step 1: Load the Dataset
file_path = 'winequality-red.csv'  # Update with your dataset path
df = pd.read_csv(file_path)

# Step 2: Check for Missing Values
print("Missing values per column:")
print(df.isnull().sum())

# Step 3: Handle Missing Values (if any)
# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object]).columns

# Impute numeric columns with the mean strategy
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute categorical columns with the most frequent category strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Step 4: Encode Categorical Data
df['type'] = LabelEncoder().fit_transform(df['type'])  # Convert 'type' to numerical values

# Step 5: Prepare Features (X) and Target (y)
X = df.drop('quality', axis=1)  # Features (all columns except 'quality')
y = df['quality']  # Target variable ('quality')

# Step 6: Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Option 1: XGBoost Regressor (Replaces RandomForestRegressor)
xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Option 2: LightGBM Regressor
lgb_model = lgb.LGBMRegressor(n_estimators=200, random_state=42)

# Fit the model to the training data
lgb_model.fit(X_train, y_train)

# Step 8: Evaluate the Models
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

lgb_train_pred = lgb_model.predict(X_train)
lgb_test_pred = lgb_model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error) for XGBoost
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))

# Calculate RMSE (Root Mean Squared Error) for LightGBM
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
lgb_test_rmse = np.sqrt(mean_squared_error(y_test, lgb_test_pred))

print(f"XGBoost - Training RMSE: {xgb_train_rmse:.4f}, Test RMSE: {xgb_test_rmse:.4f}")
print(f"LightGBM - Training RMSE: {lgb_train_rmse:.4f}, Test RMSE: {lgb_test_rmse:.4f}")

# Step 9: Save the Best Model and Scaler (Choose the one with the best performance)
if xgb_test_rmse < lgb_test_rmse:
    joblib.dump(xgb_model, 'wine_quality_xgb_model.pkl')
    print("XGBoost model saved.")
else:
    joblib.dump(lgb_model, 'wine_quality_lgb_model.pkl')
    print("LightGBM model saved.")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")
