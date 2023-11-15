# praktika11111
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Lenovo/Documents/forbes_2640_billionaires.csv")

# Choose the "rank" column as the target variable
y = data["rank"]

# Choose features (columns) for X
X = data.drop(columns=["rank"])

# Select categorical columns for encoding
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Transform categorical features into binary (one-hot encoding)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Replace the original categorical columns with the encoded ones
X.drop(categorical_cols, axis=1, inplace=True)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
X = pd.concat([X, X_encoded_df], axis=1)

# Fill missing values in the data
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base models
xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()
lr_model = LinearRegression()

# Train base models
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Get predictions from base models on the test data
xgb_test_preds = xgb_model.predict(X_test)
rf_test_preds = rf_model.predict(X_test)
lr_test_preds = lr_model.predict(X_test)

# Create a new matrix with predictions from base models
stacking_X_test = np.column_stack((xgb_test_preds, rf_test_preds, lr_test_preds))

# Train the meta-model (linear regression) on the training data
meta_model = LinearRegression()
meta_model.fit(stacking_X_test, y_test)

# Get predictions from the meta-model on the test data
stacking_y_pred = meta_model.predict(stacking_X_test)

# Evaluate the quality of stacking
mse = mean_squared_error(y_test, stacking_y_pred)
print(f"Stacking Mean Squared Error: {mse}")

# Scatter plot with specified color (e.g., 'green')
plt.scatter(y_test, stacking_y_pred, alpha=0.5, color='red')
plt.title('Stacking Model: True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()
