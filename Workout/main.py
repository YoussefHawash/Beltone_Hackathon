# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'Final.csv' with your actual dataset path in Colab)
gold_data = pd.read_csv('InputData\\final_updated.csv')

# Convert 'Date' column to datetime
gold_data['Date'] = pd.to_datetime(gold_data['Date'])




# Decompose the time series for each feature
X = gold_data.drop(['Date', 'pct_change','gold_prices'], axis=1)
for column in X.columns:
    decomposition = seasonal_decompose(X[column], model='additive', period=30, extrapolate_trend='freq')
    gold_data[f'{column}_trend'] = decomposition.trend
    gold_data[f'{column}_seasonal'] = decomposition.seasonal
    gold_data[f'{column}_residual'] = decomposition.resid

# Drop any NaN values introduced during decomposition
gold_data.dropna(inplace=True)

# Create features and target variable
X = gold_data.drop(['Date', 'pct_change','gold_prices'], axis=1)
Y = gold_data['pct_change']

# Create lagged features for XGBoost (shift by 1 day)
X_lagged = X.shift(1)
X_lagged = X_lagged.dropna()
Y = Y[X_lagged.index]


# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_lagged, Y, test_size=0.2, random_state=42)


scaler= StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

print(type(X_train))
X_train=X_train_scaled
# XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42)
xgb_model.fit(X_train, Y_train)

# XGBoost Predictions
xgb_preds = xgb_model.predict(X_test)


# Define the hyperparameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 6, 9, 12],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 1, 10]
}



# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid,
                                   n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train, Y_train)

# Get the best parameters and best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("Best parameters found: ", best_params)


# Make predictions using the tuned XGBoost model
best_preds = best_model.predict(X_test)

# Evaluate the model
r2 = r2_score(Y_test, best_preds)
rmse = np.sqrt(mean_squared_error(Y_test, best_preds))

print("R squared error : ", r2)
print("RMSE:", rmse)

# Plotting results
plt.plot(Y_test.values, color='blue', label='Actual Value')
plt.plot(best_preds, color='red', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
