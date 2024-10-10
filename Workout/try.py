import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=500, stop=1500, num=6)],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}
# Load your dataset (replace 'Final.csv' with your actual dataset)
gold_data = pd.read_csv('InputData\Final.csv')
# Convert 'Date' column to datetime
print (gold_data)
gold_data['Date'] = pd.to_datetime(gold_data['Date'])


X = gold_data.drop(['Date','gold_prices'], axis=1)
for column in X.columns:
    decomposition = seasonal_decompose(X[column], model='additive', period=30, extrapolate_trend='freq')
    # Add decomposed components to a new dataframe
    gold_data[f'{column}_trend'] = decomposition.trend
    gold_data[f'{column}_seasonal'] = decomposition.seasonal
    gold_data[f'{column}_residual'] = decomposition.resid

# Drop any NaN values introduced during decomposition
gold_data.dropna(inplace=True)

# Drop 'Date' from the features and target variable
X = gold_data.drop(['Date','gold_prices'], axis=1)
Y = gold_data['gold_prices']

# Create lagged features (e.g., shift by 1 day)
X_lagged = X.shift(1)
X_lagged = X_lagged.dropna()  # Drop the first row, since it will have NaN values after shifting

# Update the target variable accordingly
Y = Y[X_lagged.index]


X_train, X_test, Y_train, Y_test = train_test_split(X_lagged, Y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor to predict gold prices
gold_price_model = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01, random_state=42)
gold_price_model.fit(X_train, Y_train)

random_search = RandomizedSearchCV(estimator=gold_price_model, param_distributions=param_dist, 
                                   n_iter=100, scoring='neg_mean_squared_error', 
                                   cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the model with Random Search
random_search.fit(X_train, Y_train)

# Best hyperparameters
print("Best hyperparameters:", random_search.best_params_)

# Predict using the best model
best_model = random_search.best_estimator_
Preds = best_model.predict(X_test)

# Evaluate performance
rmse = mean_squared_error(Y_test, predictions, squared=False)
print("RMSE with Random Search:", rmse)
# Preds = gold_price_model.predict(X_test)




# R squared error
error_score = metrics.r2_score(Y_test, Preds)
print("R squared error : ", error_score)
rmse =mean_squared_error(Y_test, Preds, squared=False)
print(rmse)

NY_test = list(Y_test)
plt.plot(NY_test, color='blue', label = 'Actual Value')
plt.plot(Preds, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()