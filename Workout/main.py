import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


# Load your dataset (replace 'Final.csv' with your actual dataset)
gold_data = pd.read_csv('InputData/NEW.csv')
# Convert 'Date' column to datetime
gold_data['Date'] = pd.to_datetime(gold_data['Date'])


X = gold_data.drop(['Date','GoldPrice'], axis=1)


# for column in X.columns:
#     decomposition = seasonal_decompose(X[column], model='additive', period=30, extrapolate_trend='freq')
#     # Add decomposed components to a new dataframe
#     gold_data[f'{column}_trend'] = decomposition.trend
#     gold_data[f'{column}_seasonal'] = decomposition.seasonal
#     gold_data[f'{column}_residual'] = decomposition.resid

# Drop any NaN values introduced during decomposition

decomposition = seasonal_decompose(X['gold_prices'], model='additive', period=30, extrapolate_trend='freq')
gold_data['gold_trend'] = decomposition.trend
gold_data['gold_seasonal'] = decomposition.seasonal
gold_data['gold_residual'] = decomposition.resid

decomposition = seasonal_decompose(X['Headline (m/m)'], model='additive', period=30, extrapolate_trend='freq')
gold_data['inflation_trend'] = decomposition.trend
gold_data['inflation_seasonal'] = decomposition.seasonal
gold_data['inflation_residual'] = decomposition.resid
gold_data.dropna(inplace=True)



# Drop 'Date' from the features and target variable
X = gold_data.drop(['Date'], axis=1)
Y = gold_data['gold_prices']

# Create lagged features (e.g., shift by 1 day)
X_lagged = X.shift(1)
X_lagged = X_lagged.dropna()  # Drop the first row, since it will have NaN values after shifting

# Update the target variable accordingly
Y = Y[X_lagged.index]


X_train, X_test, Y_train, Y_test = train_test_split(X_lagged, Y, test_size=0.2, random_state=42)


# Split the data into training and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_lagged, Y, test_size=0.2, random_state=42)

# Train Gradient Boosting models for each feature
models = {}
for feature in X.columns:
    print(f"Training model for feature: {feature}")

    # Create and train a Gradient Boosting Regressor model for each feature
    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
    model.fit(X_train, X_train[feature])

    # Store the model for future predictions
    models[feature] = model

# Now let's predict the features for the next day
last_row = X.iloc[-1, :]  # Get the last row of X (the most recent data point)
next_day_features = {}  # Dictionary to store next day predicted features

# Use the trained models to predict the next day's feature values
for feature, model in models.items():
    next_day_features[feature] = model.predict(last_row .values.reshape(1, -1))[0]  # Reshape for prediction


# ActualNExt= X.iloc[-99,:]
# print(gold_data.iloc[-99,1])
# print (ActualNExt)

# Convert the predicted features into a DataFrame (for easier use in the next model)
next_day_features_df = pd.DataFrame([next_day_features])
print (next_day_features_df)

# Train a Gradient Boosting Regressor to predict gold prices
gold_price_model = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01, random_state=42)
gold_price_model.fit(X_train, Y_train)

# Predict the gold price for the next day using the predicted features

next_day_gold_price = gold_price_model.predict(next_day_features_df)
testcases = gold_price_model.predict(X_test)
print(f"Predicted gold price for the next day: {next_day_gold_price[0]}")






# R squared error
error_score = metrics.r2_score(Y_test, testcases)
print("R squared error : ", error_score)
rmse =mean_squared_error(Y_test, testcases, squared=False)
print(rmse)
NY_test = list(Y_test)

plt.plot(NY_test, color='blue', label = 'Actual Value')
plt.plot(testcases, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()