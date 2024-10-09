import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load your data
data = pd.read_csv('InputData\gold_prices.csv', parse_dates=['Date'])
prec = pd.read_csv('InputData\\target_gold.csv', parse_dates=['Date'])



# Step 2: Convert Dates to Numeric (days since the first date)
data['DaysSinceStart'] = (data['Date'] - data['Date'].min()).dt.days
print (data['DaysSinceStart'])
# Calculate percentage price change of gold

data['gold_pct_change'] = data['gold_prices'].pct_change() * 100
data['gold_pct_change'] .fillna(0, inplace=True)



# Split data into training (2020-2022) and testing (2023)
train_data = data[data['Date'] < '2023-01-01']
test_data = data[data['Date'] >= '2023-01-01']
# X_train = pd.DataFrame(train_data['DaysSinceStart']) 
X_train = train_data.drop(['gold_pct_change', 'Date','DaysSinceStart'], axis=1)

y_train = train_data['gold_pct_change']

# X_test =  pd.DataFrame(test_data['DaysSinceStart']) 
X_test = test_data.drop(['gold_pct_change', 'Date','DaysSinceStart'], axis=1)

y_test = test_data['gold_pct_change']


# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# Fit the model on the training data
model.fit(X_train,y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

# Initialize GridSearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')
# Predict the gold price change for 1/1/2024
X_new = test_data.iloc[-1].drop(['gold_pct_change', 'Date'])
gold_change_1_1_2024 = model.predict([X_new])
print(f'Predicted gold price change for 1/1/2024: {gold_change_1_1_2024[0]:.2f}%')