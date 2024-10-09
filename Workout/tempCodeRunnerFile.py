
y_train = train_data['gold_pct_change']

X_test =  pd.DataFrame(test_data['DaysSinceStart']) 
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