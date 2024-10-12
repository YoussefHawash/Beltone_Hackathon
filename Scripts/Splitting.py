from statsmodels.tsa.seasonal import seasonal_decompose

def split_data(All_data):

    # Split dataset
    train_set=All_data[All_data['Date'] <= '2022-12-31']
    test_set=All_data[All_data['Date'] > '2022-12-31']




    X_train = train_set.drop(['Date', 'gold_prices','pct_change'], axis=1)
    Y_train = train_set['pct_change']
    X_test = test_set.drop(['Date', 'gold_prices','pct_change'], axis=1)
    Y_test = test_set['pct_change']

    for column in X_train.columns:
        decomposition = seasonal_decompose(X_train[column], model='additive', period=30, extrapolate_trend='freq')
        X_train[f'{column}_trend'] = decomposition.trend
        X_train[f'{column}_seasonal'] = decomposition.seasonal
        X_train[f'{column}_residual'] = decomposition.resid

    for column in X_test.columns:
        decomposition = seasonal_decompose(X_test[column], model='additive', period=30, extrapolate_trend='freq')
        X_test[f'{column}_trend'] = decomposition.trend
        X_test[f'{column}_seasonal'] = decomposition.seasonal
        X_test[f'{column}_residual'] = decomposition.resid

 
    return X_train, X_test, Y_train, Y_test 