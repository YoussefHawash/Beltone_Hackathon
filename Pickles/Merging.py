from sklearn.ensemble import VotingRegressor

def voting(firsmodel,secondmodel,x,y):
    # Combine predictions using Voting Regressor
    voting_model = VotingRegressor(estimators=[('xgb', firsmodel), ('lstm', secondmodel)])
    voting_model.fit(x, y)
    return voting_model

