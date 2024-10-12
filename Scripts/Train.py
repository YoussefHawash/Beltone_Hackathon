from Scripts import DataAnalysis, Splitting,Model, MergingModels,Train,Test
import joblib

def Train(crude_oil_prices,federal_rates,corridor_rates,housing_index,inflation_mom,inflation_yoy,stock_prices,vix_indices,vixeem_indices,gold_prices):
    gold_data=DataAnalysis.CreateFinal([crude_oil_prices,federal_rates,corridor_rates,housing_index,inflation_mom,inflation_yoy,stock_prices,vix_indices,vixeem_indices,gold_prices]) 
    gold_data.dropna(inplace=True)
    X_train, X_test, Y_train, Y_test = Splitting.split_data(gold_data)
    ## Those to turn off learning 
    first,second=Model.train(X_train,Y_train)
    voting_model= MergingModels.voting(first,second, X_train, Y_train)
    joblib.dump(voting_model, 'Pickles/voting_regressor_model.pkl')
    Test.Test(X_test,Y_test)

