import numpy as np
import pandas as pd
import os 
def Oil_Average(df):
    # df = pd.read_csv(os.path.join(input_path, 'crude_oil_prices.csv'))
    df.loc[df[df.columns[1]] < 0, df.columns[1]] =  df.loc[df[df.columns[1]] < 0, df.columns[2]]
    df['AVG'] =(df.iloc[:, 1] +df.iloc[:, 2])/2
    return df['AVG'] 
def EFFR(df):
    # df = pd.read_csv(os.path.join(input_path, 'crude_oil_prices.csv'))
    df.EFFR= pd.to_numeric(df.EFFR, errors='coerce').fillna(0).astype(float)
    return df.EFFR

def ECIR(df):
    return df.iloc[:, 2]

def Housing_index(df):
    return df.iloc[:, 1]

def Inflation_mom(df):
    return df.iloc[:, 1]
def Inflation_yoy(df):
    return df.iloc[:, 1]
def Intraday(df):
    for i in range(len(df['24K'])):
        item = df.loc[i,'Timestamp']
        df.loc[i,'Timestamp'] = item[:10]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    intraday_gold_last_per_day = df.groupby(df['Timestamp'].dt.date).last().reset_index(drop=True)
    return intraday_gold_last_per_day.iloc[:, 1]
def Sotcks(df):
    df.fillna(0, inplace=True)
    sum_of_volumes = df.iloc[:, 16:31].sum(axis=1)
    sum_of_volumexprices=pd.DataFrame()
    sum_of_volumexprices=df.iloc[:,1]*df.iloc[:,16]
    for i in range(2,15):
        sum_of_volumexprices+=df.iloc[:,i]*df.iloc[:,16+i]
    stonks=sum_of_volumexprices/sum_of_volumes
    return stonks
def Vix(df):
    vix_index=(df.iloc[:, 1]+df.iloc[:, 2]+df.iloc[:, 3]+df.iloc[:, 4])/4
    return vix_index
def Vxeem(df):
    vxeem_index=(df.iloc[:, 1]+df.iloc[:, 2]+df.iloc[:, 3]+df.iloc[:, 4])/4
    return vxeem_index
def pct_calc(df):
    dfc=df.copy()
    dfc['gold_prices_shifted'] = dfc['gold_prices'].shift(-1)
    dfc['pct_change'] = (dfc['gold_prices_shifted'] - dfc['gold_prices']) / dfc['gold_prices'] * 100
    dfc.dropna(subset=['pct_change'], inplace=True)
    return dfc['pct_change']
    

def CreateFinal(a):
    df=pd.DataFrame()
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(Oil_Average(a[0])), freq='D')
    df['gold_prices']=Intraday(a[9]) 
    df['pct_change']=pct_calc(df)
    df['Oil_AVG']=Oil_Average(a[0])
    df['EFFR']=EFFR(a[1])
    df['ECIR']=ECIR(a[2]) 
    df['Housing_index']=Housing_index(a[3]) 
    df['Inflation_mom']=Inflation_mom(a[4]) 
    df['Inflation_yoy']=Inflation_yoy(a[5]) 
    df['stocks']= Sotcks(a[6])
    df['Vix']=Vix(a[7]) 
    df['Vxeem']=Vxeem(a[8]) 
    return df
