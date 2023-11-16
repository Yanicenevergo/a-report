# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:16:26 2023

@author: OUYANG XI
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


def predict(demand, gdp, cpi, am, model, scaler1, scaler2):
    sample=np.array([demand, gdp, cpi, am]) #list of inputs
    sample=scaler1.transform(sample.reshape(1,-1)) #feature transform
    pred=model.predict(sample)
    pred=scaler2.inverse_transform(pred.reshape(1,-1)).squeeze()
    return pred

def main(year):
    
    if year<2017 or year>2030:
        raise ValueError('year must be between 2017 and 2030')
    
    idx = year-2017
    
    df=pd.read_excel('data.xlsx')
    df = df[['gdp','cpi','am','demand_t','demand_t+1','demand_t-1']]
    df.dropna(axis=0, how='all', inplace=True)

    #first order difference
    df['gdp_diff']=df['gdp'].diff()
    df['cpi_diff']=df['cpi'].diff()
    df['am_diff']=df['am'].diff()
    df['demand_diff']=df['demand_t']-df['demand_t-1']
    df['demand_pred']=df['demand_t+1']-df['demand_t']
    
    data_X = df[['demand_diff','gdp_diff','cpi_diff','am_diff']].iloc[1:,]
    data_y = df['demand_pred'][1:] #label
    demands_abs = list(df['demand_t+1'].dropna())

    X_train = data_X.iloc[:31,:] #train data
    y_train = data_y[:31]
    
    #normalization
    scaler1=StandardScaler() #for feature
    scaler2=StandardScaler() #for label

    X_train = scaler1.fit_transform(X_train) 
    X_test = scaler1.transform(data_X.iloc[31,:].values.reshape(1,-1)).squeeze()

    y_train = scaler2.fit_transform(y_train.values.reshape(-1,1)).squeeze()
    
    #model with grid search
    model_base=MLPRegressor(random_state=11)
    model_grid = {'hidden_layer_sizes':[(5,),(10,),(5,5),(10,5),(10,10)],
                'activation':['tanh', 'relu','sigmoid'],
                'alpha':[0.001,0.01,0.1,1],
                'learning_rate_init':[0.001,0.01,0.1]}
    
    model = GridSearchCV(model_base, model_grid)
    #model training
    model.fit(X_train,y_train) 
    
    #prediction
    inc_pred = model.predict(X_test.reshape(1,-1))
    inc_pred = scaler2.inverse_transform(inc_pred.reshape(1,-1)).squeeze()
    pred = demands_abs[-2]+inc_pred
    gdp=data_X['gdp_diff'][32:].values
    cpi=data_X['cpi_diff'][32:].values
    am=data_X['am_diff'][32:].values
    inc_pred = demands_abs[-1]-demands_abs[-2]
    inc_acc = 0 #accumulate difference
    for i in range(idx):
        inc_pred = predict(inc_pred, gdp[i], cpi[i], am[i], model, scaler1, scaler2)
        inc_acc += inc_pred
    pred = demands_abs[-1]+inc_acc
    print(f'Prediction of {year}: ', pred)


if __name__ == '__main__':
    main(2030)