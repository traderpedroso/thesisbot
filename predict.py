from sklearn.preprocessing import MinMaxScaler
from collections import deque
from indicator import Indicators
from settings import seq_len, ej
import pandas as pd
import numpy as np

def preprocess_prediciton(iq,symbol,symbols,timeframe):
    Actives = symbols
    active = symbol
    main = pd.DataFrame()
    current = pd.DataFrame()
    for active in Actives:
        if active == symbol:
            main = ej.iq_get_fastdata(iq,symbol,timeframe).drop(columns = {'from','to'})
        else:
            current = ej.iq_get_fastdata(iq,symbol,timeframe)
            current = current.drop(columns = {'from','to','open','min','max'})
            current.columns = [f'close_{active}',f'volume_{active}']
            main = main.join(current)

    df = main

    """
    graphical analysis components
    """

    df.isnull().sum().sum() 
    df.fillna(method="ffill", inplace=True)
    df = df.loc[~df.index.duplicated(keep = 'first')]
    

    df = Indicators(df)



    #df = df.drop(columns = {'open','min','max'})

    df = df.dropna()
    df = df.fillna(method="ffill")
    df = df.dropna()

    df.sort_index(inplace = True)

    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)

    pred = pd.DataFrame(df_scaled,index = indexes)

    sequential_data = []
    prev_days = deque(maxlen = seq_len)

    for i in pred.iloc[len(pred) -seq_len :len(pred)   , :].values:
        prev_days.append([n for n in i[:]])
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days)])

    X = []

    for seq in sequential_data:
        X.append(seq)
        
        


    return np.array(X)

