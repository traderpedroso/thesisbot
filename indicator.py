from ejtrader import indicators as TA
from candlestick import candlestick as cd
import numpy as np

def Indicators(df):
    df = df.rename(columns = {'min':'low', 'max':'high'}) # raname DF columns

    df["volatilidade"] = df["close"].rolling(window=45).std() * np.sqrt(45)
    #simple moving avarage
    df['SMA_20'] = TA.SSMA(df,20)
    df['SMA_50'] = TA.SSMA(df,50)

    # exponential moving average
    df['EMA_20'] = TA.EMA(df,20, adjust=False)
    df['EMA_50'] = TA.EMA(df,50, adjust=False)


     #Stochastic Oscillator
    df['%K'] = TA.STOCH(df, 14)
    df['%D'] = TA.STOCHD(df, 14)

    #Relative Strenght Index 'RSI'
    df['rsi'] = TA.RSI(df, 14)
    df['cci'] = TA.CCI(df,period=14)

    df['ROC'] = TA.ROC(df,1)

    # THIS IS THE TARGET "ALVO"
    df['GOAL'] = TA.ROC(df,1)


    # # df['VPT'] = TA.VPT(df)
    # df['VWAP'] = TA.VWAP(df)
    # # Keltner Channels
    # keltner = TA.KC(df)
    # df['KC_UPPER'] = keltner['KC_UPPER']
    # df['KC_LOWER'] = keltner['KC_LOWER']

    # bolinger = TA.BBANDS(df)
    # df['BB_UPPER'] = bolinger['BB_UPPER']
    # df['BB_MIDDLE'] = bolinger['BB_MIDDLE']
    # df['BB_LOWER'] = bolinger['BB_LOWER']

    df = df.drop(columns = {'open','high','low'})
    

    
    # print(df)
    return df
