from ejtrader import indicators as TA
from candlestick import candlestick as cd

def Indicators(df):
    df = df.rename(columns = {'min':'low', 'max':'high'}) # raname DF columns

    # Keltner Channels
    keltner = TA.KC(df)
    df['KC_UPPER'] = keltner['KC_UPPER']
    df['KC_LOWER'] = keltner['KC_LOWER']

    bolinger = TA.BBANDS(df)
    df['BB_UPPER'] = bolinger['BB_UPPER']
    df['BB_MIDDLE'] = bolinger['BB_MIDDLE']
    df['BB_LOWER'] = bolinger['BB_LOWER']

    # # MACD
    # MACD = TA.MACD(df)
    # df['MACD'] = MACD['MACD']
    # df['SIGNAL'] = MACD['SIGNAL']


    # df['BearishHarami'] = cd.bearish_harami(df)
    # df['BullishHarami'] = cd.bullish_harami(df)

    # df['DarkCloudCover'] = cd.dark_cloud_cover(df)
    # # df['MorningStarDoji'] = cd.morning_star_doji(df) # bug
    # # df['ShootingStar'] = cd.shooting_star(df) # bug
    # df['DragonflyDoji'] = cd.dragonfly_doji(df)

    # df['BearishEngulfing'] = cd.bearish_engulfing(df)
    # df['BullishEngulfing'] = cd.bullish_engulfing(df)

    # df['HangingMan'] = cd.hanging_man(df)
    # df['MorningStar'] = cd.morning_star(df)

    # df['MorningStarDoji'] = cd.morning_star_doji(df)
    # df['PiercingPattern'] = cd.piercing_pattern(df)

    # df['RainDrop'] = cd.rain_drop(df)
    # df['RainDropDoji'] = cd.rain_drop_doji(df)

    # df['GravestoneDoji'] = cd.gravestone_doji(df)
    # df['ShootingStar'] = cd.shooting_star(df)

    # df['Hammer'] = cd.hammer(df)
    # df['doji'] = cd.doji(df)
    # df['Star'] = cd.star(df)





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

    ROC = TA.ROC(df,1)
    df['close'] = ROC


    # df['VPT'] = TA.VPT(df)
    # df['VWAP'] = TA.VWAP(df)




    # df = df.rename(columns = {'low':'min', 'high':'max'})
    # df = df.drop(columns = {'volume'})
    

    return df
