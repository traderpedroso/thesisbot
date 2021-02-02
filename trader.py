from settings import *
import ejtrader as ej
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from indicator import Indicators
from model import train_data
import tensorflow as tf
import datetime
import time
import sys



try:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
  # Memory growth must be set before GPUs have been initialized
  print(e)

def check_time(hour, minutes, time_now):
  if time_now.hour == hour and time_now.minute == minutes:
    return True
  return False






  





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
    prev_days = deque(maxlen = SEQ_LEN)

    for i in pred.iloc[len(pred) -SEQ_LEN :len(pred)   , :].values:
        prev_days.append([n for n in i[:]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days)])

    X = []

    for seq in sequential_data:
        X.append(seq)
        
        


    return np.array(X)




NAME = train_data(iq,symbol,symbols,timeframe) + '.h5'
model = tf.keras.models.load_model(f'models/{NAME}')

# define the countdown func. 
def countdown(t): 
    
    while t: 
        mins, secs = divmod(t, 60) 
        timer = '{:02d}:{:02d}'.format(mins, secs) 
        print(timer, end="\r") 
        time.sleep(1) 
        t -= 1
      
    print('Ready for the Nex Trade!!')
     
def percentage(entry1, entry2):
    try:
        return ( 100 * entry1 /entry2) 
    except ZeroDivisionError:
        return 0

min_contract = contract
win_count = 0 

sell_count = 0
buy_count = 0
Tiedtrade = 0
predict_count = 0
gale_count = 0 
bid = True
trade = True

def check_stop_time(hour,minutes):
    
    # BERLIN 05:00 - 13:00 / LONDON 06:00 - 14:00 / NEW YORK 11:00 - 19:00 / SYDNEY 19:00 - 03:00 / TOKYO 21:00 - 05:00
    forex_open_close = ['05','13','06','14','11','19','03','21']

    for times_market in forex_open_close:
        stoptime = times_market

    if str(hour) in stoptime and minutes >= 40:
        return True
    return False

while(1):
    hour = datetime.datetime.now().hour + 1
    minutes = datetime.datetime.now().minute
    if check_stop_time(hour,minutes):
        print('wating to pass opening market')
        countdown(2400)
        predict_count = 10
        
          
    t = 60
    if predict_count >= 10 and predict_count % 2 == 0:
        NAME = train_data(iq,symbol,symbols,timeframe) + '.h5'
        model = tf.keras.models.load_model(f'models/{NAME}')
        predict_count = 0
        print(ej.iq_get_remaning(iq,timeframe))
    if ej.iq_get_remaning(iq,timeframe) - 3 == ej.timeframe_to_sec(timeframe): 
        time_taker = time.time()
        pred = preprocess_prediciton(iq,symbol,symbols,timeframe)             
        pred = pred.reshape(1,SEQ_LEN,pred.shape[3])     
        result = model.predict(pred)
       
        print("probability of PUT: {:.2f}%".format(round(result[0][0],2)))
        print("probability of CALL: {:.2f}%".format(round(result[0][1],2)))
        print(f'Time taken : {int(time.time()-time_taker)} seconds')
        predict_count = predict_count + 1
        payout = ej.iq_get_payout(iq,symbol)
        balance = ej.iq_get_balance(iq)
        print(f'Simbol : {symbol}')
        print(f'Balance : {balance}')
        print("Payout: {:.2f}%".format(payout))
        print("BET: {:.2f}$".format(contract))
        print("Next Martingale: {:.2f}$".format(contract * round(gale_multiply/ej.iq_get_payout(iq,symbol),2)))
        print ("Winning Rate : {:.2f}%".format(percentage(win_count,buy_count+sell_count))+'\n'+"Trade N°: "+str(sell_count+buy_count)+'\n')
        print ("Winning Rate Tied Calculation : {:.2f}%".format(percentage(win_count,buy_count+sell_count-Tiedtrade))+'\n'+"Trade N°: "+str(sell_count+buy_count)+'\n')
        
       

    
        if result[0][0] > result[0][1] and result[0][0] > min_prob and ej.iq_get_payout(iq,symbol) >=min_payout and balance > min_balance:
            print("PUT")
            id = ej.iq_sell_binary(iq,contract,symbol,timeframe)
            sell_count += 1
            predict_count = predict_count + 1
            trade = True
        elif result[0][1] > result[0][0] and result[0][1] > min_prob and ej.iq_get_payout(iq,symbol) >=min_payout and balance > min_balance:
            print("CALL")
            id = ej.iq_buy_binary(iq,contract,symbol,timeframe) 
            buy_count += 1
            predict_count = predict_count + 1
            trade = True
        else:
            trade = False
            predict_count = predict_count + 1

            
        if trade:
            win = ej.iq_checkwin(iq,id)
            
            if win > 0:
                print("Win")
                gale_count = 0
                win_count += 1
                contract = min_contract
            elif win < 0:
                print("Loss")
                gale_count = gale_count + 1
                if gale_count >= gale_seq:
                    gale_count = 0
                    predict_count = 10
                    contract = min_contract  
                else:
                    contract = contract * round(gale_multiply/ej.iq_get_payout(iq,symbol),2)

            elif win == 0:
                print('Tied Wait for 10 minutes befor next Trade')
                Tiedtrade += 1
                countdown(600)
                predict_count = 10
                  
