from settings import (
    iq, ej, gale_multiply, 
    gale_seq, symbol, symbols, 
    timeframe, seq_len, contract,
     min_balance, min_payout, min_prob,
     expire_time
     )
from predict import preprocess_prediciton

from model import train_data
import tensorflow.compat.v2 as tf
import datetime
import time
import sys



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
    forex_open_close = ['4','12','5','13','10','18','2','20']
    
    for times_market in forex_open_close:
        stoptime = times_market

    if str(hour) == stoptime and minutes >= 40:
        return True
    return False

while(1):
    hour = datetime.datetime.now().hour 
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
    if ej.iq_get_remaning(iq,timeframe) - 3 == ej.timeframe_to_sec(timeframe): 
        time_taker = time.time()
        pred = preprocess_prediciton(iq,symbol,symbols,timeframe)             
        pred = pred.reshape(1,seq_len,pred.shape[3])     
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
        # print ("Winning Rate : {:.2f}%".format(percentage(win_count,buy_count+sell_count))+'\n'+"Trade N°: "+str(sell_count+buy_count)+'\n')
        print ("Winning Rate: {:.2f}%".format(percentage(win_count,buy_count+sell_count-Tiedtrade))+'\n'+"Trade N°: "+str(sell_count+buy_count)+'\n')
        
       

    
        if result[0][0] > result[0][1] and result[0][0] > min_prob and ej.iq_get_payout(iq,symbol) >=min_payout and balance > min_balance:
            print("PUT")
            id = ej.iq_sell_binary(iq,contract,symbol,expire_time)
            sell_count += 1
            predict_count = predict_count + 1
            trade = True
        elif result[0][1] > result[0][0] and result[0][1] > min_prob and ej.iq_get_payout(iq,symbol) >=min_payout and balance > min_balance:
            print("CALL")
            id = ej.iq_buy_binary(iq,contract,symbol,expire_time) 
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
                countdown(180)
                predict_count = 10
                  
