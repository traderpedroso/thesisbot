import ejtrader as ej
from dotenv import load_dotenv
load_dotenv()
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# para login add usuario e senha no arquivo .env
iq = ej.iq_login(email = os.getenv("USERNAME"), password=os.getenv("PASSWORD"), AccountType=os.getenv("ACCOUNT_TYPE")) # REAL OR DEMO


symbol = 'EURUSD'       
timeframe = "M1"   
contract = 5

expire_time = timeframe    
symbols = [symbol]

gale_multiply = 2       # FATOR DE MULTIPLICAÇÃO DE MARTINGALE
gale_seq = 5            # QUANTIDADE MAXIMA DE MARTINGALE
          
min_payout = 0.80      # MINIMO PAYOUT PARA ABRIR UMA ORDEM
min_balance =  0        # VALOR MINIMO NA CONTA PARA ABRIR UM TRADE
min_prob = 0.90         # % MINIMA DE PROBABILIDADE PARA ABIR UM TRADE 



# data parameters
seq_len = 5  
predict_period = 1 


# Hyperparameters
VALIDATION_TRAIN = True # True for cross validation  or False for Trading only
NEURONS = 25
LEARNING_RATE = 0.01 
EPOCHS = 40
BATCH_SIZE = 32
EARLYSTOP = 20




