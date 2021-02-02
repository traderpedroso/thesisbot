import ejtrader as ej
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


iq = ej.iq_login(email = 'user', password='senha', AccountType='DEMO') # REAL OR DEMO


symbol = 'EURUSD'       # SIMBOLO ATIVO
timeframe = "M1"
symbols = ['EURUSD']
contract = 2             # VALOR INICIAL DO TRADE
gale_seq = 15            # QUANTIDADE MAXIMA DE MARTINGALE
min_payout = 0.85        # MINIMO PAYOUT PARA TRADE
min_balance = 0          # VALOR MINIMO NA CONTA PARA ABRIR UM TRADE
min_prob = 0.60          # PORCENTAGEM MINIMA PARA ABIR UM TRADE EXEMPLO 50 QUALQUER LADO COM 51 SEGUE ESTE
gale_multiply = 2




# Hyperparameters
VALIDATION_TRAIN = True # True for cross validation  or False for Trading only
SEQ_LEN = 5 
NEURONS = 128
FUTURE_PERIOD_PREDICT = 1 
LEARNING_RATE = 0.001 
EPOCHS = 40
BATCH_SIZE = 32
EARLYSTOP = 20




