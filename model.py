import numpy as np
import pandas as pd
import random
from collections import deque
# import tensorflow as tf

# mac configurations
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import time
import os
from ejtrader import iq_login, iq_get_data
from settings import seq_len, predict_period, LEARNING_RATE, EPOCHS, BATCH_SIZE, EARLYSTOP, VALIDATION_TRAIN, NEURONS, ismac

if ismac:
    from tensorflow.python.compiler.mlcompute import mlcompute
    mlcompute.set_mlc_device(device_name='cpu')


from indicator import Indicators

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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





def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future", 1)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)

    df = pd.DataFrame(df_scaled,index = indexes)

    sequential_data = []
    prev_days = deque(maxlen=seq_len)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)


    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]


    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y



def train_data(iq,symbol,symbols,timeframe):




    df = iq_get_data(iq,symbol,symbols,timeframe)

    # df.to_csv(r'EURUSD.csv')
    df = Indicators(df)



    df.isnull().sum().sum() # there are no nans
    df.fillna(method="ffill", inplace=True)
    df = df.loc[~df.index.duplicated(keep = 'first')]
    df['future'] = df["GOAL"].shift(-predict_period)




    #df = df.drop(columns = {'open','min','max'})

    df = df.dropna()
    dataset = df.fillna(method="ffill")
    dataset = dataset.dropna()

    dataset.sort_index(inplace = True)

    main_df = dataset

    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)

    main_df['target'] = list(map(classify, main_df['GOAL'], main_df['future']))

    main_df.dropna(inplace=True)

    main_df['target'].value_counts()

    main_df.dropna(inplace=True)

    main_df = main_df.astype('float32')
    if VALIDATION_TRAIN:
        times = sorted(main_df.index.values)
        last_5pct = sorted(main_df.index.values)[-int(0.1*len(times))]
        
        validation_main_df = main_df[(main_df.index >= last_5pct)]
        main_df = main_df[(main_df.index < last_5pct)]
        
        train_x, train_y = preprocess_df(main_df)
        validation_x, validation_y = preprocess_df(validation_main_df)
        
        print(f"train data: {len(train_x)} validation: {len(validation_x)}")
        print(f"sells: {train_y.count(0)}, buys: {train_y.count(1)}")
        print(f"VALIDATION sells: {validation_y.count(0)}, buys : {validation_y.count(1)}")
        
        train_y = np.asarray(train_y)
        validation_y = np.asarray(validation_y)
    else:
        train_x, train_y = preprocess_df(main_df)
        print(f"train data: {len(train_x)}")
        print(f"sells: {train_y.count(0)}, buys: {train_y.count(1)}")
        train_y = np.asarray(train_y)
        

    NAME = f"{LEARNING_RATE}-{seq_len}-SEQ-{predict_period}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"  
    




    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLYSTOP)
    model = Sequential()
    model.add(LSTM(NEURONS, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(NEURONS, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(NEURONS))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(NEURONS, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)


    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


    filepath = "ThesisBrain"
    if VALIDATION_TRAIN:
        checkpoint = ModelCheckpoint("models/{}.h5".format(filepath), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
        # Train model
        model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint, earlyStoppingCallback],
        )
    else:
        checkpoint = ModelCheckpoint("models/{}.h5".format(filepath), monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
        # Train model
        model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tensorboard, checkpoint, earlyStoppingCallback],
        )
    
    return filepath

