import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def create_AE_model(hidden):  # オートエンコーダを構築する関数
    inputL = Input(shape=(inputs))
    
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(32, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(inputs, name="dense4")(x2)
    COPD_DAE = Model(inputs = inputL, outputs = outputL)
    
    COPD_DAE.compile(optimizer=Adam(lr=0.001), loss="mae")
    
    COPD_DAE.summary()
    
    return COPD_DAE


def create_FT_model(hidden, w_AE1, w_AE2, w_AE3):  # ファインチューニングモデルを構築する関数
  
    inputL = Input(shape=(inputs))
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(32, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(1, activation="relu", name="dense4")(x2)
    FT_model = Model(inputs = inputL, outputs = outputL)
    
    FT_model.summary()

    FT_model.layers[1].set_weights(w_AE1)
    FT_model.layers[2].set_weights(w_AE2)
    FT_model.layers[3].set_weights(w_AE3)

    FT_model.compile(optimizer=Adam(lr=0.001), loss='mae')
    
    FT_model.summary()
    
    return FT_model



df_tweet_dataset = pd.read_csv('./Dataset/Dataset.csv', encoding='utf_8_sig')

good_num = df_tweet_dataset['いいね数']

x_train, x_test, t_train, t_test = train_test_split(df_tweet_dataset, good_num, train_size=0.8, random_state=0)

x_train_tweet = x_train['ツイートText']
x_test_tweet = x_test['ツイートText']

x_train = x_train.drop(['Unnamed: 0', 'ツイートID', 'ツイートText', 'ツイートURL', 'キーワード', 'ツイート時刻', '鍵垢flag', 'ユーザID', 'いいね数'], axis=1)
x_test = x_test.drop(['Unnamed: 0', 'ツイートID', 'ツイートText', 'ツイートURL', 'キーワード', 'ツイート時刻', '鍵垢flag', 'ユーザID', 'いいね数'], axis=1)

mm = preprocessing.MinMaxScaler()
norm_x_train = mm.fit_transform(x_train)
norm_x_test = mm.fit_transform(x_test)

np.save('./saved_data/normalization_user_infomation.npy', norm_x_test)

inputs = len(x_train.columns)
hidden = 64


AE_model = create_AE_model(hidden)

history = AE_model.fit(norm_x_train, t_train,
                       epochs=100,
                       batch_size=32,
                       shuffle=True)

AE_model.save('./saved_model/AutoEncoder_model.h5')
# AE_model.load_weights('./saved_model/AutoEncoder_model.h5')

weight_AE1 = []
weight_AE2 = []
weight_AE3 = []

weight_AE1.append(AE_model.layers[1].get_weights())
weight_AE2.append(AE_model.layers[2].get_weights())
weight_AE3.append(AE_model.layers[3].get_weights())
weight_AE2[0]

FT_model = create_FT_model(hidden, weight_AE1[0], weight_AE2[0], weight_AE3[0])  

history2 = FT_model.fit(norm_x_train, t_train, 
                     batch_size=29, 
                     epochs=100)

FT_model.save('./saved_model/fine_tuning_model.h5')
# FT_model.load_weights('./saved_model/fine_tuning_model.h5')


