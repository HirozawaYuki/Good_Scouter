import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

def create_AE_model(hidden, inputs):  # オートエンコーダを構築する関数
    inputL = Input(shape=(inputs))
    
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(hidden//2, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(inputs, name="dense4")(x2)
    COPD_DAE = Model(inputs = inputL, outputs = outputL)
    
    COPD_DAE.compile(optimizer=Adam(lr=0.001), loss="mae")
    
    COPD_DAE.summary()
    
    return COPD_DAE


def create_FT_model(hidden, w_AE1, w_AE2, inputs):  # ファインチューニングモデルを構築する関数
  
    inputL = Input(shape=(inputs))
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(hidden//2, activation="relu", name="dense2")(x)
    # x2 = Dense(hidden//4, activation="relu", name="dense3")(x1)
    # outputL = Dense(1, activation="relu", name="dense4")(x2)
    outputL = Dense(1, activation="relu", name="dense4")(x1)
    FT_model = Model(inputs = inputL, outputs = outputL)
    
    FT_model.summary()

    FT_model.layers[1].set_weights(w_AE1)
    FT_model.layers[2].set_weights(w_AE2)
    #FT_model.layers[3].set_weights(w_AE3)

    FT_model.compile(optimizer=Adam(lr=0.001), loss='mae')
    
    FT_model.summary()
    
    return FT_model


# 日付情報を取得
f = open('./saved_data/date_info.txt', 'rb')
date_info = pickle.load(f)
f.close()

# df_tweet_dataset = pd.read_csv('./Dataset/Dataset.csv', encoding='utf_8_sig')
df_tweet_dataset = pd.read_csv('./Dataset/Dataset_'+date_info+'.csv', encoding='utf_8_sig')

good_num = df_tweet_dataset['いいね数']

past_user_information = df_tweet_dataset.drop(['Unnamed: 0', 'ツイートID', 'ツイートText', 'ツイートURL', 'キーワード', 'ツイート時刻', '鍵垢flag', 'ユーザID', 'いいね数'], axis=1)
past_user_information = past_user_information.to_numpy()

# np.save('./saved_data/past_user_information.npy', past_user_information)
np.save('./saved_data/past_user_information_'+date_info+'.npy', past_user_information)

x_train, x_test, t_train, t_test = train_test_split(df_tweet_dataset, good_num, train_size=0.8, random_state=0)

x_train_tweet = x_train['ツイートText']
x_test_tweet = x_test['ツイートText']

x_train = x_train.drop(['Unnamed: 0', 'ツイートID', 'ツイートText', 'ツイートURL', 'キーワード', 'ツイート時刻', '鍵垢flag', 'ユーザID', 'いいね数'], axis=1)
x_test = x_test.drop(['Unnamed: 0', 'ツイートID', 'ツイートText', 'ツイートURL', 'キーワード', 'ツイート時刻', '鍵垢flag', 'ユーザID', 'いいね数'], axis=1)

mm = preprocessing.MinMaxScaler()
norm_x_train = mm.fit_transform(x_train)
norm_x_test = mm.fit_transform(x_test)

# np.save('./saved_data/normalization_user_information.npy', norm_x_test)
np.save('./saved_data/normalization_user_information_'+date_info+'.npy', norm_x_test)

AE_inputs_columns = len(x_train.columns)
hidden = 64


AE_model = create_AE_model(hidden, AE_inputs_columns)

history = AE_model.fit(norm_x_train, norm_x_train,
                       epochs=200,
                       batch_size=32,
                       shuffle=True)

# AE_model.save('./saved_model/AutoEncoder_model.h5')
AE_model.save('./saved_model/AutoEncoder_model_'+date_info+'.h5')

weight_AE1 = []
weight_AE2 = []
weight_AE3 = []

weight_AE1.append(AE_model.layers[1].get_weights())
weight_AE2.append(AE_model.layers[2].get_weights())
weight_AE3.append(AE_model.layers[3].get_weights())
weight_AE2[0]

# FT_model = create_FT_model(hidden, weight_AE1[0], weight_AE2[0], weight_AE3[0], AE_inputs_columns)
FT_model = create_FT_model(hidden, weight_AE1[0], weight_AE2[0], AE_inputs_columns)

history2 = FT_model.fit(norm_x_train, t_train, 
                     batch_size=29, 
                     epochs=100)

# FT_model.save('./saved_model/fine_tuning_model.h5')
FT_model.save('./saved_model/fine_tuning_model_'+date_info+'.h5')
