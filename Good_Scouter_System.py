import tweepy
import datetime
import pandas as pd
import MeCab
import pickle
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (Embedding, Dense, 
                                     GlobalAveragePooling2D, Conv2D, Multiply,
                                     Lambda, Input, LSTM, Bidirectional, Dropout, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def make_tweet_w2v(input_sentence, timesteps, dictionary):
  
    tweet_w2v = []
    row_mecab = []

    # 各行(row)を形態素で分割
    m = MeCab.Tagger('-Owakati')
    # m = MeCab.Tagger('-chasen')
   
    row_mecab = m.parse(input_sentence)
    row_mecab = row_mecab.split(' ')


    vec_sentence = []
    count = 0

    for word in row_mecab:
      #辞書に含まれていない単語を逐次追加
        if word not in dictionary:
            tweet_w2v.append(0)
        else:
            dictionary.append(word)

        count += 1
    
    tweet_w2v = np.array(tweet_w2v)
    tweet_w2v = np.pad(tweet_w2v, [timesteps-count, 0])

    return tweet_w2v


def create_AE_model(hidden):
    
    inputL = Input(shape=(inputs))
    
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(32, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(inputs, name="dense4")(x2)
    COPD_DAE = Model(inputs = inputL, outputs = outputL)
    
    COPD_DAE.compile(optimizer=Adam(lr=0.001), loss="mae")
    
    COPD_DAE.summary()
    
    return COPD_DAE


def create_FT_model(hidden, w_AE1, w_AE2, w_AE3):
  
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


keys = pd.read_csv('./Twitter_API_Key_dummy.csv', encoding='CP932')
# print(keys['key'])

key_data = keys['key']

API_KEY = key_data[0]
API_SECRET = key_data[1]
ACCESS_TOKEN = key_data[2]
ACCESS_TOKEN_SECRET = key_data[3]

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

name = input("ユーザ名(半角英字)を入力してください:")
candidate_tweet = input("ツイートを入力してください(140字以内):")
time = input('何時間後のいいね数を予測しますか。(半角数字):')


user = api.get_user(name)  # 入力情報を基にユーザ名の取得
follow = user.friends_count  # 該当ユーザのフォロー数を代入
follower = user.followers_count  # 該当ユーザのフォロワー数を代入
total_tw = user.statuses_count  # 該当ユーザの総ツイート数を表示

hidden = 64
dim_z = 25
timesteps = 115  # 今回のシステムでは、140にしておけばよかったと後悔
cls_num = 5
w2v = np.load('./saved_data/w2v.npy')
input_dim_w2v = w2v.shape[0]
dim_embedding = w2v.shape[1]

inputs = Input(shape=(timesteps, ))

# inputsをw2vでベクトル化
embed = Embedding(input_dim_w2v, 
                  dim_embedding, 
                  input_length=timesteps, 
                  weights=[w2v], 
                  mask_zero=True, 
                  trainable=False, 
                  name='layer_0')(inputs)

embed_dim = Lambda(lambda x: K.expand_dims(x, 1), output_shape=(1, timesteps, dim_embedding, ))(embed)

c0x = GlobalAveragePooling2D(name='layer_2x')(embed_dim)
c1x = Dense(dim_embedding//2, activation="relu", name='layer_3x')(c0x)
c2x = Dense(dim_embedding, activation="sigmoid", name='layer_4x')(c1x)
c2x = Reshape((1, dim_embedding), input_shape=(dim_embedding,))(c2x)

cx = Multiply(name='layer_5x')([embed, c2x])

sx = Conv2D(1, 1, activation='sigmoid', name='layer_6x')(embed_dim)
sx = Multiply(name='layer_7x')([embed, sx])

sx = Lambda(lambda x: tf.squeeze(x, 1), name='layer_8x')(sx)

attx = layers.add([cx, sx])

encodedx = Bidirectional(
    LSTM(dim_z, 
         batch_input_shape=(None, timesteps, dim_embedding), 
         activation="tanh", 
         recurrent_activation="sigmoid", 
         return_sequences=False
         ),
    name='bidirectionalx')(attx)

encodedx = Dropout(0.2)(encodedx)

out = Dense(units=cls_num, activation='softmax', name='out')(encodedx)

# ---------------------------------------------------------------------------------------

LSTM_Classification = Model(inputs, out, name='LSTM_Classification')

LSTM_Classification.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), 
                loss={ 'out':'categorical_crossentropy'}, 
                metrics=['acc'])

LSTM_Classification.summary()

LSTM_Classification.load_weights('./saved_model/LSTM_Classification_weight.h5')

f = open('./saved_data/dictionary.txt', 'rb')
dictionary = pickle.load(f)

tweet_w2v = make_tweet_w2v(candidate_tweet, timesteps, dictionary)
tweet_w2v = tweet_w2v.reshape(-1, timesteps)

layer_name = 'bidirectionalx'
# layer_name = 'layer_0'
get_sentence_feature = Model(inputs=LSTM_Classification.input, outputs=LSTM_Classification.get_layer(layer_name).output)

test_sentence_feature = get_sentence_feature([tweet_w2v])

normalization_user_information = np.load('./saved_data/normalization_user_information.npy')

test_input = []
test_input.append(follow)
test_input.append(follower)
test_input.append(int(time))
test_input.append(total_tw)
test_input.extend(test_sentence_feature[0])
test_input = np.array(test_input)
test_input = test_input.reshape(-1, normalization_user_information.shape[1])
print(test_input)

mm = preprocessing.MinMaxScaler()
normalization_user_inv = mm.inverse_transform(normalization_user_information)

tweet_data_include_candidate_tweet = np.insert(normalization_user_inv, normalization_user_inv.shape[0], test_input, axis=0)

normalization_tweet_data_include_candidate_tweet = mm.fit_transform(tweet_data_include_candidate_tweet)

normalization_candidate_tweet = normalization_tweet_data_include_candidate_tweet[-1]
normalization_candidate_tweet = np.array(normalization_candidate_tweet)
normalization_candidate_tweet = normalization_candidate_tweet.reshape(-1,  normalization_candidate_tweet.shape[0])


AE_model = create_AE_model(hidden)
AE_model.load_weights('./saved_model/AutoEncoder_model.h5')
weight_AE1 = []
weight_AE2 = []
weight_AE3 = []

weight_AE1.append(AE_model.layers[1].get_weights())
weight_AE2.append(AE_model.layers[2].get_weights())
weight_AE3.append(AE_model.layers[3].get_weights())

FT_model = create_FT_model(hidden, weight_AE1[0], weight_AE2[0], weight_AE3[0])  
FT_model.load_weights('./saved_model/fine_tuning_model.h5')

good_test_output = FT_model.predict(normalization_candidate_tweet)
print('予測されるいいね数:', int(good_test_output[0][0]))
