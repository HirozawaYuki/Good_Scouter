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
from flask import Flask, render_template, Response, redirect, url_for, session, request


def make_tweet_w2v(input_sentence, timesteps, dictionary):
  
    tweet_w2v = []
    row_mecab = []

    # 各行(row)を形態素で分割
    m = MeCab.Tagger('-Owakati')
    # m = MeCab.Tagger('-chasen')
   
    row_mecab = m.parse(input_sentence)
    row_mecab = row_mecab.split(' ')

    count = 0

    for word in row_mecab:
      #辞書に含まれていない単語を逐次追加
        if word not in dictionary:
            tweet_w2v.append(0)
        else:
            tweet_w2v.append(dictionary.index[word])

        count += 1
    
    tweet_w2v = np.array(tweet_w2v)
    tweet_w2v = np.pad(tweet_w2v, [timesteps-count, 0])

    return tweet_w2v


def create_AE_model(hidden, inputs):
    
    inputL = Input(shape=(inputs))
    
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(32, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(inputs, name="dense4")(x2)
    COPD_DAE = Model(inputs = inputL, outputs = outputL)
    
    COPD_DAE.compile(optimizer=Adam(lr=0.001), loss="mae")
    
    # COPD_DAE.summary()
    
    return COPD_DAE


def create_FT_model(hidden, w_AE1, w_AE2, w_AE3, inputs):
  
    inputL = Input(shape=(inputs))
    x = Dense(hidden, activation="relu", name="dense1")(inputL)
    x1 = Dense(32, activation="relu", name="dense2")(x)
    x2 = Dense(hidden, activation="relu", name="dense3")(x1)
    outputL = Dense(1, activation="relu", name="dense4")(x2)
    FT_model = Model(inputs = inputL, outputs = outputL)

    FT_model.layers[1].set_weights(w_AE1)
    FT_model.layers[2].set_weights(w_AE2)
    FT_model.layers[3].set_weights(w_AE3)

    FT_model.compile(optimizer=Adam(lr=0.001), loss='mae')
    
    # FT_model.summary()
    
    return FT_model

app = Flask(__name__)

@app.route('/')
def index():
    user_name = ""  # ユーザー名の初期化
    hour = 0  # 時間の初期化
    
    return render_template('./index.html')

@app.route("/input_tweet", methods=["post"])
def tweet():
    user_name = request.form["user_name"]

    keys = pd.read_csv('./Twitter_API_Key_dummy.csv', encoding='CP932')


    key_data = keys['key']

    API_KEY = key_data[0]
    API_SECRET = key_data[1]
    ACCESS_TOKEN = key_data[2]
    ACCESS_TOKEN_SECRET = key_data[3]

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    global api, user
    api = tweepy.API(auth)

    # ユーザー情報を取得する処理
    try:
        user = api.get_user(user_name)
    # エラー処理
    except tweepy.error.TweepError:

        return render_template("index.html")

    else:
        user = api.get_user(user_name)  # 入力情報を基にユーザー名の取得
        icon_url = user.profile_image_url_https  # ★ユーザーのプロフィール画像のURLを代入

        return render_template("input_tweet.html", name=user.name, icon_url=icon_url)


@app.route("/result", methods=["post"])
def result():
    time = request.form["time_hour"]

    candidate_tweet = request.form["tweet"]  # ツイート情報を取得
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

    LSTM_Classification.load_weights('./saved_model/LSTM_Classification_weight.h5')

    f = open('./saved_data/dictionary.txt', 'rb')
    dictionary = pickle.load(f)
    f.close()

    tweet_w2v = make_tweet_w2v(candidate_tweet, timesteps, dictionary)
    tweet_w2v = tweet_w2v.reshape(-1, timesteps)

    layer_name = 'bidirectionalx'
    get_sentence_feature = Model(inputs=LSTM_Classification.input, outputs=LSTM_Classification.get_layer(layer_name).output)

    test_sentence_feature = get_sentence_feature([tweet_w2v])

    past_user_information = np.load('./saved_data/past_user_information.npy')

    test_input = []
    test_input.append(follow)
    test_input.append(follower)
    test_input.append(int(time))
    test_input.append(total_tw)
    test_input.extend(test_sentence_feature[0])
    test_input = np.array(test_input)
    test_input = test_input.reshape(-1, past_user_information.shape[1])

    mm = preprocessing.MinMaxScaler()

    tweet_data_include_candidate_tweet = np.insert(past_user_information, past_user_information.shape[0], test_input, axis=0)

    normalization_tweet_data_include_candidate_tweet = mm.fit_transform(tweet_data_include_candidate_tweet)

    normalization_candidate_tweet = normalization_tweet_data_include_candidate_tweet[-1]
    normalization_candidate_tweet = np.array(normalization_candidate_tweet)
    normalization_candidate_tweet = normalization_candidate_tweet.reshape(-1,  normalization_candidate_tweet.shape[0])

    AE_inputs_shape = normalization_candidate_tweet.shape[1]
    AE_model = create_AE_model(hidden, AE_inputs_shape)
    AE_model.load_weights('./saved_model/AutoEncoder_model.h5')
    weight_AE1 = []
    weight_AE2 = []
    weight_AE3 = []

    weight_AE1.append(AE_model.layers[1].get_weights())
    weight_AE2.append(AE_model.layers[2].get_weights())
    weight_AE3.append(AE_model.layers[3].get_weights())

    FT_model = create_FT_model(hidden, weight_AE1[0], weight_AE2[0], weight_AE3[0], AE_inputs_shape)  
    FT_model.load_weights('./saved_model/fine_tuning_model.h5')

    good_test_output = FT_model.predict(normalization_candidate_tweet)
    heart = int(good_test_output[0][0])

    return render_template("result.html", heart=heart)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=8080)
