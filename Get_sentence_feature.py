import pandas as pd
import MeCab
import pickle
import numpy as np
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


# 辞書に単語を登録するための関数

def make_dictionary(data, dictionary):
  vec_data = []
  len_sentence = []

  for row in data:

    # 各行(row)を形態素で分割
    m = MeCab.Tagger('-Owakati')
    # m = MeCab.Tagger('-chasen')
    if not row == row:
      print('error')
      row = ' '
    row_mecab = m.parse(row)
    row_mecab = row_mecab.split(' ')


    # print(row_mecab)
    # print(pos)
    vec_sentence = []
    count = 0

    for word in row_mecab:
      #辞書に含まれていない単語を逐次追加
      if word not in dictionary:
        dictionary.append(word)

      #各行を単語INDEXで表現
      vec_sentence.append(dictionary.index(word))
      count += 1
    #データ全体の単語INDEXを取得
    vec_data.append(vec_sentence)
    len_sentence.append(count)

  return dictionary, np.array(vec_data), np.array(len_sentence)


dim_embedding = 16
epochs_w2v = 10
epochs_lstm = 100
batchsize_w2v = 128
batchsize_lstm = 128
dim_z = 25  # LSTMの特徴ベクトル次元数

tf.random.set_seed(0)

# 日付情報を取得
f = open('./saved_data/date_info.txt', 'rb')
date_info = pickle.load(f)
f.close()

# tweet_information = pd.read_csv('./Dataset/test.csv', encoding='CP932')
tweet_information = pd.read_csv('./Dataset/test_'+date_info+'.csv', encoding='utf_8_sig')

# print(tweet_information)

tweet_text = tweet_information['ツイートText']
num_tweet_text = len(tweet_text)


#辞書INDEX0番目は空白として予約する
dictionary = [' ']

dictionary, vec_data, len_sentence = make_dictionary(tweet_text, dictionary)


# ---------------辞書データの保存----------------
# f = open('./saved_data/dictionary.txt', 'wb')
f = open('./saved_data/dictionary_'+date_info+'.txt', 'wb')
list_row = dictionary
pickle.dump(list_row, f)
f.close()
# ----------------------------------------------



# ------------Word2Vec学習用データセットの作成(各単語の前後5単語の情報を作成)--------------------
window_size = 5
input_data = []
output = []

for row in vec_data:
 
    # Lは各文の形態素数
    L = len(row)
    
    for i in range(L):
        begin = i - window_size
        end = i + window_size + 1
        input_seq = []
    
        # 当該単語の周辺の単語の辞書番号を格納(文章がない部分は0パディング)
        for j in range(begin, end):
            if (j < 0):
                input_seq.append(0)
            elif(0 <= j < L and j != i):
                input_seq.append(row[j])
            elif(j >= L):
                input_seq.append(0)

        output.append(row[i])
        input_data.append(input_seq)


x_train = np.array(input_data)
# outputをone-hot化．クラス数は辞書数にmask用単語の領域を引いた(-1した)ものと同じ
y_train = np.array(to_categorical(output, len(dictionary)))

# print('Shape of x_train:', x_train.shape)
# print('Shape of y_train:', y_train.shape)

# ------------------------------------------------------------------------------------------------

# Word2Vecで学習(CBOW)
# Embeddingの重みを単語の特徴量として利用する
num_word = y_train.shape[1] #辞書の単語数

# 形態素の平均ベクトルから中央の形態素を予測
model = Sequential([
    layers.Embedding(num_word, dim_embedding, input_length=window_size*2),
    layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim_embedding,)),
    layers.Dense(num_word, activation='softmax'),
])

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#学習
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs_w2v)
# model.save_weights('./saved_model/20210902_w2v_weight.h5')

w2v = model.get_weights()[0]

# np.save('./saved_data/w2v.npy', w2v)
np.save('./saved_data/w2v_'+date_info+'.npy', w2v)

data_cls = tweet_information['キーワード']

# -----ジャンル名に対するINDEX番号の付与(教師データの作成)--------------------
dic_class = []
y_train_1 = []

for i in range(len(data_cls)):
  if not data_cls[i] in dic_class:
    dic_class.append(data_cls[i])

for i in range(len(dic_class)):
  if not dic_class[i] == dic_class[i]:
    dic_class[i] = ' '

for i in range(len(data_cls)):
  for j in range(len(dic_class)):
    if data_cls[i] == dic_class[j] or (data_cls[i] != data_cls[i] and dic_class[j] == ' '):
      label_num = j
  y_train_1.append(label_num)


y_train_1 = np.array(to_categorical(y_train_1))

# -------------------------------------------------------------------------

# Paddingで文章長をそろえ、データの形を確認する
input_w2v = pad_sequences(vec_data, dtype='float32')
timesteps = input_w2v.shape[1]

cls_num = len(dic_class)

input_dim_w2v = len(w2v)
output_dim_w2v = w2v.shape[1]


# -------------ツイート文章を基にジャンル名を予測する学習モデルの構築------------
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



# path_weight = './saved_model/LSTM_Classification_weight.h5'
path_weight = './saved_model/LSTM_Classification_weight_'+date_info+'.h5'

best_saver = tf.keras.callbacks.ModelCheckpoint(filepath=path_weight, monitor='loss', verbose=1, save_best_only=True, mode='auto')

# loss_list = []
# out_list = []

for i in range(epochs_lstm):
  print('epochs:', i+1)
  result = LSTM_Classification.fit(input_w2v, 
                      y_train_1, 
                      epochs=1, 
                      batch_size=batchsize_lstm, 
                      shuffle=True, 
                      callbacks=[best_saver],
                      )

#   loss_list.extend(result.history['loss'])
#   out_list.extend(result.history['acc'])


# LSTM_Classification.load_weights(path_weight)


# ----------input_w2v(ツイート集)を入力としたときの各文章の特徴量部分を抽出(cls)-----------
layer_name = 'bidirectionalx'
model_trial = Model(inputs=LSTM_Classification.input, outputs=LSTM_Classification.get_layer(layer_name).output)
cls = model_trial.predict([input_w2v])
# -------------------------------------------------------------------------------------


# pd.DataFrame(cls).to_csv('./Dataset/feature.csv')  # 取得した特徴量をcsvファイルに出力
pd.DataFrame(cls).to_csv('./Dataset/feature_'+date_info+'.csv')  # 日付毎にデータを保持したい場合はこちらを実行
