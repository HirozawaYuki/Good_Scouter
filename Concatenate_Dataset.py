import pandas as pd

# ----ツイート情報の抽出-----
tweet_information = pd.read_csv('./Dataset/test.csv', encoding='CP932')

t_columns = list(tweet_information.columns)
del t_columns[0]
len_tweet_information_index = len(tweet_information.index.values)

tweet_information = tweet_information.loc[0:len_tweet_information_index-1, t_columns]

# ---------------------------
# -----特徴量部の抽出---------
data = pd.read_csv('./Dataset/feature.csv', encoding='CP932')

l_columns = list(data.columns)
del l_columns[0]
len_index = len(data.index.values)

sentence_feature = data.loc[0:len_index-1, l_columns]

# --------------------------------
# ---列方向に結合して、csvに格納----
df_concat = pd.concat([tweet_information, sentence_feature], axis=1)

df_concat.to_csv('./Dataset/Dataset.csv', mode='w', encoding='utf_8_sig')
