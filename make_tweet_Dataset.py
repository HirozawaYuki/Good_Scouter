import tweepy
import datetime
import pandas as pd
import pickle
import msoffcrypto
import getpass
import io

# キーワードの結果をcsvファイルに書き出すために情報をDataFrameに登録する関数　引数(現在のDataFrame, キーワード, 文章数, ジャンル(自身で決定))

def register_sentences(df, search_keyword, sentence_count, genre):
#   print(search_keyword)
  tweets = api.search(q=search_keyword, count=sentence_count, tweet_mode='extended')
  

  date = datetime.datetime.now() # 現在時刻の取得

  for t in tweets:
    # ツイート時刻を取得して、日本時間に変換（UTC→JST）
    tweeted_at = t.created_at + datetime.timedelta(hours=9)

    # ツイートしてから何時間経過しているかを取得(切り捨て)
    time_difference = date - tweeted_at
    hour_difference = time_difference.days*24+int(time_difference.seconds/3600)  # 時間単位でツイート後の経過時間を取得

    # 鍵アカウントか否かを二値化(鍵垢:1, 非鍵垢:0)
    protected_flag = 0
    if(t.user.protected == True):
      protected_flag = 1

    df = df.append({'ツイートID': t.id, 
                    'ツイートText': t.full_text, 
                    'いいね数': t.favorite_count,
                    'フォロー数':t.user.friends_count, 
                    'ユーザフォロワー数': t.user.followers_count,
                    'ツイート時刻': tweeted_at, 
                    'ツイート後経過時間(h)': hour_difference, 
                    'ユーザID': t.user.screen_name, 
                    'ユーザ総ツイート数': t.user.statuses_count, 
                    '鍵垢flag': protected_flag, 
                    'ツイートURL': f"https://twitter.com/{t.user.screen_name}/status/{t.id}",
                    'キーワード':genre},
                  ignore_index=True
                    )

  return df


#---------------APIキーの読み込み----------------
print('APIキーを読み込みます\nTwitter_API_Key_pass.xlsxのパスワードを入力して下さい')
excelpass = getpass.getpass('パスワード：')

decrypted = io.BytesIO()

with open("Twitter_API_Key_pass.xlsx", "rb") as f:
    file = msoffcrypto.OfficeFile(f)
    file.load_key(password=excelpass)
    file.decrypt(decrypted)

keys = pd.read_excel(decrypted, engine="openpyxl")
key_data = keys['key']

API_KEY = key_data[0]
API_SECRET = key_data[1]
ACCESS_TOKEN = key_data[2]
ACCESS_TOKEN_SECRET = key_data[3]

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
#-----------------------------------------------


df = pd.DataFrame(columns=['ツイートID', 'ツイートText', 'いいね数', 'フォロー数', 'ユーザフォロワー数', 'ツイート時刻', 'ツイート後経過時間(h)', 'ユーザID', 'ユーザ総ツイート数', '鍵垢flag', 'ツイートURL', 'キーワード'])

# キーワード検索で探索(条件：リプライやURL,画像, 動画付きツイートを除外、いいね数が5以上で2021年8月31日22時までにツイートされたもの) 8月31日23時40分現在

# df = register_sentences(df, search_keyword, sentence_count, genre)
# SEARCH_WORD = "人工知能 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"  # 「人工知能」とキーワード検索
# df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
# SEARCH_WORD = "野球 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
# SEARCH_WORD = "5G min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
# SEARCH_WORD = "サッカー min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
# SEARCH_WORD = "バドミントン min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
# SEARCH_WORD = "フジロック min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "音楽")
# SEARCH_WORD = "紅白歌合戦 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "音楽")
# SEARCH_WORD = "為替 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "金融")
# SEARCH_WORD = "日銀 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "金融")
# SEARCH_WORD = "衆院選 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "政治")
# SEARCH_WORD = "IoT min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
# SEARCH_WORD = "自民党 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "政治")
# SEARCH_WORD = "NBA min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
# SEARCH_WORD = "DX min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
# SEARCH_WORD = "ピアノ min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "音楽")
# SEARCH_WORD = "アニソン min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
# df = register_sentences(df, SEARCH_WORD, 100, "音楽")
#SEARCH_WORD = "大谷翔平 min_faves:3 exclude:replies since:2021-09-25_11:00:00_JST until:2021-09-25_16:00:00_JST -filter:links"
#df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")


date = datetime.datetime.now()
date_info = '{}{}'.format(str(date.month).zfill(2), str(date.day).zfill(2))

# ---------------日時データの保存----------------
f = open('./saved_data/date_info.txt', 'wb')
pickle.dump(date_info, f)
f.close()
#-----------------------------------------------

# df.to_csv('./Dataset/test.csv', mode='w', encoding='utf_8_sig')
df.to_csv('./Dataset/test_'+date_info+'.csv', mode='w', encoding='utf_8_sig')  # 日にち毎に違うフォルダにデータを保存したい場合はこちらを利用
