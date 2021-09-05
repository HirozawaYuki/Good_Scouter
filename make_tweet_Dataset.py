import tweepy
import datetime
import pandas as pd

# キーワードの結果をcsvファイルに書き出すために情報をDataFrameに登録する関数　引数(現在のDataFrame, キーワード, 文章数, ジャンル(自身で決定))

def register_sentences(df, search_keyword, sentence_count, genre):
#   print(search_keyword)
  tweets = api.search(q=search_keyword, count=sentence_count, tweet_mode='extended')
  tweet_data = []

  date = datetime.datetime.now() + datetime.timedelta(hours=9)  # 現在時刻の取得

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




keys = pd.read_csv('./Twitter_API_Key.csv', encoding='CP932')
# print(keys['key'])

key_data = keys['key']

API_KEY = key_data[0]
API_SECRET = key_data[1]
ACCESS_TOKEN = key_data[2]
ACCESS_TOKEN_SECRET = key_data[3]

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


df = pd.DataFrame(columns=['ツイートID', 'ツイートText', 'いいね数', 'フォロー数', 'ユーザフォロワー数', 'ツイート時刻', 'ツイート後経過時間(h)', 'ユーザID', 'ユーザ総ツイート数', '鍵垢flag', 'ツイートURL', 'キーワード'])

# キーワード検索で探索(条件：リプライやURL,画像, 動画付きツイートを除外、いいね数が5以上で2021年8月31日22時までにツイートされたもの) 8月31日23時40分現在

# df = register_sentences(df, search_keyword, sentence_count, genre)
SEARCH_WORD = "人工知能 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"  # 「人工知能」とキーワード検索
df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
SEARCH_WORD = "野球 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
SEARCH_WORD = "5G exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
SEARCH_WORD = "サッカー exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
SEARCH_WORD = "バドミントン exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "スポーツ")
SEARCH_WORD = "フジロック exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "音楽")
SEARCH_WORD = "紅白歌合戦 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "音楽")
SEARCH_WORD = "為替 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "金融")
SEARCH_WORD = "日銀 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "金融")
SEARCH_WORD = "衆院選 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "政治")
SEARCH_WORD = "IoT exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "テクノロジー")
SEARCH_WORD = "自民党 exclude:replies -filter:links min_faves:5 exclude:replies until:2021-08-31_22:00:00_JST"
df = register_sentences(df, SEARCH_WORD, 100, "政治")

df.to_csv('./Dataset/test.csv', mode='w', encoding='utf_8_sig')

