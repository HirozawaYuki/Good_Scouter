# Good_Scouter

## * 環境構築
$ pip install mecab-python3 <br>
$ pip install unidic-lite <br>
$ pip install tweepy==3.10.0 <br>
$ pip install scikit-learn <br>
他<br><br>

## * プログラムについて
  * make_tweet_Dataset.py　(実行は一度で良い) <br>
    APIキーやACCESS Tokenを登録して(念のためダミーデータを入れているので実行する際は正しいデータに置き換える必要あり)、ツイートをキーワード検索してcsv(test.csv)に格納する。 <br><br>
  * Get_sentence_feature.py　(実行は一度で良い) <br>
    ツイート情報(test.csv)内のツイート文章を基に、単語のベクトル化(Word2Vec)とジャンル名予測問題を通じて各文章の特徴量50次元を取得する。 <br>
     各文章特徴量取得後、csvファイル(feature.csv)に格納する。 <br><br>
  * Concatenate_Dataset.py　(実行は一度で良い) <br>
    ツイート情報(test.csv)と文章特徴量(feature.csv)を列方向に結合して、csv(Dataset.csv)に格納 <br><br>
  * AutoEncoder_learning.py　(実行は一度で良い) <br>
    Dataset.csvのいくつかの情報をAutoEncoderで学習し、ファインチューニングを行い、いいね数を予測する。 <br>
    AutoEncoderとFine tugning学習モデルの重みとデータセットを正規化した時の値を保存しておく必要あり。<br><br>
    
  * Good_Scouter_System.py
  　いいねスカウター実装版 <br>
    「該当ユーザ名(@の後の半角英字)」、「何時間後のいいねを予測してほしいか」、「ツイートしたい文章」を入力するといいね数を返してくれる。 <br><br>

## * 今後の予定
  * フロントエンド部の実装<br>
  * 余裕があれば、ユーザ情報入力後、そのユーザーの直近ツイートといいね数も学習に加える機能の追加
