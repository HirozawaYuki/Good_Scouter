# いいねScouter
2021年の研究室の実践ゼミで作成中の作品です。<br>
4人で開発しています。<br>

## * システム概要
 「Twitterのユーザ名」と、「Twitterにツイートしたい文章」、「何時間後のいいねを予測してほしいか」を入力すると、システムが予測されるいいね数を出力してくれます。<br><br>

## * システムの流れ
  * ユーザ名入力画面(初期画面) <br>
    <img alt="input_user_name" src="static/image/System_image1.JPG"><br><br>
    以下のようにユーザ名を入力する <br>
    <img alt="input_user_name2" src="static/image/System_image1_2.JPG"> <br><br>
    
  * ツイート入力画面 <br>
    ツイートしたい文章を140字以内で入力する。<br>
    <img alt="input_tweet" src="static/image/System_image2_1.JPG"> <br><br>
    
    <br><br>140字を越えた場合はいいね数予測画面に遷移することができない。<br>
    <img alt="input_tweet2" src="static/image/System_image2_2.JPG"> <br><br>
    
  * 予測いいね数出力画面 <br>
    n時間後のいいね数を予測して出力。<br>
    <img alt="output_result" src="static/image/System_image3.JPG"> <br><br>
    
上記のツイートでは、3時間経っても0いいねであると予測される。<br><br>

## * 環境構築
$ pip install mecab-python3 <br>
$ pip install unidic-lite <br>
$ pip install tweepy==3.10.0 <br>
$ pip install scikit-learn <br>
$ pip install Flask <br>
他<br><br>

## * 動作環境
python == 3.6.1 <br>
TensorFlow == 2.2.2 <br>
（パッケージcoloramaもインストールする必要があるかもしれません） <br>
<br>

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
    AutoEncoderとFine tuning学習モデルの重みとデータセットを正規化した時の値を保存しておく必要あり。<br><br>
    
  * Good_Scouter_System.py
  　いいねスカウター実装版 <br>
    「該当ユーザ名(@の後の半角英字)」、「何時間後のいいねを予測してほしいか」、「ツイートしたい文章」を入力するといいね数を返してくれる。 <br><br>

## * 今後の予定
  * ユーザ情報入力後、そのユーザーの直近ツイートといいね数も学習に加える機能の追加
  * ツイート候補文章からキーワードとなる単語を割り出し、そのキーワードに対する直近30分のツイートといいね数も学習に加える機能の追加
  * リツイート数とリプライ数の予測機能
