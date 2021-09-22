
from flask import Flask, render_template, Response, redirect, url_for, session, request


app = Flask(__name__)

@app.route('/')
def index():
    user_name = ""  # ユーザー名の初期化
    hour = 0  # 時間の初期化
    
    return render_template('./index.html')

@app.route("/input_tweet", methods=["post"])
def tweet():
    user_name = request.form["user_name"]
    global icon_url
    icon_url = "/static/image/star.jpg"

    return render_template("input_tweet.html", name=user_name, icon_url=icon_url)


@app.route("/result", methods=["post"])
def result():
    tweet_text = request.form["tweet_text"]

    heart = 10

    return render_template("result.html", heart=heart, tweet_text=tweet_text, icon_url=icon_url)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=8080)
