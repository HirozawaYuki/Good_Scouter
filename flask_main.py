from flask import Flask, render_template, Response, redirect, url_for, session, request

app = Flask(__name__)

@app.route('/')
def index():
    user_name = ""
    hour = 0
    return render_template('./index.html')



@app.route("/input_tweet",methods=["post"])
def tweet():
    user_name = request.form["user_name"]
    # ユーザー情報を取得する処理

    
    # エラー処理（ユーザー情報取得失敗）
    

    return render_template("input_tweet.html", name=user_name)

@app.route("/result",methods=["post"])
def result():
    hour = request.form["time_hour"]
    # tweetのいいね数を予測する処理

    heart = 10
    return render_template("result.html", heart=heart)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=8080)