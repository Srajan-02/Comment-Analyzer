from model import analyze_youtube_comments
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

folder_path = "C:\\Users\\india\\OneDrive\\Documents\\SRAJAN\\ML projects\\comment_analyzer\\Files"


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        video_link = request.form['video_link']
        result = analyze_youtube_comments(
            video_link)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
