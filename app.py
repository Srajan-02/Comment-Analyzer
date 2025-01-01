# from model import analyze_youtube_comments
# # from Bert import analyze_youtube_comments
# # from lstm import analyze_youtube_comments
# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__)

# folder_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files"


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None

#     if request.method == 'POST':
#         video_link = request.form['video_link']
#         result = analyze_youtube_comments(
#             video_link)

#     return render_template('index.html', result=result)


# if __name__ == '__main__':
#     app.run(debug=True)





#------------------------------- BERT----------------------------------------------------------------
# from model import analyze_youtube_comments
# # from Bert import analyze_youtube_comments
# import os
# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__ , static_folder = 'static')
# static_image_folder = os.path.join(app.root_path, 'static/Images01')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     pie_chart_url = None

#     if request.method == 'POST':
#        video_link = request.form['video_link']
#        feedback, _, pie_chart_path = analyze_youtube_comments(video_link)
#        result = feedback
#        if pie_chart_path:
#         pie_chart_url = url_for('static', filename=f'images/{os.path.basename(pie_chart_path)}')
#     return render_template('index.html', result=result, pie_chart_url=pie_chart_url)


# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, render_template, request, redirect, url_for
from model import analyze_youtube_comments
import os
app = Flask(__name__, static_folder='static')

static_image_folder = os.path.join(app.root_path, 'static/Images01')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    pie_chart_url = None
    line_chart_url = None
    graph_url = None

    if request.method == 'POST':
        video_link = request.form['video_link']
        sentiment_feedback, line_chart_path, graph_path = analyze_youtube_comments(video_link)
        # sentiment_feedback = analyze_youtube_comments(video_link)


        result = sentiment_feedback
        
        print("Sentiment Feedback:", sentiment_feedback)
        print("Length of Sentiment Feedback:", len(sentiment_feedback))

        # if pie_chart_path:
        #     pie_chart_url = url_for('static', filename=f'Pie_chart/{os.path.basename(pie_chart_path)}')
        if line_chart_path:
            line_chart_url = url_for('static', filename=f'Line_chart/{os.path.basename(line_chart_path)}')
        if graph_path:
            graph_url = url_for('static', filename=f'Graph_chart/{os.path.basename(graph_path)}')

    return render_template('index.html', result=result, line_chart_url=line_chart_url, graph_url=graph_url)









# from flask import Flask, render_template, request
# from model import analyze_youtube_comments
# import os
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# folder_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files"
# pie_chart_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Pie_Chart"
# line_chart_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Line_chart"
# graph_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Graph_chart"

# # Define default file names for each type of chart
# default_pie_chart_name = "pie_chart.png"
# default_line_chart_name = "line_chart.png"
# default_graph_name = "graph.png"

# def save_plot(plt, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     plt.savefig(os.path.join(folder, filename))
#     plt.close()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None

#     if request.method == 'POST':
#         video_link = request.form['video_link']
#         feedback, accuracy, df, pie_chart_path, line_chart_path, graph_path = analyze_youtube_comments(video_link)

#         # Save charts
#         save_plot(plt, pie_chart_folder, default_pie_chart_name)
#         save_plot(plt, line_chart_folder, default_line_chart_name)
#         save_plot(plt, graph_folder, default_graph_name)

#         result = {
#             'feedback': feedback,
#             'accuracy': accuracy,
#             'pie_chart_path': pie_chart_path,
#             'line_chart_path': line_chart_path,
#             'graph_path': graph_path
#         }

#     return render_template('index.html', result=result)


# if __name__ == '__main__':
#     app.run(debug=True)

