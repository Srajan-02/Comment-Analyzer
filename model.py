import uuid
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import googleapiclient.discovery
import pandas as pd
from urllib.parse import urlparse, parse_qs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import re
import numpy as np
import os
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from langdetect import detect
from indicnlp.tokenize import indic_tokenize
import nltk
import itertools  

nltk.download('indian')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

default_pie_chart_name = "pie_chart.png"
default_line_chart_name = "line_chart.png"
default_graph_name = "graph.png"

folder_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files"

# file_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files\\dataset.csv"
final_comments_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files\\final"
# pie_chart_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Pie_Chart"
line_chart_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Line_chart"
graph_folder = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\static\\Graph_chart"


def generate_unique_filename(folder, extension=".csv"):
    unique_filename = str(uuid.uuid4()) + extension
    return os.path.join(folder, unique_filename)

def generate_line_chart(df, file_path, default_line_chart_name):
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['published_date'] = df['published_at'].dt.date
    grouped_df = df.groupby('published_date').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df['published_date'], grouped_df['count'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.title('Number of Comments Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    line_chart_path = os.path.join(file_path, default_line_chart_name)
    plt.savefig(line_chart_path)
    # plt.close()
    # plt.show()

    return line_chart_path

import os
import matplotlib.pyplot as plt

def generate_graph(df, file_path, default_graph_name):
    sentiment_labels = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

    labels = df['Sentiment'].value_counts().index.tolist()
    sizes = df['Sentiment'].value_counts().values.tolist()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(labels)), sizes, color=['green', 'gray', 'red'])  # Swapped colors
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.xticks(range(len(labels)), [sentiment_labels[label] for label in labels])  # Set custom x-tick labels
    plt.grid(True)
    
    # Annotate bars with labels
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), size, ha='center', va='bottom')

    graph_path = os.path.join(file_path, default_graph_name)
    plt.savefig(graph_path)
    # plt.close()
    # plt.show()

    return graph_path

def analyze_youtube_comments(video_link, max_comments=3000):
    
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyC1YUwsMt-dRKwU1WRJ9YUZZewj7eNjVZQ"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    
    # Parse the video ID from the video link
    parsed_url = urlparse(video_link)
    video_id = parse_qs(parsed_url.query).get("v")
    if video_id:
        video_id = video_id[0]
        comments = []
        next_page_token = None

        while True:
            # Request comments
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['updatedAt'],
                    comment['likeCount'],
                    comment['textDisplay']
                ])
                # Stop fetching if the desired limit is reached
                if len(comments) >= max_comments:
                    break

            # Check if there are more pages of comments
            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(comments) >= max_comments:
                break

        # Create a DataFrame for the comments
        df = pd.DataFrame(comments, columns=[
            'author', 'published_at', 'updated_at', 'like_count', 'text'])

        # Remove existing CSV files in the folder
        folder_path = "./"
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"Removed: {csv_file}")
            except Exception as e:
                print(f"Error removing {csv_file}: {e}")

        # Save the dataset as 'dataset.csv'
        csv_filename = os.path.join(folder_path, 'dataset.csv')
        df.to_csv(csv_filename, index=False)
    else:
        print("Invalid YouTube video link.")
        return

    # Sentiment analysis
    dataset = pd.read_csv(csv_filename)
    dataset.drop(columns=['author', 'published_at', 'updated_at', 'like_count'], axis=1, inplace=True)
    sentiments = SentimentIntensityAnalyzer()
    dataset["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in dataset["text"]]
    dataset["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in dataset["text"]]
    dataset["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in dataset["text"]]
    dataset['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in dataset["text"]]
    score = dataset["Compound"].values
    sentiment = ['Positive' if i >= 0.05 else 'Negative' if i <= -0.05 else 'Neutral' for i in score]
    dataset["Sentiment"] = sentiment

    # Clean up the dataset
    data_new = dataset.drop(['Positive', 'Negative', 'Neutral', 'Compound'], axis=1)
    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemer = SnowballStemmer(language="english")

    def text_processing(text):
      if not text.strip():
          return ''

      if all(ord(char) < 128 for char in text):
          stop_words = set(stopwords.words('english'))
          tokens = word_tokenize(text)
          text = ' '.join([word for word in tokens if word not in stop_words])
          return text

      text = text.lower()
      text = re.sub(r'\n', ' ', text)
      text = re.sub('[%s]' % re.escape(punctuation), "", text)
      text = re.sub("^a-zA-Z0-9$,.", "", text)
      text = re.sub(r'\s+', ' ', text, flags=re.I)

      try:
          lang = detect(text)
      except:
          lang = 'en' 

      stop_words = set(stopwords.words('english'))
      tokens = word_tokenize(text)
      text = ' '.join([word for word in tokens if word not in stop_words])
      return text

    data_copy = data_new.copy()
    data_copy.Comment = data_copy.text.apply(
        lambda text: text_processing(text))
    le = LabelEncoder()
    data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])
    processed_data = {
        'Sentence': data_copy.Comment,
        'Sentiment': data_copy['Sentiment']
    }
    processed_data = pd.DataFrame(processed_data)
    df_neutral = processed_data[(processed_data['Sentiment'] == 1)]
    df_negative = processed_data[(processed_data['Sentiment'] == 0)]
    df_positive = processed_data[(processed_data['Sentiment'] == 2)]
    df_negative_upsampled = resample(df_negative,
                                     replace=True,
                                     n_samples=205,
                                     random_state=42)
    df_neutral_upsampled = resample(df_neutral,
                                    replace=True,
                                    n_samples=205,
                                    random_state=42)
    df_positive_upsampled = resample(df_neutral,
                                    replace=True,
                                    n_samples=205,
                                    random_state=42)

    final_data = pd.concat(
        [df_negative_upsampled, df_neutral_upsampled, df_positive])
    
    final_csv_filename = os.path.join(final_comments_folder, 'comments_final.csv')
    final_data.to_csv(final_csv_filename, index=False)
    print(f"Final data written to: {final_csv_filename}")

    corpus = []
    for comment in final_data['Sentence']:
        corpus.append(comment)
    corpus[0:5]
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = final_data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = le.classes_
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    line_chart_path = generate_line_chart(df, line_chart_folder, "line_chart.png")
    # pie_chart_path = generate_pie_chart(final_data, pie_chart_folder, "pie_chart.png")
    graph_path = generate_graph(final_data, graph_folder, "graph.png")
    
    comments = final_data['Sentence'].tolist()
    comment_sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
    average_sentiment = sum(comment_sentiments) / len(comment_sentiments)

    if average_sentiment > 0:
        sentiment_feedback = "Whooooo! The video is perceived positively."
    elif average_sentiment <0:
        sentiment_feedback = "Ooops! The video is perceived negatively."
    else:
         sentiment_feedback = "Hmmmm! The video is perceived as Neutral."
    
    return sentiment_feedback, line_chart_path, graph_path

# video_link = "https://www.youtube.com/watch?v=m-zbIfF-RYA&ab_channel=ShemarooComedy"
# feedback = analyze_youtube_comments(video_link)
# print(feedback)
