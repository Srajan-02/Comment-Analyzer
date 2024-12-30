import uuid
import glob
import os
import re
import nltk
import pandas as pd
import numpy as np
import googleapiclient.discovery
from urllib.parse import urlparse, parse_qs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

folder_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Github\\Comment-Analyzer\\Comment_analyzer\\Files"

def generate_unique_filename(folder, extension=".csv"):
    unique_filename = str(uuid.uuid4()) + extension
    return os.path.join(folder, unique_filename)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    return text

def analyze_youtube_comments(video_link):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyC1YUwsMt-dRKwU1WRJ9YUZZewj7eNjVZQ"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)
    parsed_url = urlparse(video_link)
    video_id = parse_qs(parsed_url.query).get("v")
    if video_id:
        video_id = video_id[0]

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=1000000
        )
        response = request.execute()
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])
        df = pd.DataFrame(comments, columns=[
            'author', 'published_at', 'updated_at', 'like_count', 'text'])

        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"Removed: {csv_file}")
            except Exception as e:
                print(f"Error removing {csv_file}: {e}")

        csv_filename = generate_unique_filename(folder_path)
        df.to_csv(csv_filename, index=False)
    else:
        print("Invalid YouTube video link.")

    dataset = pd.read_csv(csv_filename)
    dataset.drop(columns='author', axis=1, inplace=True)
    dataset.drop(columns='published_at', axis=1, inplace=True)
    dataset.drop(columns='updated_at', axis=1, inplace=True)
    dataset.drop(columns='like_count', axis=1, inplace=True)

    sentiments = SentimentIntensityAnalyzer()
    dataset["Compound"] = [sentiments.polarity_scores(
        i)["compound"] for i in dataset["text"]]

    score = dataset["Compound"].values
    sentiment = ['Positive' if s >= 0.05 else 'Negative' if s <= -0.05 else 'Neutral' for s in score]

    dataset["Sentiment"] = sentiment

    # Text Preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    dataset['text'] = dataset['text'].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(dataset['text'])
    X = tokenizer.texts_to_sequences(dataset['text'])
    X = pad_sequences(X, maxlen=100)

    # Encoding sentiment labels
    label_encoder = LabelEncoder()
    dataset['Sentiment'] = label_encoder.fit_transform(dataset['Sentiment'])

    y = dataset['Sentiment']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LSTM Model
    model = Sequential()
    model.add(Embedding(5000, 128))  # Remove input_length parameter
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=2)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy:", accuracy)

    # Providing feedback based on average sentiment score
    average_score = np.mean(score)
    if average_score > 0.05:
        sentiment_feedback = "Whooooo! The video is perceived positively."
    elif average_score < -0.05:
        sentiment_feedback = "Ooops! The video is perceived negatively."
    else:
        sentiment_feedback = "Hmmmm! The video is perceived as Neutral."
    
    return sentiment_feedback, accuracy

video_link = "https://www.youtube.com/watch?v=X0tOpBuYasI&t=74s&pp=ygUYYmxhY2sgYWRhbSBtb3ZpZSB0cmFpbGVy"
sentiment_feedback, accuracy = analyze_youtube_comments(video_link)
print("Sentiment Feedback:", sentiment_feedback)
print("Accuracy:", accuracy)
