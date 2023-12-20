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
import os
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

folder_path = "C:\\Users\\india\\OneDrive\\Documents\\SRAJAN\\ML projects\\comment_analyzer\\Files\\"


def generate_unique_filename(folder, extension=".csv"):
    unique_filename = str(uuid.uuid4()) + extension
    return os.path.join(folder, unique_filename)\



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
            maxResults=1000
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
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    dataset["Positive"] = [sentiments.polarity_scores(
        i)["pos"] for i in dataset["text"]]
    dataset["Negative"] = [sentiments.polarity_scores(
        i)["neg"] for i in dataset["text"]]
    dataset["Neutral"] = [sentiments.polarity_scores(
        i)["neu"] for i in dataset["text"]]
    dataset['Compound'] = [sentiments.polarity_scores(
        i)["compound"] for i in dataset["text"]]
    score = dataset["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05:
            sentiment.append('Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')

    dataset["Sentiment"] = sentiment
    data_new = dataset.drop(
        ['Positive', 'Negative', 'Neutral', 'Compound'], axis=1)
    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemer = SnowballStemmer(language="english")
    lzr = WordNetLemmatizer()

    def text_processing(text):
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub('[%s]' % re.escape(punctuation), "", text)
        text = re.sub("^a-zA-Z0-9$,.", "", text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub(r'\W', ' ', text)
        text = ' '.join([word for word in word_tokenize(text)
                        if word not in stop_words])
        text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
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
    final_data = pd.concat(
        [df_negative_upsampled, df_neutral_upsampled, df_positive])
    corpus = []
    for sentence in final_data['Sentence']:
        corpus.append(sentence)
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
    from textblob import TextBlob
    comments = [
        "This video is awesome!",
        "I didn't like this video.",
    ]
    comment_sentiments = [
        TextBlob(comment).sentiment.polarity for comment in comments]
    average_sentiment = sum(comment_sentiments) / len(comment_sentiments)
    if average_sentiment > 0:
        return "Whooooo! The video is perceived positively."
    elif average_sentiment < 0:
        return "Ooops! The video is perceived negatively."
    else:
        return "Hmmmm! The video is perceived as Neutral."
