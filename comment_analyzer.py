import nltk
from textblob import TextBlob
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import googleapiclient.discovery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from urllib.parse import urlparse, parse_qs

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "YOUR_API_KEY"
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

video_link = input("Enter the YouTube Video Link: ")

parsed_url = urlparse(video_link)
video_id = parse_qs(parsed_url.query).get("v")
if video_id:
    video_id = video_id[0]

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=500
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

    csv_filename = "youtube_comments.csv"
    df.to_csv(csv_filename, index=False)
else:
    print("Invalid YouTube video link.")

dataset = pd.read_csv('youtube_comments.csv')

dataset.shape

dataset.head(10)
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
nltk.download('stopwords')
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


nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
data_copy = data_new.copy()
data_copy.Comment = data_copy.text.apply(lambda text: text_processing(text))

le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

processed_data = {
    'Sentence': data_copy.Comment,
    'Sentiment': data_copy['Sentiment']
}

processed_data = pd.DataFrame(processed_data)
processed_data.head()

processed_data['Sentiment'].value_counts()

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

final_data['Sentiment'].value_counts()

corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

nb_score = accuracy_score(y_test, y_pred)
print('accuracy', nb_score)

filename = 'comment_analyzer.sav'
pickle.dump(classifier, open(filename, 'wb'))
loaded_model = pickle.load(open('comment_analyzer.sav', 'rb'))

comments = [
    "This video is awesome!",
    "I didn't like this video.",
]
comment_sentiments = [
    TextBlob(comment).sentiment.polarity for comment in comments]
average_sentiment = sum(comment_sentiments) / len(comment_sentiments)
if average_sentiment > 0:
    print("Overall sentiment: Positive - The video is perceived positively.")
elif average_sentiment < 0:
    print("Overall sentiment: Negative - The video is perceived negatively.")
else:
    print("Overall sentiment: Neutral - The sentiment towards the video is neutral.")
