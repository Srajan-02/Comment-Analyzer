# -*- coding: utf-8 -*-
"""KNN_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DfuuSiSXz1RUtSg2GOCQh-cHn12LH5Eg
"""

# Commented out IPython magic to ensure Python compatibility.
import googleapiclient.discovery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from urllib.parse import urlparse, parse_qs

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyC1YUwsMt-dRKwU1WRJ9YUZZewj7eNjVZQ"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

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

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

    csv_filename = "youtube_comments.csv"
    df.to_csv(csv_filename, index=False)
else:
    print("Invalid YouTube video link.")

dataset = pd.read_csv('youtube_comments.csv')

dataset.drop(columns = 'author',axis = 1, inplace = True)
dataset.drop(columns = 'published_at',axis = 1, inplace = True)
dataset.drop(columns = 'updated_at',axis = 1, inplace = True)
dataset.drop(columns = 'like_count',axis = 1, inplace = True)

!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import the SentimentIntensityAnalyzer class
import nltk
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
dataset["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in dataset["text"]]
dataset["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in dataset["text"]]
dataset["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in dataset["text"]]
dataset['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in dataset["text"]]
score = dataset["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
dataset["Sentiment"] = sentiment
dataset.head()

data_new=dataset.drop(['Positive','Negative','Neutral','Compound'],axis=1)
data_new.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation

import nltk
import re
nltk.download('stopwords')
stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemer = SnowballStemmer(language="english")
lzr = WordNetLemmatizer()

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

import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
data_copy = data_new.copy()
data_copy.Comment = data_copy.text.apply(lambda text: text_processing(text))

le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

processed_data = {
    'Sentence':data_copy.Comment,
    'Sentiment':data_copy['Sentiment']
}

processed_data = pd.DataFrame(processed_data)
processed_data.head()

processed_data['Sentiment'].value_counts()

df_neutral = processed_data[(processed_data['Sentiment']==1)]
df_negative = processed_data[(processed_data['Sentiment']==0)]
df_positive = processed_data[(processed_data['Sentiment']==2)]

df_negative_upsampled = resample(df_negative,
                                 replace=True,
                                 n_samples= 205,
                                 random_state=42)

df_neutral_upsampled = resample(df_neutral,
                                 replace=True,
                                 n_samples= 205,
                                 random_state=42)

final_data = pd.concat([df_negative_upsampled,df_neutral_upsampled,df_positive])

final_data['Sentiment'].value_counts()

corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

from textblob import TextBlob
comments = [
    "This video is awesome!",
    "I didn't like this video.",
]
comment_sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
average_sentiment = sum(comment_sentiments) / len(comment_sentiments)
if average_sentiment > 0:
    print("Overall sentiment: Positive  The video is perceived positively.")
elif average_sentiment < 0:
    print("Overall sentiment: Negative - The video is perceived negatively.")
else:
    print("Overall sentiment: Neutral - The sentiment towards the video is neutral.")

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
# Calculate precision, recall, and f1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (KNN)')
plt.colorbar()
classes = np.unique(y_test)
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
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import itertools

# Calculate precision, recall, f1-score, and accuracy
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Plotting
metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
values = [precision, recall, f1, accuracy]

plt.figure(figsize=(8, 6))
plt.plot(metrics, values, marker='o', color='blue', linestyle='-')
plt.title('Metrics Comparison (KNN)')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()