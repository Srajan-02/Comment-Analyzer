# -*- coding: utf-8 -*-
"""bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s25kAQr3hVILTjcL6ncdrg8RZbV11ODg
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

!pip install vadersentiment

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

corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.eval()

input_ids = []
attention_masks = []

for comment in final_data['Sentence']:
    encoded_dict = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Move tensors to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)

# Create DataLoader for BERT
batch_size = 32
prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Predict using BERT model
predictions = []
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    predictions.extend(np.argmax(logits, axis=1))

# Generate confusion matrix
y_pred_bert = predictions
y_test = final_data['Sentiment'].tolist()

# Generate confusion matrix
cm_bert = confusion_matrix(y_test, y_pred_bert)

# Your existing code for plotting confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm_bert, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (BERT)')
plt.colorbar()
classes = le.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm_bert.max() / 2.
for i, j in itertools.product(range(cm_bert.shape[0]), range(cm_bert.shape[1])):
    plt.text(j, i, format(cm_bert[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_bert[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

from  textblob import TextBlob

comments = final_data['Sentence'].tolist()
comment_sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
average_sentiment = sum(comment_sentiments) / len(comment_sentiments)

if average_sentiment > 0:
    print("Whooooo! The video is perceived positively.")
elif average_sentiment < 0:
    print("Ooops! The video is perceived negatively.")
else:
    print("Hmmmm! The video is perceived as Neutral.")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import itertools

precision = precision_score(y_test, y_pred_bert, average='weighted')
recall = recall_score(y_test, y_pred_bert, average='weighted')
f1 = f1_score(y_test, y_pred_bert, average='weighted')
accuracy = accuracy_score(y_test, y_pred_bert)

metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
values = [precision, recall, f1, accuracy]

plt.figure(figsize=(8, 6))
plt.plot(metrics, values, marker='o', color='blue', linestyle='-')
plt.title('Metrics Comparison')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

nb_score = accuracy_score(y_test, y_pred_bert)
print('accuracy',nb_score)