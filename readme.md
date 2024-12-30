# YouTube Comments Sentiment Analysis

A Python-based sentiment analysis tool that analyzes comments from YouTube videos using Natural Language Processing (NLP) and Machine Learning techniques. The project uses NLTK and scikit-learn to classify comments as positive, negative, or neutral, with a Flask web interface for easy interaction.

## Features

- YouTube API integration for comment extraction (up to 3000 comments per video)
- Multi-language support with automatic language detection
- Sentiment analysis using VADER and TextBlob
- Advanced text preprocessing pipeline
- Machine learning classification using Gaussian Naive Bayes
- Data visualization with line charts and sentiment distribution graphs
- Web interface built with Flask
- Automatic file management with unique identifiers

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- flask
- nltk
- textblob
- scikit-learn
- google-api-python-client
- pandas
- numpy
- matplotlib
- vaderSentiment
- langdetect
- indic-nlp-library
- uuid

## Project Structure

```
project/
│
├── app.py                 # Flask application main file
├── model.py              # Sentiment analysis model
├── static/            
│   ├── Images01/         # Static images
│   ├── Line_chart/       # Generated line charts
│   └── Graph_chart/      # Generated graphs
├── templates/
│   └── index.html        # Main web interface template
├── Files/
│   └── final/           # Processed comments storage
└── requirements.txt
```

## Detailed Implementation

### 1. Text Processing Pipeline

```python
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
```

### 2. Sentiment Analysis Process

The sentiment analysis follows these steps:

1. **Comment Extraction**:
   - Uses YouTube Data API v3
   - Supports pagination for fetching up to 3000 comments
   - Stores author, publish date, likes, and comment text

2. **Preprocessing**:
   - Language detection
   - Text cleaning (punctuation removal, lowercasing)
   - Stop word removal
   - Tokenization
   - Special character handling

3. **Sentiment Scoring**:
   - VADER Sentiment analysis for initial scoring
   - Compound score calculation
   - Classification into Positive, Negative, or Neutral
   - Additional TextBlob analysis for verification

4. **Machine Learning Classification**:
   - Feature extraction using CountVectorizer
   - Training Gaussian Naive Bayes classifier
   - Confusion matrix generation
   - Performance evaluation

### 3. Visualization Generation

```python
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
    
    return os.path.join(file_path, default_line_chart_name)
```

### 4. Data Balancing

The project implements data balancing using upsampling:
```python
df_negative_upsampled = resample(df_negative,
                                replace=True,
                                n_samples=205,
                                random_state=42)
df_neutral_upsampled = resample(df_neutral,
                               replace=True,
                               n_samples=205,
                               random_state=42)
```

## Setup and Configuration

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up YouTube API:
   - Create a Google Cloud Project
   - Enable YouTube Data API v3
   - Generate API key
   - Replace in code:
```python
DEVELOPER_KEY = "YOUR_API_KEY_HERE"
```

4. Configure file paths in model.py:
```python
folder_path = "path/to/your/project/Files"
final_comments_folder = "path/to/your/project/Files/final"
line_chart_folder = "path/to/your/project/static/Line_chart"
graph_folder = "path/to/your/project/static/Graph_chart"
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open browser at:
```
http://localhost:5000
```

3. Enter YouTube video URL and submit

## Output Format

The tool provides:
- Sentiment analysis result ("Positive", "Negative", or "Neutral")
- Line chart showing comment frequency over time
- Bar graph showing sentiment distribution
- Processed comments CSV file
- Confusion matrix for model evaluation

## UI Screenshots

[Note: Please add your UI screenshots here. You can use the following format:]

![UI Screenshot 1](path/to/screenshot1.png)
*Caption: Main interface for entering YouTube URL*

![UI Screenshot 2](path/to/screenshot2.png)
*Caption: Sentiment analysis results visualization*

## Future Improvements

- Deep learning models implementation
- Real-time comment analysis
- Extended language support
- Interactive visualization dashboard
- Batch video processing
- Sentiment trend analysis over time
- API endpoint for programmatic access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK team for NLP tools
- YouTube Data API team
- scikit-learn community
- Flask framework developers
- Contributors to visualization libraries
- VADER sentiment analysis team

