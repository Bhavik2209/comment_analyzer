import streamlit as st
import plotly.graph_objects as go
import joblib
import re
from wordcloud import WordCloud
import nltk
import string
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load saved model and vectorizer
lr = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

try:
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading NLTK stopwords: {e}")

lemmatizer = nltk.stem.WordNetLemmatizer()
stemming = nltk.stem.PorterStemmer()

def preprocessing(text):
    preprocessed_text = ""
    sentences = nltk.sent_tokenize(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word not in stopwords_set]
        words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
        words = [word for word in words if word.isalnum()]
        words = [lemmatizer.lemmatize(word, pos='v') for word in words]
        words = [stemming.stem(word) for word in words]
        preprocessed_text += " ".join(words) + " "
    return preprocessed_text.strip()

def extract_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def get_youtube_client():
    api_key = st.secrets["default"]['API_KEY']
    return build('youtube', 'v3', developerKey=api_key)

def fetch_comments(video_id):
    youtube = get_youtube_client()
    try:
        comments = []
        nextPageToken = None
        max_comments = 200
        fetched_comments = 0
        while True:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
                comments.append(comment)
                fetched_comments += 1
                if fetched_comments >= max_comments:
                    break

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

        return comments
    except HttpError as e:
        st.error(f"An error occurred: {e}")
        return []

def vader_sentiment_analysis(comments):
    analyzer = SentimentIntensityAnalyzer()
    positive_comments = []
    neutral_comments = []
    negative_comments = []

    for comment in comments:
        score = analyzer.polarity_scores(comment)
        if score['compound'] >= 0.05:
            positive_comments.append(comment)
        elif score['compound'] <= -0.05:
            negative_comments.append(comment)
        else:
            neutral_comments.append(comment)

    return positive_comments, neutral_comments, negative_comments

def analyze_sentiment(comments):
    polarities = []
    positive_comments = []
    negative_comments = []
    for comment in comments:
        preprocessed_comment = preprocessing(comment)
        transformed_comment = tfidf.transform([preprocessed_comment])
        prediction = lr.predict(transformed_comment)[0]
        
        # Debug prints
        print(f"Original: {comment}")
        print(f"Preprocessed: {preprocessed_comment}")
        print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")

        # Assign comment to appropriate category
        polarities.append(prediction)
        if prediction == 1:
            positive_comments.append(comment)
        else:
            negative_comments.append(comment)

    return polarities, positive_comments, negative_comments

def generate_word_cloud(comments):
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def main():
    st.title('YouTube Comment Sentiment Analysis')
    st.write("Version 2.0.0", unsafe_allow_html=True)
    st.subheader('Enter YouTube Video URL:')

    url = st.text_input("Enter the URL:")
    video_id = extract_video_id(url)
    if st.button('Analyze'):
        if video_id:
            with st.spinner('Fetching comments...'):
                comments = fetch_comments(video_id)
            
            if comments:
                polarities, positive_comments, negative_comments = analyze_sentiment(comments)
                positive_comments_vader, neutral_comments_vader, negative_comments_vader = vader_sentiment_analysis(comments)

                avg_polarity = sum(polarities) / len(polarities)
                st.write(f'Average Polarity: {avg_polarity}')
                if avg_polarity > 0.05:
                    st.write('The Video has got a Positive response')
                elif avg_polarity < 0:
                    st.write('The Video has got a Negative response')
                else:
                    st.write('The Video has got a Neutral response')

                st.subheader('Word Cloud of Comments')
                wordcloud = generate_word_cloud(comments)
                st.image(wordcloud.to_array(), use_column_width=True)

                st.subheader('Top 5 Positive Comments')
                for i, comment in enumerate(positive_comments[:5], 1):
                    st.write(f"{i}. {comment}")

                st.subheader('Top 5 Negative Comments')
                for i, comment in enumerate(negative_comments[:5], 1):
                    st.write(f"{i}. {comment}")

                fig = go.Figure(data=[go.Bar(x=['Positive', 'Neutral', 'Negative'],
                                             y=[len(positive_comments_vader), len(neutral_comments_vader), len(negative_comments_vader)],
                                             marker=dict(color=['#1f77b4', '#aec7e8', '#d62728']))])
                fig.update_layout(title='Sentiment Analysis of Comments (VADER)', xaxis_title='Sentiment',
                                  yaxis_title='Comment Count')
                st.plotly_chart(fig)

                fig_pie = go.Figure(data=[go.Pie(labels=['Positive', 'Neutral', 'Negative'],
                                                 values=[len(positive_comments_vader), len(neutral_comments_vader), len(negative_comments_vader)],
                                                 marker=dict(colors=['#1f77b4', '#aec7e8', '#d62728']))])
                fig_pie.update_layout(title='Sentiment Analysis Pie Chart (VADER)',xaxis_title='Sentiment',
                                    yaxis_title='Comment Count')
                st.plotly_chart(fig_pie)

            else:
                st.write("No comments found or an error occurred. Please try again later.")
        else:
            st.write("Please enter a valid URL.")

if __name__ == '__main__':
    main()
