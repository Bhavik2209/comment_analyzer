import streamlit as st
import plotly.graph_objects as go
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from wordcloud import WordCloud


API_KEY = st.secrets["default"]['API_KEY'] 
youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def fetch_comments(video_id):
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

def analyze_sentiment(comments):
    analyzer = SentimentIntensityAnalyzer()
    polarities = []
    positive_comments = []
    negative_comments = []
    neutral_comments = []

    for comment in comments:
        polarity = analyzer.polarity_scores(comment)['compound']
        polarities.append(polarity)
        if polarity > 0.05:
            positive_comments.append(comment)
        elif polarity < -0.05:
            negative_comments.append(comment)
        else:
            neutral_comments.append(comment)

    return polarities, positive_comments, negative_comments, neutral_comments

def generate_word_cloud(comments):
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud



def main():
    st.title('YouTube Comment Sentiment Analysis')
    st.subheader('Enter YouTube Video URL:')
    url = st.text_input("enter the URL :")
    video_id = extract_video_id(url)
    if st.button('Analyze'):
        if video_id:
            comments = fetch_comments(video_id)
            polarities, positive_comments, negative_comments, neutral_comments = analyze_sentiment(comments)
            avg_polarity = sum(polarities) / len(polarities)
            st.write(f'Average Polarity: {avg_polarity}')
            if avg_polarity > 0.05:
                st.write('The Video has got a Positive response')
            elif avg_polarity < 0:
                st.write('The Video has got a Negative response')
            else:
                st.write('The Video has got a Neutral response')

            # Display word cloud
            st.subheader('Word Cloud of Comments')
            wordcloud = generate_word_cloud(comments)
            st.image(wordcloud.to_array(), use_column_width=True)
            st.write(positive_comments[:5])
            st.write(negative_comments[:5])
            # Plot sentiment analysis
            fig = go.Figure(data=[go.Bar(x=['Positive', 'Negative', 'Neutral'],
                                         y=[len(positive_comments), len(negative_comments), len(neutral_comments)],
                                         marker=dict(color=['green', 'red', 'grey']))])
            fig.update_layout(title='Sentiment Analysis of Comments', xaxis_title='Sentiment',
                              yaxis_title='Comment Count')
            st.plotly_chart(fig)

            # Plot sentiment analysis pie chart
            fig_pie = go.Figure(data=[go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                             values=[len(positive_comments), len(negative_comments),
                                                     len(neutral_comments)])])
            fig_pie.update_layout(title='Sentiment Analysis Pie Chart')
            st.plotly_chart(fig_pie)
        else:
            st.write("Give valid URL")

if __name__ == '__main__':
    
    main()
