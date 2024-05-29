# Model training and saving script
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess data
df = pd.read_csv("emotions.csv")
df = df.sample(50000)
df['label'] = df['label'].replace({1: 1, 2: 1, 5: 1, 0: 0, 4: 0, 3: 0})
df.drop_duplicates(keep='first', inplace=True)

nltk.download('punkt')
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemming = PorterStemmer()

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

df['text'] = df['text'].apply(preprocessing)

# Vectorize text data
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate model
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
