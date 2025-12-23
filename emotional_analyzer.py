import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data.csv")

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Clean data
data['clean_text'] = data['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# -------- USER INPUT --------
user_text = input("\nEnter a sentence to analyze sentiment: ")

cleaned_input = clean_text(user_text)
input_vector = vectorizer.transform([cleaned_input])

prediction = model.predict(input_vector)

print("\nPredicted Sentiment:", prediction[0])




