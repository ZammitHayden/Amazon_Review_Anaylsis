from bs4 import BeautifulSoup
from nltk import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

def ScrapeReviews(url):
    """Scrape reviews from a webpage"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.findAll("span", class_="a-size-base review-text")

def TokenizeSentences(text):
    """Tokenize text into sentences"""
    return sent_tokenize(text)

def AnalyzeSentenceSentiment(sentence):
    """Analyze the sentiment of a sentence"""
    sentim_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentim_analyzer.polarity_scores(sentence)
    if sentiment_score['compound'] > 0.05:
        return "Positive"
    elif sentiment_score['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

def ImportantWords(review_text):
    """Extract important words from a review using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([review_text])
    feature_names = vectorizer.get_feature_names_out()
    important_words = []
    for col in tfidf.nonzero()[1]:
        important_words.append(feature_names[col])
    return important_words

def AnalyzeSentiment(url):
    """Analyze the sentiment of a webpage"""
    reviews = ScrapeReviews(url)
    results = []

    for review in reviews:
        sentences = TokenizeSentences(review.text)
        important_sentences = []
        result = ""
        important_words = ImportantWords(review.text)

        for sentence in sentences:
            sentiment = AnalyzeSentenceSentiment(sentence)
            if sentiment!= "Neutral":
                important_sentences.append(sentence)
            result = sentiment

        results.append({
            "important_sentences": important_sentences,
            "important_words": important_words,
            "result": result
        })

    return results

@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/submit', methods=['GET'])
def submit():
    url = request.args.get('url')
    if url:
        results = AnalyzeSentiment(url)
    else:
        results = None
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)