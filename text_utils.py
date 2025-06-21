
import string
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
# Removing html tags
    text = BeautifulSoup(text, "html.parser").get_text()

# Lowercase
    text = text.lower()
#Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))#Tokenize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
# Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

#Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]

    return " ".join(stemmed)