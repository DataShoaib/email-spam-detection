import streamlit as st
import pickle
from text_utils import clean_text
import nltk
from nltk.data import find
from nltk import download

def ensure_nltk_dependencies():
    try:
        find('tokenizers/punkt')
    except LookupError:
        download('punkt')

    try:
        find('corpora/stopwords')
    except LookupError:
        download('stopwords')

ensure_nltk_dependencies()

st.sidebar.markdown("### 👨‍💻 GitHub Username")
st.sidebar.write("[Dataahoaib](https://github.com/DataShoaib)")
st.sidebar.markdown("---")
st.sidebar.markdown("📫 Contact me: mdshoaiba478@gmail.com")

with open('email_spam_detection.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidfv-vectorizer.pkl', 'rb') as file:
    tfidfv = pickle.load(file)

st.title("📧 Email Spam Detection with Advanced Preprocessing")

input_text = st.text_area("Enter your email message:", height=200)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(input_text)
        vectorized_text = tfidfv.transform([cleaned_text])
        prediction = model.predict(vectorized_text)

        if prediction[0] == 1:
            st.error("❌ This is a SPAM email!")
        else:
            st.success("✅ This is NOT a spam email.")
