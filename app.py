import streamlit as st
import pickle
from text_utils import clean_text

# Sidebar me GitHub username, profile link aur contact info
st.sidebar.markdown("### 👨‍💻 GitHub Username")
st.sidebar.write("[Dataahoaib](https://github.com/DataShoaib)")

st.sidebar.markdown("---")  # ek horizontal line

st.sidebar.markdown("📫 Contact me: mdshoaiba478@gmail.com")

# Model aur vectorizer load karna
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
