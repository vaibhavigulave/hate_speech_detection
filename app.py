app_code = '''
import streamlit as st
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'\\[.*?\\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'\\w*\\d\\w*', '', text)

    text = [word for word in text.split() if word not in stop_words]
    text = " ".join(text)

    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)

    return text

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Twitter Hate Speech Detection")

user_input = st.text_area("Enter Tweet")

if st.button("Predict"):
    cleaned = clean_data(user_input)
    vector = cv.transform([cleaned]).toarray()
    result = model.predict(vector)

    st.success(f"Prediction: {result[0]}")
'''

with open("app.py", "w") as file:
    file.write(app_code)

print("app.py created successfully")
