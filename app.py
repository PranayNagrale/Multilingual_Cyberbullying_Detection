import fasttext
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
import json
import requests

model = fasttext.load_model('fast_model.bin')

with open('rf_ft_model.pkl', 'rb') as f:
    clf = pickle.load(f, encoding='latin1')

with open('all_stopword.txt', 'r') as f:
      stop_words = json.load(f)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)




# Define the title
st.title("Multilingual Cyberbullying Detection")
st.write(
    "This is a Multilingual Cyberbullying Detection model which is built to detect cyberbullying or hate speeches.\
     It can detect the cyberbullying/hate speeches for English, Hindi(hindi & hindi-english code mixed), Marathi(marathi and marathi-english code mixed), Bengali and Tamil languages.\
     The Accuracy of this model is 92.75%, Precision is 92.97%, f1_score is 92.40% & recall_score is 91.86%.")


text = st.text_area("Enter the message")

if st.button('Predict'):
    # def output(text):
        if text in stop_words:
            st.header('Not Cyberbullying')
        elif text in string.punctuation:
            st.header('Not Cyberbullying')
        elif text.isdigit() or text.replace(".", "").isdigit():
            st.header('Not Cyberbullying')
        elif text not in stop_words and text not in string.punctuation:
            transformed_msg = transform_text(text)
            vector_input = model.get_sentence_vector(transformed_msg)
            pred =  clf.predict([vector_input])[0]
            if pred == 1:
                st.header('Cyberbullyinbg')
            else:
                st.header('Not Cyberbullying')

