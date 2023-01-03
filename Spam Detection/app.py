import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from flask import Flask ,render_template,request

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


app = Flask(__name__)
@app.route('/')

def hello():
    return render_template('home.html')


@app.route('/',methods=['GET','POST'])
def predict():
    data=request.form['message']
    data=[transform_text(data)]
    vector=tfidf.transform(data).toarray()
    pred=model.predict(vector)
    return render_template('result.html',prediction=pred)

  

if __name__=='__main__':
  app.run(debug=True)