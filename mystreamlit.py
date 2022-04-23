import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
import pickle

def user_input():
  comment = st.text_input("enter your review")
  data={"comment": comment}
  features= pd.DataFrame(data,index=[0])
  return features

st.title("Sentiment Analysis - Foody.vn")
st.subheader("Phân loại review tích cực hay tiêu cực")
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css" integrity="sha384-zCbKRCUGaJDkqS1kPbPd7TveP5iyJE0EjAuZQTgFLD2ylzuqKfdKlfG/eSrtxUkn" crossorigin="anonymous">', unsafe_allow_html=True)

data = pd.read_csv('final_under_sampling.csv', encoding='utf-8')
data.comment=data.comment.astype(str)
data['rating'] = data['rating'].replace([1],"Positive")
data['rating'] = data['rating'].replace([0],"Negative")

X, y = train_test_split(data, test_size=.20, random_state=42)

tfidfconverter = TfidfVectorizer(min_df=0.002)
X_train = tfidfconverter.fit_transform(X['comment']).toarray()
X_test = tfidfconverter.transform(y['comment'])

# Put 'rating' column of each dataframe into y
y_train = np.asarray(X['rating'])
y_test = np.asarray(y['rating'])

multinomialNB = MultinomialNB(alpha=1)
multinomialNB.fit(X_train, y_train)

#st.write(dframe)
menu = ["Home", "About"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    query_params = st.experimental_get_query_params()
    tabs = ["Input text", "Upload file"]
    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "Input text"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="Input text")
        active_tab = "Input text"

    li_items = "".join(f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}" target="_self">{t}</a>
        </li>
        """
    for t in tabs)
    tabs_html = f"""
        <ul class="nav nav-tabs">
            { li_items }
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if active_tab == "Input text":
        dframe = user_input()
        X_single = tfidfconverter.transform(dframe['comment'])
        y_pred = multinomialNB.predict(X_single)

        st.write(y_pred)
    elif active_tab == "Upload file":
        filecsv = st.file_uploader("Choose a file...", type=['csv'])
        if filecsv is not None:
            df = pd.read_csv(filecsv)
            X_single = tfidfconverter.transform(df['comment'])
            y_pred = multinomialNB.predict(X_single)
            df['predict'] = y_pred
            st.write(df)
    else:
        st.error("Something has gone terribly wrong.")



