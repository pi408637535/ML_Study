
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

if __name__ == '__main__':
    spam_csv = "/Users/piguanghua/Downloads/spam.csv"
    df = pd.read_csv(spam_csv, usecols=[0, 1], encoding='latin-1')
    df.columns = ["label", "text"]

    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    # flatten the list
    sentences = [x for x in df["text"].tolist()]

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    x_train, x_test, y_train, y_test = train_test_split(clean_sentences, df["label"], test_size=0.2, random_state=0)



