from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
import numpy as np

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


'''
StandardScaler 不能对spare matrix使用

'''

if __name__ == '__main__':
    '''
    c = [[5, 10]]  # c=[[a,b]],这里要注意a的shape，如果是list形式，则将a.shape=-1,1
    pl = PolynomialFeatures(degree=2, include_bias=False)
    b = pl.fit_transform(c)
    print(b)
    '''



    data = datasets.fetch_20newsgroups()

    clean_sentences = pd.Series(data.data).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    vectorizer = TfidfVectorizer(input="word",
                                 max_df=0.5, sublinear_tf=True)
    #count_vect = CountVectorizer()
    #X_train_counts = count_vect.fit_transform(clean_sentences)

    train_data = vectorizer.fit_transform(clean_sentences)
    #特征选取
    ch2 = SelectKBest(chi2, k = 300)
    train_data = ch2.fit_transform(train_data, data.target)
    x_train, x_test, y_train, y_test =\
        train_test_split(train_data, data.target, train_size=0.8)
    gaos_nb = Pipeline([
     #   ("sc", StandardScaler()),
     #   ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ("nb", GaussianNB())
    ])
    gaos_nb.fit(x_train, y_train)
    y_pred = gaos_nb.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)







