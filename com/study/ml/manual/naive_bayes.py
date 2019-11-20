import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.naive_bayes import BernoulliNB
def load_data_set():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(sen):
    global stop_words
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def create_word_lexicon(df):
    global stop_words
    lexicon = set()
    for column in df:
        clean_sentences = pd.Series(column).str.replace("[^a-zA-Z]", " ")
        clean_sentences = [s.lower() for s in clean_sentences]
        clean_sentences = [r for r in clean_sentences if r not in stop_words]

        lexicon.update(clean_sentences)
    return lexicon

def build_vec_matrix(lexicon, df):
    global stop_words
    lexicon_df = np.zeros((len(lexicon), len(lexicon)))
    for column in df:
        i = 0
        clean_sentences = pd.Series(column).str.replace("[^a-zA-Z]", " ")
        clean_sentences = [s.lower() for s in clean_sentences]
        clean_sentences = [r for r in clean_sentences if r not in stop_words]
        for word in clean_sentences:
            j = 0
            if word in lexicon:
                lexicon_df[i][j] += 1
            j +=1
        i += 1
    word_probability = {}
    lexicon_df.columns = lexicon
    for word_index in range(len(lexicon)):
        word_probability[lexicon_df.columns[word_index]] = sum(lexicon_df[lexicon_df != 0])


if __name__ == '__main__':
    '''
    postingList, classVec = load_data_set()
    df = pd.DataFrame({"text":postingList, "label":classVec})
    lexicon = create_word_lexicon(df["text"])
    print(lexicon)
    '''
    postingList, classVec = load_data_set()
    clf = BernoulliNB()
    X = np.random.randint(2, size=(6, 100))
    Y = np.array([1, 0, 1, 0, 1, 0])
    clf.fit(X, Y)
    print(clf.predict(X[2:3]))

