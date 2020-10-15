from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from csv import reader
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

if __name__ == '__main__':

    dataset = list()
    path = "./sonar.all-data.csv"
    df = pd.read_csv(path, header=None)
    Y = df.iloc[:,-1]
    X = df.iloc[:, :len(df.columns) - 1]
    kf = KFold(n_splits=5)

    dtree = DecisionTreeClassifier(criterion="gini", max_depth=20)

    #train_X, train_y = X.iloc[train_index,:], Y[train_index]
    scores = cross_val_score(dtree, X, Y)
    print(scores)

    # make_classification(n_samples=len(X), n_features=df.columns - 1,
    # n_informative = 2, n_redundant = 0,
    # random_state = 0, shuffle = False)



    for n_trees in [1, 5, 10, 50, 150]:
        model = RandomForestClassifier(max_depth=17,
                                  n_estimators=n_trees)
        scores = cross_val_score(model, X, Y)
        print(scores)





