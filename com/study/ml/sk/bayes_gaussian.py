from sklearn.naive_bayes import GaussianNB

from sklearn import datasets
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score




if __name__ == '__main__':
    '''
    c = [[5, 10]]  # c=[[a,b]],这里要注意a的shape，如果是list形式，则将a.shape=-1,1
    pl = PolynomialFeatures(degree=2, include_bias=False)
    b = pl.fit_transform(c)
    print(b)
    '''



    iris = datasets.load_iris()

    print(iris.target_names)
    x_train, x_test, y_train, y_test =\
        train_test_split(iris.data, iris.target, train_size=0.8)
    gaos_nb = Pipeline([
        ("sc", StandardScaler()),
        ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ("nb", GaussianNB())
    ])
    gaos_nb.fit(x_train, y_train)
    y_pred = gaos_nb.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)







