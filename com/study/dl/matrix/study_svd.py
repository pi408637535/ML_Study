import numpy as np
from scipy.linalg import  svd

if __name__ == '__main__':
    A = np.array([
        [1,2],
        [3,4],
        [5,6]
    ])
    U, s, VT = svd(A)
    Sigma = np.zeros(A.shape)
    for i in range(len(s)):
        Sigma[i][i] = s[i]


