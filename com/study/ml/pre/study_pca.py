from sklearn import datasets
import numpy  as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    lris_df = datasets.load_iris()
    x_axis = lris_df.data[:,0]
    y_axis = lris_df.data[:,1]

    #plt.scatter(x_axis, y_axis, c=lris_df.target)
    #plt.scatter(x_axis, y_axis, c=lris_df.target)
    #plt.show()


    from sklearn.preprocessing import StandardScaler

    x_axis = x_axis.reshape(1, -1)
    X_std = StandardScaler().fit_transform(x_axis)

    mean_vec = np.mean(X_std, axis=0)
    #voc_mat Covariance matrix
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    #voc_mat = np.cov(X_std.T)
    cov_mat = np.cov(X_std.T)
    eig_vals,eig_vecs = np.linalg.eig(cov_mat) #获取特征值,特征向量
    #特征值,特征向量对应 做成tuple
    eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals)) ]
    eig_pairs.sort(key=lambda x:x[0], reverse=True)

    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                          eig_pairs[1][1].reshape(4,1)))

    Y = X_std.dot(matrix_w)

