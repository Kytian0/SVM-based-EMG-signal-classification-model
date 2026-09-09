import numpy as np
from sklearn.decomposition import PCA

a = np.array([[1,2,3],[3,4,5],[6,7,8],[6,7,8]])
b = a[2:,:]
print("b",b)
n_components = 2
pca_model = PCA(n_components=n_components)
pca_fit = pca_model.fit(a)
a_transformed= pca_fit.transform(a)
pca_matrix = pca_model.components_
mean = pca_model.mean_
cov = pca_model.get_covariance()
print("covariance,mean", (cov,mean))
a_tr = (a - mean)
# a_mult = np.matmul(a -np.mean(a,axis=0),pca_matrix.T)
a_mult = np.matmul(a - mean,pca_matrix.T)
print("a", a)
print("a shape", a.shape)
print("a cov", np.cov(a))
print("pca matrix", pca_matrix)
print("pca matrix shape", pca_matrix.shape)
print("a_transformed", a_transformed)
print("a_mult", a_mult)

b[0,0] = 42.0 # answer to everything :v
b_transformed = pca_fit.transform(b)
b_tr = (b - mean)
print("b no mean:", b_tr)
b_mult = np.matmul(b - mean,pca_matrix.T)
print("b_transformed", b_transformed)
print("b_mult", b_mult)