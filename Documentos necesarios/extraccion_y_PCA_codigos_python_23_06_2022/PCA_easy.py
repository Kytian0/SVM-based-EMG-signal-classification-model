import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
# matplotlib inline
# source https://programmathically.com/principal-components-analysis-explained-for-dummies/
digits = load_digits()
digits.data.shape # (1797, 64)

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(8, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 10))
    plt.show()
#run the function
plot_digits(digits.data)

# pca = PCA(n_components=2)  # project from 64 to 2 dimensions
# reduced = pca.fit_transform(digits.data)
# print(reduced.shape) #(1797, 2)


pca = PCA(n_components=2)  # project from 64 to 2 dimensions
fit_pca = pca.fit(digits.data)
reduced = fit_pca.transform(digits.data)
print(reduced.shape)

plt.scatter(reduced[:, 0], reduced[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Accent', 10))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar()
plt.show()


pca = PCA().fit(digits.data)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('explained variance');

plt.show()