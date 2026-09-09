import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

genes = ['gene' + str(i) for i in range(1,101)]

wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]
print('genes =',genes, "\nwt =", wt, "\nko = ", ko)
data = pd.DataFrame(columns=[*wt, *ko], index=genes)

for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
print("\ndata head =", data.head())
print("\nshape = ", data.shape)

#feature print example
print(data.loc['gene1'])
a = data.loc['gene1']
for k in a.index:
    #b = a[]['wt1']
    print(a[k])

# features means
mean = {}

for gene in data.index:
    a = data.loc[gene]
    print(a)
    mean[gene] = sum(a)
    mean[gene] = mean[gene]/(data.shape[1])
print(mean)

# center features
center_data = data

for gene in center_data.index:
    a = center_data.loc[gene]
    for k in a.index:
        a[k] = a[k] - mean[gene]
    center_data.loc[gene] = a

print("\ncenter data head =", center_data.head())
print("\nshape = ", data.shape)

