import numpy as np
from scipy.special import factorial
class SVM_aprox:
    def __init__(self, SVM_model, inner_product_type='rbf_inner', aprox_order=5):
        self.svm_model = SVM_model
        self.gamma = self.svm_model.gamma
        self.lambdas = self.svm_model.lambdas
        self.y = self.svm_model.y
        self.X = self.svm_model.X
        self.b = self.svm_model.b
        self.aprox_order = aprox_order
        self.mu = np.zeros(self.aprox_order)
        self.inner_product_type = {'rbf_inner': lambda x, y: -self.gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)}[inner_product_type]

    def get_aproximation(self, inner_product):
        aprox = 0
        #exponential aproximation using Maclaurin Series
        for idx in range(self.aprox_order):
            self.mu[idx] = 1 / factorial(idx, exact=True)
            aprox = aprox + (self.mu[idx] * ((inner_product) ** idx))
        return aprox

    def decision_function(self, X):
        # get inner product between X and support vectors
        self.inner_product = self.inner_product_type(X, self.X)
        # define exponential aprox
        # self.exp_aprox = np.zeros_like(self.inner_product, dtype=float)
        # simplify inner product to 1-D vector
        self.inner_product_stripe = np.reshape(self.inner_product, np.size(self.inner_product))
        temporal_stripe = np.zeros_like(self.inner_product_stripe)
        # Make an M-order exponential aproximation
        temporal_stripe[:] = self.get_aproximation(self.inner_product_stripe[:])
        self.kernel_aprox = np.reshape(temporal_stripe, np.shape(self.inner_product))
        # print('hice la aproximacion de la exponencial:', self.kernel_aprox, 'dimensiones de la aproximación', np.shape(self.kernel_aprox))
        print('dimensiones de la aproximación exponencial', np.shape(self.kernel_aprox), "\n")
        print("lambdas obtenidos: ", self.lambdas, "lambdas shape: ", self.lambdas.shape, "etiquetas de clase: ", self.y, "labels shape: ", self.y.shape)

        return np.sum(self.kernel_aprox * self.y * self.lambdas, axis=1) + self.b

