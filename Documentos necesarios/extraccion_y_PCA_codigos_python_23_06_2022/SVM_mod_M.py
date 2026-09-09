import numpy as np
from exponential_LUTs import exponential_LUTs

class SVM_mod_M:
    def __init__(self,svm_ap= None, alpha=1,beta=4, gamma=1, feature_dimensions=2, lambdas= None, labels= None, plane_samples= None, bias=None, gain_plane= None, gain_lambdas= None):
        self.mod_set = np.array([2 ** beta - 1, 2 ** alpha, 2 ** beta + 1])
        self.M = np.prod(self.mod_set)
        self.data_max = (self.M/2)-1
        self.gamma = gamma
        self.feature_dimensions = feature_dimensions
        self.gamma = gamma
        self.raw_lambdas = lambdas
        self.raw_labels = labels
        self.raw_plane_samples = plane_samples
        self.raw_bias = bias
        self.lambdas_gain = gain_lambdas
        self.plane_gain = gain_plane
        self.exponential_gain = None
        self.bias_gain = None
        self.normal_lambdas = None
        self.normal_plane = None
        self.normal_bias = None
        self.normal_labels = None
        self.max_lambdas= None
        self.max_hiperplane = None
        self.normalize_hiperplane_mod_M()
        self.normalize_lambdas_mod_M()
        self.normalize_labels_mod_M()
        self.get_exponential_gain()
        self.get_bias_gain()
        self.normalize_bias_mod_M()
        self.exp_lut = exponential_LUTs(alpha=alpha,beta=beta,gamma=gamma)
        ########## pruebas sin usar ganancias, solo la de la exponencial(Amplitud) ####
        # self.no_gains_used()
        ############
        print("exponential gain:", self.exponential_gain, "\n")
        (self.exp_mod_M, self.exp_rns) = self.exp_lut.create_exponential_table_mod_M(self.exponential_gain)
        # rns support vectors
        self.svm_ap = svm_ap
        self.X = self.svm_ap.X
    def no_gains_used(self):
        self.normal_lambdas = self.raw_lambdas
        # Amplitude = np.floor((self.M - 1) / (2 * self.feature_dimensions))
        Amplitude = np.floor(self.data_max/self.feature_dimensions)
        self.normal_bias = self.raw_bias*Amplitude
        self.exponential_gain = 1
    def normalize_lambdas_mod_M(self):
        self.max_lambdas = np.amax(self.raw_lambdas)
        self.normal_lambdas = self.raw_lambdas*(self.lambdas_gain / self.max_lambdas)
        #new -> integer lambdas
        self.normal_lambdas = np.floor(self.normal_lambdas)
        ##
        print("max lambdas:", self.max_lambdas)
    def normalize_hiperplane_mod_M(self):
        self.max_hiperplane = np.amax(np.absolute(self.raw_plane_samples))
        # holgura para el maximo en el hiperplano
        self.max_hiperplane = self.max_hiperplane
        self.normal_plane = self.raw_plane_samples*(self.plane_gain / self.max_hiperplane)
        print("mod M plane", self.normal_plane,"max and min : ", np.amax(self.normal_plane), np.amin(self.normal_plane))

    def normalize_bias_mod_M(self):
        self.normal_bias = self.raw_bias*self.bias_gain
        # new -> integer bias
        self.normal_bias = np.floor(self.normal_bias)
    def get_exponential_gain(self):
        # self.exponential_gain = (self.lambdas_gain*(self.M - 1)/(2*self.max_lambdas*self.feature_dimensions))\
        #                         *(self.max_hiperplane/self.plane_gain)
        self.exponential_gain = (self.lambdas_gain*self.data_max/(self.max_lambdas*self.feature_dimensions))\
                                *(self.max_hiperplane/self.plane_gain)
    def get_bias_gain(self):
        self.bias_gain = (self.plane_gain/self.max_hiperplane)


    def normalize_labels_mod_M(self):
        self.normal_labels = self.raw_labels

    def get_exp_rns(self, distance_rns):
        return self.exp_lut.get_exp_data_from_rns_lut(distance_rns)

    def get_exp_mod_M(self, distance_mod_M):
        return self.exp_mod_M[distance_mod_M]

    def get_inner_product_mod_M(self, x_data, y_data):
        # x_conversion_factor = np.sqrt((self.M - 1) / (2 * self.feature_dimensions))
        # y_conversion_factor = np.sqrt((self.M - 1) / (2 * self.feature_dimensions))

        x_conversion_factor = np.sqrt(self.data_max/ self.feature_dimensions)
        y_conversion_factor = np.sqrt(self.data_max/ self.feature_dimensions)

        x_mod_M = np.floor(x_data * x_conversion_factor)
        y_mod_M = np.floor(y_data * y_conversion_factor)

        # print("resta de componentes en el producto interno:",(y_mod_M - x_mod_M[:, np.newaxis]) ** 2)
        print("maximos del producto interno:", np.amax(x_mod_M),np.amax(y_mod_M),np.amax((y_mod_M - x_mod_M[:, np.newaxis]) ** 2),"\n")
        print("minimos del producto interno:", np.amin(x_mod_M), np.amin(y_mod_M),
              np.amin((y_mod_M - x_mod_M[:, np.newaxis]) ** 2), "\n")

        d_M = np.sum((y_mod_M - x_mod_M[:, np.newaxis]) ** 2, axis=-1)
        return d_M

    def get_kernel_matrix(self, inner_product):
        inner_flat_shape = (1,inner_product.shape[0]*inner_product.shape[1])
        inner_flat = np.reshape(inner_product, inner_flat_shape)
        inner_flat = inner_flat[0,:]
        print("producto interno aplanado: ", inner_flat,"\n")
        # kernel_flat = np.apply_along_axis(self.get_exp_mod_M, axis=0, arr=inner_flat)
        kernel_flat = np.vstack(self.get_exp_mod_M(inner_flat[idx]) for idx in range(inner_flat.shape[0]))
        kernel = np.reshape(kernel_flat, inner_product.shape)
        return kernel

    def decision_function(self, X):
        print("feature dimensions",self.feature_dimensions,"\n","x_conversion_factor:", np.sqrt(self.data_max/ self.feature_dimensions), "\n")
        print("empezaré el producto interno modulo M \n")
        self.inner_product = self.get_inner_product_mod_M(X,self.X)
        print("terminé el producto interno modulo M:",self.inner_product,"\n")

        #Todo: crear la matriz kernel con la lookuptable exponencial (incluir gamma)
        self.kernel_matrix = self.get_kernel_matrix(self.inner_product)
        print("terminé el cálculo de la matriz kernel modulo M:", self.kernel_matrix, "\n")

        #Todo: crear el hiperplano considerando la matriz kernel
        print("producto kernel*lambdas mod M", self.kernel_matrix * self.normal_lambdas,
              "\n")
        print("producto kernel*lambdas*labels mod M", self.kernel_matrix * self.normal_lambdas * self.normal_labels,"\n")
        return np.sum(self.kernel_matrix * self.normal_lambdas * self.normal_labels, axis=1) + self.normal_bias

        #Todo: realizar la comparación (Clase 1 o Clase 2)

    def verify_gains(self):
        print("g_lim ", self.max_hiperplane)
        print("G_plane", self.plane_gain)
        print("g_bias", self.bias_gain)
        print("g_lambdas", self.lambdas_gain)
        print("g_exp ", self.exponential_gain)
        print("G_plane/g_lim", self.plane_gain/self.max_hiperplane)
        num= self.lambdas_gain*self.data_max
        den = self.max_lambdas*self.feature_dimensions*self.exponential_gain
        print("(G_lambda.data_max)/(max_lambda*k*Gexp)", num/den)
        print("raw lambdas:", self.raw_lambdas)
        print("normal lambdas:", self.normal_lambdas)
        print("raw bias:", self.raw_bias)
        print("normal bias: ", self.normal_bias)
        print("raw labels:", self.raw_labels)
        print("normal labels:", self.normal_labels)

