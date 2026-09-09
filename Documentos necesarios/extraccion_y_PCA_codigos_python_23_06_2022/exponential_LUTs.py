import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class exponential_LUTs:
    def __init__(self, alpha=1,beta=4, gamma=1, feature_dimensions=2):
        self.mod_set = np.array([2 ** beta - 1, 2 ** alpha, 2 ** beta + 1])
        self.M = np.prod(self.mod_set)
        self.data_max = (self.M / 2) - 1
        self.gamma = gamma
        self.feature_dimensions = feature_dimensions
        self.gamma = gamma
    def create_exponential_table_mod_M(self, gain=1):
        # Amplitude = np.floor((self.M - 1) / (2 * self.feature_dimensions*gain))
        Amplitude = np.floor(self.data_max/(self.feature_dimensions*gain))
        # max_d_squared = np.floor((self.M-1)/(2))
        max_d_squared = np.floor(self.data_max)
        print("max d_squared allowed: ", max_d_squared, "\n")
        d_squared = np.arange(0., max_d_squared, 1)
        gamma_conversion_factor = (self.feature_dimensions / self.data_max)
        gamma_mod_M = self.gamma*gamma_conversion_factor
        exponential = Amplitude*np.exp(-gamma_mod_M*d_squared)
        exponential_int = np.round(exponential)
        average_error = np.average(exponential-exponential_int)
        print("integer exponential:", exponential_int, "\n")
        print("exponential average error:", average_error)

        # Create lookup table in modulo M (dictionary type)

        self.exponential_lut_mod_m = dict(zip(d_squared, exponential_int))
        self.exponential_lut_rns = self.get_exponential_rns_table(d_squared, exponential_int)

        return (self.exponential_lut_mod_m, self.exponential_lut_rns)
    def get_exp_data_from_rns_lut(self,input_data):
        condition = input_data
        condition_size = np.size(condition)
        table_shape = self.exponential_lut_rns.shape
        # print("condition_size: ", condition_size,"\n")
        # print("rns table shape: ", self.exponential_lut_rns.shape , "\n")
        # print("distance values: ", self.exponential_lut_rns[:, 0:condition_size ],"\n")
        idx = np.where((self.exponential_lut_rns[:, 0] == condition[0])
                       & (self.exponential_lut_rns[:, 1] == condition[1])
                       & (self.exponential_lut_rns[:, 2] == condition[2]))
        # print("lut index: ", idx, "\n")
        # print("exponential lut: ", self.exponential_lut_rns[idx], "\n")
        row_match = self.exponential_lut_rns[idx]
        exponential_match = row_match[0, condition_size: ]
        # return self.exponential_lut_rns[idx, condition_size: table_shape[1]]
        return exponential_match



    def get_rns_element(self, element, mod_set):
        return np.remainder(element,mod_set)

    def get_exponential_rns_table(self,d_squared, exponential_int):
        vectorize_remainder = np.vectorize(self.get_rns_element)
        print("d_squared: ", d_squared, "\n")

        ones_vector = np.ones_like(d_squared)
        mod_set_matrix= ones_vector[:,np.newaxis]*self.mod_set
        # print("mod_set_matrix: ", mod_set_matrix, "\n")

        d_squared_rns = vectorize_remainder(d_squared[:,np.newaxis],mod_set_matrix)
        print("mod_set:", self.mod_set,"\n","d_squared_rns: ", d_squared_rns, "\n")
        exponential_int_rns = vectorize_remainder(exponential_int[:, np.newaxis],mod_set_matrix)

        # Create lookup table in RNS (numpy matrix)
        exponential_lut_rns = np.concatenate((d_squared_rns,exponential_int_rns), axis=1)
        print("exponential_lut_rns: ",  exponential_lut_rns, "\n")

        return exponential_lut_rns
