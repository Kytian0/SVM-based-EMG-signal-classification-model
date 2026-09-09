import numpy as np
from math import prod

class export_data_files:

    def __init__(self, svm_rns_object, kernel_data_file_name, params_data_file_name, bias_data_file_name):
        self.svm_rns = svm_rns_object
        self.exp_lut_obj = self.svm_rns.exp_rns_lut
        self.alpha = self.svm_rns.alpha
        self.beta = self.svm_rns.beta
        self.mod_set = self.svm_rns.mod_set
        self.feature_dimensions = self.svm_rns.feature_dimensions
        self.create_data_file_kernel(self.alpha, self.beta, self.mod_set, kernel_data_file_name)
        self.create_data_file_params(params_data_file_name)
        self.create_data_file_bias(bias_data_file_name)


    # ToDo: crear e implementar la función create_data_file_params para guardar el documento de datos de vectores
    #  de soporte, bias, lambdas

    #ToDo 2: probar el correcto funcionamiento de create_data_file_kernel

    def create_data_file_kernel(self,alpha,beta, mod_set_in,data_name):

        #Create LUT for Kernel

        max_data = int(self.svm_rns.data_max)
        number_of_lines = 2**(2*beta + alpha + 1)
        word_width = 2*beta + alpha + 1
        zeros_data = ''.join('0' for i in range(word_width)) + '\n'
        # data_to_write = ['' for j in range(number_of_lines)] # original
        data_to_write = [zeros_data for j in range(number_of_lines)]

        for i in range(max_data):
            x = np.remainder(i, mod_set_in)

            address = (x[2] << (alpha + beta))+(x[1] << beta)+(x[0]) ## original

            data_vector = self.svm_rns.get_exp_data_from_rns_lut(x)
            data_word = int((int(data_vector[2]) << (alpha + beta)) + (int(data_vector[1]) << beta) + (int(data_vector[0]))) ## original


            data_str = str(bin(data_word))
            data_str = data_str[2:len(data_str)]
            data_to_write[address] = ''.join('0' for k in range(word_width- len(data_str) )) + data_str+ "\n"
        for i in range(max_data):
            if data_to_write[i] == '':
                data_to_write[i] = ''.join('0' for i in range(word_width)) + '\n'


        file = open(data_name, 'w') # Open in write mode / create file
        file.truncate(0) # reset data
        file.close()
        file = open(data_name, 'w')
        file.writelines(data_to_write) # Write data
        file.close() # close file

    def create_data_file_params(self, data_file_name):
        lambdas_rns = self.svm_rns.lambdas_rns
        labels_rns = self.svm_rns.labels_rns
        sup_vectors_rns = self.svm_rns.X_rns
        print("PARAMETROS A EXPORTAR\n")
        print("lambdas:", lambdas_rns, "\n")
        print("labels:", labels_rns, "\n")
        print("support vectors:", sup_vectors_rns, "\n")

        # ToDo: obtener las memory lines para todos los lambdas que no sean cero
        idx = 0
        sup_vector_flat_shape = (1,int((prod(sup_vectors_rns[idx,:].shape))))
        sup_vector_flat = np.reshape(sup_vectors_rns[idx,:],sup_vector_flat_shape )
        data_line = np.concatenate((lambdas_rns[idx, :], labels_rns[idx, :], sup_vector_flat), axis=None)

        param_width = 2 * self.beta + self.alpha + 1
        word_width = (self.feature_dimensions + 2)*param_width
        # zeros_data = ''.join('0' for i in range(word_width)) + '\n'
        # number_of_lines = number_of_non_zero_lambdas
        # data_to_write = [zeros_data for j in range(number_of_lines)]
        data_to_write = []
        lambdas_zip = (int(lambdas_rns[idx, 2]) << int(self.alpha + self.beta)) + (int(lambdas_rns[idx, 1]) << int(self.beta)) \
                      + int(lambdas_rns[idx, 0])
        labels_zip = (int(labels_rns[idx, 2]) << int(self.alpha + self.beta)) + (int(labels_rns[idx, 1]) << int(self.beta)) \
                      + int(labels_rns[idx, 0])
        sup_vector_zip = 0
        for feat_idx in range(self.feature_dimensions):
            sup_vector_zip = (((int(sup_vectors_rns[idx,feat_idx, 2])<< int(self.alpha + self.beta)) +
                              (int(sup_vectors_rns[idx,feat_idx, 1])<< int(self.beta)) +
                              int(sup_vectors_rns[idx,feat_idx, 0]))<<int(feat_idx*param_width)) + sup_vector_zip

        memory_line = (lambdas_zip << int((self.feature_dimensions + 1)*param_width)) +\
                      (labels_zip << int(self.feature_dimensions*param_width)) + \
                      sup_vector_zip

        print("data line:", data_line, "index:", idx, "\n")
        print("lambda zip", lambdas_zip, "\n")
        print("labels zip", labels_zip, "\n")
        print("SV zip", sup_vector_zip, "\n")
        print("memory line in hex", hex(memory_line), "index:", idx, "\n")

        data_to_write = []
        k = 0
        for idx in range(lambdas_rns.shape[0]):

            if not(int(lambdas_rns[idx, 2]) == 0 and int(lambdas_rns[idx, 1]) == 0 and int(lambdas_rns[idx, 0]) == 0):

                lambdas_zip = (int(lambdas_rns[idx, 2]) << int(self.alpha + self.beta)) + (
                            int(lambdas_rns[idx, 1]) << int(self.beta)) \
                              + int(lambdas_rns[idx, 0])
                labels_zip = (int(labels_rns[idx, 2]) << int(self.alpha + self.beta)) + (
                            int(labels_rns[idx, 1]) << int(self.beta)) \
                             + int(labels_rns[idx, 0])
                sup_vector_zip = 0
                for feat_idx in range(self.feature_dimensions):
                    sup_vector_zip = (((int(sup_vectors_rns[idx, feat_idx, 2]) << int(self.alpha + self.beta)) +
                                       (int(sup_vectors_rns[idx, feat_idx, 1]) << int(self.beta)) +
                                       int(sup_vectors_rns[idx, feat_idx, 0])) << int(
                        feat_idx * param_width)) + sup_vector_zip

                memory_line = (lambdas_zip << int((self.feature_dimensions + 1) * param_width)) + \
                              (labels_zip << int(self.feature_dimensions * param_width)) + \
                              sup_vector_zip

                data_str = str(bin(memory_line))
                data_str = data_str[2:len(data_str)]
                print("data str:", data_str, "\n", "word width - len(data_str):", word_width - len(data_str), "\n")
                data_to_write.append(''.join('0' for j in range(word_width - len(data_str))) + data_str + "\n")

                k = k + 1
        file = open(data_file_name, 'w') # Open in write mode / create file
        file.truncate(0) # reset data
        file.close()
        file = open(data_file_name, 'w')
        file.writelines(data_to_write) # Write data
        file.close() # close file

    def create_data_file_bias(self, bias_data_file_name):
        bias_rns = self.svm_rns.bias_rns
        word_width = 2 * self.beta + self.alpha + 1

        data_vector = bias_rns
        data_word = int((int(data_vector[2]) << (self.alpha + self.beta)) + (int(data_vector[1]) << self.beta) + (int(data_vector[0]))) ## original


        data_str = str(bin(data_word))
        data_str = data_str[2:len(data_str)]
        data_to_write = ''.join('0' for k in range(word_width- len(data_str) )) + data_str+ "\n"
        file = open(bias_data_file_name, 'w')  # Open in write mode / create file
        file.truncate(0)  # reset data
        file.close()
        file = open(bias_data_file_name, 'w')
        file.writelines(data_to_write)  # Write data
        file.close()  # close file


