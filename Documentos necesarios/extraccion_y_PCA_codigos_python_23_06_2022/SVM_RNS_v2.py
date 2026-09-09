import numpy as np
from math import prod
from exponential_LUTs import exponential_LUTs

class SVM_RNS_v2:
    def __init__(self,exp_rns_lut, svm_model_in, lambdas_mod_M, bias_mod_M,labels_mod_M, alpha, beta, feature_dimensions):
        self.feature_dimensions = feature_dimensions
        self.exp_rns_lut = exp_rns_lut
        self.lambdas_mod_M =  lambdas_mod_M
        self.bias_mod_M = bias_mod_M
        self.labels_mod_M = labels_mod_M
        self.alpha = alpha
        self.beta = beta
        self.mod_set = np.array([2 ** beta - 1, 2 ** alpha, 2 ** beta + 1])
        self.M = np.prod(self.mod_set)
        self.lambdas_rns = None
        self.bias_rns = None
        self.labels_rns = None
        self.svm_ap = svm_model_in
        self.X = self.svm_ap.X
        self.X_rns = None
        self.data_max = (self.M/2)-1
        self.data_conversion_factor = np.sqrt(self.data_max/ self.feature_dimensions)
        self.convert_svm_params_to_rns()
        self.M1 = self.mod_set[1]*self.mod_set[2]
        self.M2 = self.mod_set[0]*self.mod_set[2]
        self.M3 = self.mod_set[0]*self.mod_set[1]
        self.LPN_table = np.zeros((self.M, len(self.mod_set) + 2))
        self.LPN_size = np.shape(self.LPN_table)
        self.lpn_table_create()

    def convert_scalar_2_rns(self, scalar):
        scalar_rns = np.remainder(scalar, self.mod_set)
        return scalar_rns

    def convert_vectors_2_rns(self, vector):
        return np.remainder(vector[:, np.newaxis], self.mod_set)

    def convert_matrix_2_rns(self, matrix):
        matrix_rns_flat = np.vstack(self.convert_vectors_2_rns(matrix[idx, :]) for idx in range(matrix.shape[0]))
        matrix_rns = matrix_rns_flat.reshape(matrix.shape[0], matrix.shape[1], 3)
        return matrix_rns
    def convert_data_matrix_2_rns(self, data_matrix):
        data_matrix_mod_M =  np.floor(self.data_conversion_factor*data_matrix)
        data_matrix_rns = self.convert_matrix_2_rns(data_matrix_mod_M)
        return data_matrix_rns

    def convert_svm_params_to_rns(self):
        # Vector params to RNS
        self.lambdas_rns = np.remainder(self.lambdas_mod_M[:, np.newaxis], self.mod_set)
        self.labels_rns = np.remainder(self.labels_mod_M[:,np.newaxis], self.mod_set)
        # Scalar params to rns
        self.bias_rns = np.remainder(self.bias_mod_M, self.mod_set)
        # Convert Support Vectors to RNS
        self.X_mod_M = np.floor(self.data_conversion_factor*self.X)
        self.X_rns = self.convert_matrix_2_rns(self.X_mod_M)


    def get_inner_product_rns(self, x_data):
        x_data_rns = self.convert_data_matrix_2_rns(x_data)
        x_new = x_data_rns[:,np.newaxis]

        # X_support - X_data
        sub_X_data = self.X_rns - x_new

        # flat substraction
        sub_X_data_flat_shape = (int((prod(sub_X_data.shape)) / sub_X_data.shape[-1]), sub_X_data.shape[-1])
        sub_X_data_flat = np.reshape(sub_X_data, sub_X_data_flat_shape)

        # take remainder of substraction in moduli set: (X_support - X_data)_mi
        sub_X_data_flat_res = np.remainder(sub_X_data_flat, self.mod_set)

        # square substraction in moduli set: ((X_support - X_data)**2)_mi
        sub_X_data_square = np.remainder(sub_X_data_flat_res * sub_X_data_flat_res, self.mod_set)

        # reorganize squared array into original shape
        sub_X_data_square_res = np.reshape(sub_X_data_square, sub_X_data.shape)
        distance = np.sum(sub_X_data_square_res, axis=2)
        distance_rns = np.remainder(distance, self.mod_set)

        return distance_rns

    def get_exp_data_from_rns_lut(self, input_data):
        condition = input_data
        condition_size = np.size(condition)

        idx = np.where((self.exp_rns_lut[:, 0] == condition[0])
                       & (self.exp_rns_lut[:, 1] == condition[1])
                       & (self.exp_rns_lut[:, 2] == condition[2]))
        row_match = self.exp_rns_lut[idx]
        # print("row match:", row_match)
        exponential_match = row_match[0, condition_size: ]
        return exponential_match

    def get_kernel_rns(self, inner_product_rns):

        # flat inner product
        inner_product_flat_shape = (
        int((prod(inner_product_rns.shape)) / inner_product_rns.shape[-1]), inner_product_rns.shape[-1])
        inner_product_flat = np.reshape(inner_product_rns, inner_product_flat_shape)

        # flat exponential rns
        kernel_flat = np.vstack(self.get_exp_data_from_rns_lut(inner_product_flat[idx,:]) for idx in range(inner_product_flat_shape[0]))
        kernel = np.reshape(kernel_flat,inner_product_rns.shape)

        return kernel

    def get_hyperplane_rns(self, data):
        # get inner product
        self.inner_product_rns = self.get_inner_product_rns(data)

        # get kernel matrix
        print("empezando a calcular kernel rns")
        self.kernel = self.get_kernel_rns(self.inner_product_rns)
        # print("kernel rns calculado:")
        print("kernel rns calculado:", self.kernel)
        # Hasta aquí el kernel rns corresponde con el kernel mod M
        # ToDo: Revisar si corresponde con el hiperplano requerido y sumar el bias
        print("empezando a calcular hiperplano rns")
        # hyperplane_matrix = ((self.kernel * self.lambdas_rns) * self.labels_rns) + self.bias_rns
        hyperplane_matrix = ((self.kernel * self.lambdas_rns) * self.labels_rns)
        print("primera posición del hiperplano rns antes de la sumatoria : ",hyperplane_matrix [0,:])
        sum_first_hyperplane_element = np.sum(hyperplane_matrix [0,:], axis=0)
        print("primera posición del hiperplano rns sumada : ", sum_first_hyperplane_element)
        # el producto kernel*lambdas coincide, pero debe hacerse una reduccion en el conjunto de residuos
        print("producto kernel*lambdas rns:", self.kernel * self.lambdas_rns, "\n")
        # el producto kernel*lambdas*labels coincide, pero debe hacerse una reduccion en el conjunto de residuos
        print("producto kernel*lambdas*labels rns:",(self.kernel * self.lambdas_rns) * self.labels_rns,"\n")

        sum_hyperplane = np.sum(hyperplane_matrix, axis=1) + self.bias_rns
        # el hiperplano rns SI coincide con el hiperplano mod M
        sum_hyperplane_rns = np.remainder(sum_hyperplane, self.mod_set)
        # print("hiperplano rns calculado")
        print("hiperplano rns calculado: ",sum_hyperplane_rns, "\n")
        print("calculando signos con el metodo del lpn")
        # rns_offset_compare = np.remainder(np.floor((self.M / 2) - 1), self.mod_set)
        rns_offset_compare = np.remainder(np.floor((self.M / 2)), self.mod_set)

        # sign_hyperplane = np.apply_along_axis(self.rns_compare, axis=1, arr=sum_hyperplane_rns, b=rns_offset_compare)
        sign_hyperplane = np.apply_along_axis(self.rns_sign, axis=1, arr=sum_hyperplane_rns, offset=rns_offset_compare)


        return sign_hyperplane
    def rns_sign(self, a, offset):
        # a<offset -> comp = 1, a>offset -> comp = -1, a = offset -> comp = -1
        b = offset
        comp = 0
        r_a = a[1]
        r_b = b[1]

        # Least Possible Numbers and Reference Residue
        # lpn_rr_a = self.find_lpn(a)
        lpn_rr_a = self.find_lpn_v2(a)
        lpn_a = lpn_rr_a[0]
        rr_a = lpn_rr_a[1]

        # lpn_rr_b = self.find_lpn(b)
        lpn_rr_b = self.find_lpn_v2(b)
        lpn_b = lpn_rr_b[0]
        rr_b = lpn_rr_b[1]

        #Comparison algorithm
        ref_a = rr_a - r_a
        ref_b = rr_b - r_b

        # Compensate negative references. A and B must be positives (range(M))
        if ref_a < 0:
            ref_a = self.mod_set[1] + ref_a
        if ref_b < 0:
            ref_b = self.mod_set[1] + ref_b


        if ref_a > ref_b:
            # a > b
            comp = -1
        elif ref_a < ref_b:
            # a < b
            comp = 1
        elif ref_a == ref_b:
            if lpn_a > lpn_b:
                # a > b
                comp = -1
            elif lpn_a < lpn_b:
                # a < b
                comp = 1
            elif lpn_a == lpn_b:
                # a == b
                comp = -1
        if not np.any(a):
            # a = 0
            comp = 0

        return comp


    def rns_compare(self, a, b):
        # a<b -> comp = 0, a>b -> comp = 1, a = b -> comp = 2
        comp = 0
        r_a = a[1]
        r_b = b[1]

        # Least Possible Numbers and Reference Residue
        # lpn_rr_a = self.find_lpn(a)
        lpn_rr_a = self.find_lpn_v2(a)
        lpn_a = lpn_rr_a[0]
        rr_a = lpn_rr_a[1]

        # lpn_rr_b = self.find_lpn(b)
        lpn_rr_b = self.find_lpn_v2(b)
        lpn_b = lpn_rr_b[0]
        rr_b = lpn_rr_b[1]

        #Comparison algorithm
        ref_a = rr_a - r_a
        ref_b = rr_b - r_b

        # Compensate negative references. A and B must be positives (range(M))
        if ref_a < 0:
            ref_a = self.mod_set[1] + ref_a
        if ref_b < 0:
            ref_b = self.mod_set[1] + ref_b


        if ref_a > ref_b:
            # a > b
            comp = 1
        elif ref_a < ref_b:
            # a < b
            comp = -1
        elif ref_a == ref_b:
            if lpn_a > lpn_b:
                # a > b
                comp = 1
            elif lpn_a < lpn_b:
                # a < b
                comp = -1
            elif lpn_a == lpn_b:
                # a == b
                comp = 0

        return comp

    def find_lpn(self, x_rns):

        for idx in range(self.LPN_size[0]):
            x0 = self.LPN_table[idx, 0]
            x1 = self.LPN_table[idx, 1]
            x2 = self.LPN_table[idx, 2]
            lpn_x = self.LPN_table[idx, 3]
            rr_x = self.LPN_table[idx, 4]
            if x0 == x_rns[0] and x1 == x_rns[1] and x2 == x_rns[2]:
                return np.array([lpn_x, rr_x])
    def find_lpn_v2(self,x_rns):
        condition = x_rns
        condition_size = np.size(condition)

        idx = np.where((self.LPN_table[:, 0] == condition[0])
                       & (self.LPN_table[:, 1] == condition[1])
                       & (self.LPN_table[:, 2] == condition[2]))
        # print("lut index: ", idx, "\n")
        # print("exponential lut: ", self.exponential_lut_rns[idx], "\n")
        row_match = self.LPN_table[idx]
        lpn_x = row_match[0, 3]
        rr_x = row_match[0, 4]
        # return self.exponential_lut_rns[idx, condition_size: table_shape[1]]
        return np.array([lpn_x, rr_x])
    def lpn_table_create(self):
        # create LPN lookup table: cols = x1 x2 x3 LPNx RRx
        for idx in range(self.LPN_size[0]):
            x_rns = np.remainder(idx, self.mod_set)
            lpn_x = np.remainder(idx, self.M2)
            lpn_array = np.array([lpn_x, np.remainder(lpn_x, self.mod_set[1])])
            self.LPN_table[idx] = np.concatenate((x_rns, lpn_array))
