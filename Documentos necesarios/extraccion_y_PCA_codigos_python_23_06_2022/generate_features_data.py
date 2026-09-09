import numpy as np
import os

def create_h_file(PCA_matrix,features_list,used_features,vs_movements, test_data):
    PCA_matrix_flat = PCA_matrix.reshape((1,PCA_matrix.shape[0]*PCA_matrix.shape[1]))
    PCA_matrix_flat = PCA_matrix_flat[0]
    file_name = "features_for_movements_{}_vs_{}.h".format(vs_movements[0],vs_movements[1])
    file = open(file_name,'w')
    file.truncate(0)
    file.close()
    file = open(file_name, 'a+')
    # file.write("#ifndef FEATURES_FOR_MOVEMENTS_{}_{}\n".format(vs_movements[0],vs_movements[1]))
    # file.write("#def FEATURES_FOR_MOVEMENTS_{}_{}\n".format(vs_movements[0], vs_movements[1]))
    file.write("#ifndef _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0],vs_movements[1]))
    file.write("#define _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0],vs_movements[1]))
    file.write("#include <stdint.h>\n")
    file.write("//definitions\n")
    for i in range(0,len(features_list)):
        file.write("#define "+ features_list[i] )
        file.write("_mov_{}_vs_{}".format(vs_movements[0], vs_movements[1]))
        file.write(" "+ str(i) + "\n")

    file.write("#define PCA_rows {}\n".format(PCA_matrix.shape[0]))
    file.write("#define PCA_columns {}\n".format(PCA_matrix.shape[1]))
    file.write("const float PCA_matrix_mov_{}_vs_{}[PCA_rows][PCA_columns] = ".format(vs_movements[0],vs_movements[1]))
    file.write("{")
    for i in range(0,len(PCA_matrix_flat) - 1):
        file.write(str(PCA_matrix_flat[i]) + ", ")
    file.write(str(PCA_matrix_flat[len(PCA_matrix_flat) - 1]) + "};\n")

    test_data_flat = test_data.reshape((1,test_data.shape[0]*test_data.shape[1]))
    test_data_flat = test_data_flat[0]

    file.write("#define test_rows {}\n".format(test_data.shape[0]))
    file.write("#define test_columns {}\n".format(test_data.shape[1]))
    file.write("const float test_data[test_rows][test_columns] = {")
    for i in range(0,len(test_data_flat) - 1):
        file.write(str(test_data_flat[i]) + ", ")
    file.write(str(test_data_flat[len(test_data_flat) - 1]) + "};\n")
    file.write("typedef struct \n { \n ")
    file.write("int used_features[{}] ; \n".format(len(used_features)))
    file.write("}")
    file.write("params_mov_{}_vs_{}_t;\n".format(vs_movements[0], vs_movements[1]))
    file.write("params_mov_{}_vs_{}_t ".format(vs_movements[0], vs_movements[1]))
    file.write("params_mov_{}_vs_{}; \n".format(vs_movements[0], vs_movements[1]))
    file.write("void init_params_mov_{}_vs_{}();\n".format(vs_movements[0], vs_movements[1]))
    file.write("#endif")
    file.close()


def create_c_file( used_features, vs_movements):


    file_name = "features_for_movements_{}_vs_{}.c".format(vs_movements[0],vs_movements[1])
    file = open(file_name,'w')
    file.truncate(0)
    file.close()
    file = open(file_name, 'a+')
    file.write("#ifdef _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0], vs_movements[1]))
    file.write("#include ")
    file.write("\"features_for_movements_{}_vs_{}.h\" \n".format(vs_movements[0], vs_movements[1]))
    file.write("void init_params_mov_{}_vs_{}()\n".format(vs_movements[0], vs_movements[1]))
    file.write("{\n")
    for i in range(0, len(used_features)):
        file.write("params_mov_{}_vs_{}.used_features[{}] = ".format(vs_movements[0], vs_movements[1],i))
        file.write("{}_mov_{}_vs_{}".format(used_features[i], vs_movements[0], vs_movements[1]))
        file.write(";\n")
    file.write("}\n")
    file.write("#endif")
    file.close()


features_list = ['MAV_emg_0','MAV_emg_1','MAV_emg_2','MAV_emg_3','MAV_emg_4','MAV_emg_5','MAV_emg_6','MAV_emg_7','MAV_emg_8','MAV_emg_9','MAV_emg_10','MAV_emg_11',
                 'WL_emg_0','WL_emg_1','WL_emg_2','WL_emg_3','WL_emg_4','WL_emg_5','WL_emg_6','WL_emg_7','WL_emg_8','WL_emg_9','WL_emg_10','WL_emg_11',
                 'ZC_emg_0','ZC_emg_1','ZC_emg_2','ZC_emg_3','ZC_emg_4','ZC_emg_5','ZC_emg_6','ZC_emg_7','ZC_emg_8','ZC_emg_9','ZC_emg_10','ZC_emg_11',
                 'RMS_emg_0','RMS_emg_1','RMS_emg_2','RMS_emg_3','RMS_emg_4','RMS_emg_5','RMS_emg_6','RMS_emg_7','RMS_emg_8','RMS_emg_9','RMS_emg_10','RMS_emg_11',
                 'SSC_emg_0','SSC_emg_1','SSC_emg_2','SSC_emg_3','SSC_emg_4','SSC_emg_5','SSC_emg_6','SSC_emg_7','SSC_emg_8','SSC_emg_9','SSC_emg_10','SSC_emg_11',
                 'MAVS_emg_0', 'MAVS_emg_1','MAVS_emg_2','MAVS_emg_3','MAVS_emg_4','MAVS_emg_5','MAVS_emg_6','MAVS_emg_7','MAVS_emg_8','MAVS_emg_9','MAVS_emg_10','MAVS_emg_11']
used_features = ['MAV_emg_0','MAV_emg_1','MAV_emg_2']
vs_movements = [0, 1]
PCA_matrix = np.array([[1.2 , 3.5, 5.0], [2.0 , 8.1, 3.4]])
test_data = np.array([[2.2 , 4.5, 2.3], [3.0, 9.1, 3.4], [3.3, 1.1, 3.4]])

create_h_file(PCA_matrix, features_list,used_features,vs_movements, test_data)
create_c_file(used_features, vs_movements)