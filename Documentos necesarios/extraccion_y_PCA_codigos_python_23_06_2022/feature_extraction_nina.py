# libraries
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

import time

import random

class feature_extraction_system:
    def __init__(self, movements_to_classify=(22, 27, 28, 29)):
        self.movements_to_classify = movements_to_classify
        self.raw_data = None
        self.raw_keys = {}
        self.keys = {}
        self.Subject = {}
        # Electrode channels
        # channel 0 - channel 7 -> forearm electrodes
        # channel 8 -> Flexor digitorum
        # channel 9 -> Extensor digitorum
        # channel 10 -> Biceps Brachii
        # channel 11 -> Triceps Brachii

        self.channels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # f_s = 2 kHz -> T_s = 0.0005 s ; window time = 200 ms
        self.f_s = 2000
        self.T_s = 1 / self.f_s
        self.window_time = 200e-3
        self.samples_per_window = self.window_time * self.f_s
        self.movements_data_segmented = {}

        self.features_list = ['MAV','WL', 'ZC', 'RMS', 'SSC', 'MAVS']
        self.features_per_channel = len(self.features_list)
        # movement pair represents indexes in "movements_to_classify" array
        self.movement_pair = [0, 1]
        self.reduced_movement_1_rfe = None
        self.reduced_movement_2_rfe = None
        self.C = 1e7
        self.gamma = 'auto'
        self.all_features_names = ['MAV_emg_0', 'MAV_emg_1', 'MAV_emg_2', 'MAV_emg_3', 'MAV_emg_4', 'MAV_emg_5', 'MAV_emg_6',
                         'MAV_emg_7', 'MAV_emg_8', 'MAV_emg_9', 'MAV_emg_10', 'MAV_emg_11',
                         'WL_emg_0', 'WL_emg_1', 'WL_emg_2', 'WL_emg_3', 'WL_emg_4', 'WL_emg_5', 'WL_emg_6', 'WL_emg_7',
                         'WL_emg_8', 'WL_emg_9', 'WL_emg_10', 'WL_emg_11',
                         'ZC_emg_0', 'ZC_emg_1', 'ZC_emg_2', 'ZC_emg_3', 'ZC_emg_4', 'ZC_emg_5', 'ZC_emg_6', 'ZC_emg_7',
                         'ZC_emg_8', 'ZC_emg_9', 'ZC_emg_10', 'ZC_emg_11',
                         'RMS_emg_0', 'RMS_emg_1', 'RMS_emg_2', 'RMS_emg_3', 'RMS_emg_4', 'RMS_emg_5', 'RMS_emg_6',
                         'RMS_emg_7', 'RMS_emg_8', 'RMS_emg_9', 'RMS_emg_10', 'RMS_emg_11',
                         'SSC_emg_0', 'SSC_emg_1', 'SSC_emg_2', 'SSC_emg_3', 'SSC_emg_4', 'SSC_emg_5', 'SSC_emg_6',
                         'SSC_emg_7', 'SSC_emg_8', 'SSC_emg_9', 'SSC_emg_10', 'SSC_emg_11',
                         'MAVS_emg_0', 'MAVS_emg_1', 'MAVS_emg_2', 'MAVS_emg_3', 'MAVS_emg_4', 'MAVS_emg_5',
                         'MAVS_emg_6', 'MAVS_emg_7', 'MAVS_emg_8', 'MAVS_emg_9', 'MAVS_emg_10', 'MAVS_emg_11']

    def create_h_file(self):
        features_list = self.all_features_names
        used_features = self.selected_features_rfe
        vs_movements = self.movement_pair
        PCA_matrix_data = self.PCA_matrix
        PCA_matrix_flat = PCA_matrix_data.reshape((1, PCA_matrix_data.shape[0] * PCA_matrix_data.shape[1]))
        PCA_matrix_flat = PCA_matrix_flat[0]
        PCA_mean = self.PCA_mean.reshape((1, self.PCA_mean.shape[0]))
        PCA_mean = PCA_mean[0]
        file_name = "features_for_movements_{}_vs_{}.h".format(vs_movements[0], vs_movements[1])
        file = open(file_name, 'w')
        file.truncate(0)
        file.close()
        file = open(file_name, 'a+')
        # file.write("#ifndef FEATURES_FOR_MOVEMENTS_{}_{}\n".format(vs_movements[0],vs_movements[1]))
        # file.write("#def FEATURES_FOR_MOVEMENTS_{}_{}\n".format(vs_movements[0], vs_movements[1]))
        file.write("#ifndef _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0], vs_movements[1]))
        file.write("#define _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0], vs_movements[1]))
        file.write("#include <stdint.h>\n")
        file.write("//definitions\n")
        for i in range(0, len(features_list)):
            file.write("#define " + features_list[i])
            file.write("_mov_{}_vs_{}".format(vs_movements[0], vs_movements[1]))
            file.write(" " + str(i) + "\n")

        file.write("#define PCA_matrix_rows {}\n".format(PCA_matrix_data.shape[0]))
        file.write("#define PCA_matrix_columns {}\n".format(PCA_matrix_data.shape[1]))

        file.write(
            "const float PCA_matrix_mov_{}_vs_{}[PCA_matrix_rows][PCA_matrix_columns] = ".format(vs_movements[0], vs_movements[1]))
        file.write("{{")
        for i in range(0, len(PCA_matrix_flat) - 1):

            if (i % PCA_matrix_data.shape[1] == 0) and (i != len(PCA_matrix_flat) - 2) and (i !=0):

                file.write("},{")
                file.write(str(PCA_matrix_flat[i]) + ", ")
            else:
                if (i+1) % PCA_matrix_data.shape[1] == 0:
                    file.write(str(PCA_matrix_flat[i]))
                else:
                    file.write(str(PCA_matrix_flat[i])+ ", ")



        file.write(str(PCA_matrix_flat[len(PCA_matrix_flat) - 1]) + "}};\n")

        file.write("#define PCA_mean_rows {}\n".format(PCA_mean.shape[0]))

        file.write(
            "const float PCA_mean_mov_{}_vs_{}[PCA_mean_rows] = ".format(vs_movements[0], vs_movements[1]))
        file.write("{")
        for i in range(0, len(PCA_mean) - 1):
            file.write(str(PCA_mean[i]) + ", ")
        file.write(str(PCA_mean[len(PCA_mean) - 1]) + "};\n")
        # ToDo: no pasar los datos de test transformados por PCA, pasar las senales EMG

        print("movement 1[0][0] to *.h shape", np.shape(self.movement_1_test[0,0]))
        segment_length = np.shape(self.movement_1_test[0,0])
        segment_length = segment_length[0]
        emg_channels_data_mov_1 = np.zeros((self.number_of_emg_channels, segment_length))
        file.write("#define number_of_used_segments {}\n".format(self.used_segments))
        file.write("#define number_of_channels_used {}\n".format(self.number_of_emg_channels))
        file.write('#define segment_length {}\n'.format(segment_length))
        for i in range(self.used_segments):
            file.write("const float test_data_mov1_segment{}[number_of_channels_used][segment_length] =".format(i))
            file.write(" {{")
            for j in range(self.number_of_emg_channels):
                test_data_1 = self.movement_1_test[i,j]
                test_data_1_flat = test_data_1.reshape((1, segment_length))
                test_data_1_flat = test_data_1_flat[0]
                emg_channels_data_mov_1[j,:] = test_data_1_flat

                for k in range(segment_length):
                    file.write(str(emg_channels_data_mov_1[j, k]))
                    if (k == segment_length - 1) and (j == self.number_of_emg_channels-1):
                        file.write("}};\n\n ")
                    elif((k + 1) % segment_length == 0 ) and (k != 0):
                        file.write("},{")
                    if((k + 1) % segment_length != 0 ) or (k == 0):
                        file.write(", ")



        print("movement 2[0][0] to *.h shape", np.shape(self.movement_2_test[0,0]))
        segment_length = np.shape(self.movement_2_test[0,0])
        segment_length = segment_length[0]
        emg_channels_data_mov_2 = np.zeros((self.number_of_emg_channels, segment_length))
        # file.write("#define number_of_used_segments {}\n".format(self.used_segments))
        # file.write("#define number_of_channels_used {}\n".format(self.number_of_emg_channels))
        # file.write('#define segment_length {}\n'.format(segment_length))
        for i in range(self.used_segments):
            file.write("const float test_data_mov2_segment{}[number_of_channels_used][segment_length] =".format(i))
            file.write(" {{")
            for j in range(self.number_of_emg_channels):
                test_data_2 = self.movement_2_test[i,j]
                test_data_2_flat = test_data_2.reshape((1, segment_length))
                test_data_2_flat = test_data_2_flat[0]
                emg_channels_data_mov_2[j,:] = test_data_2_flat

                for k in range(segment_length):
                    file.write(str(emg_channels_data_mov_2[j, k]))
                    if (k == segment_length - 1) and (j == self.number_of_emg_channels-1):
                        file.write("}};\n\n ")
                    elif ((k + 1) % segment_length == 0) and (k != 0):
                        file.write("},{")
                    if ((k + 1) % segment_length != 0) or (k == 0):
                        file.write(", ")
        # testing features
        total_features_emg = self.features_per_channel * self.channels.shape[0]
        file.write("#define total_features_emg {}\n".format(total_features_emg))
        file.write("const float test_features_mov1[number_of_used_segments ][total_features_emg] =")
        file.write("{{")
        for i in range(0, self.used_segments):

            for j in range(0, total_features_emg):
                feat_data = self.movement_1_features_test[i].to_numpy()
                file.write(str(feat_data[j]))
                if(i == self.used_segments - 1) and (j == total_features_emg-1):
                    file.write("}};\n\n ")
                elif ((j + 1) % total_features_emg == 0) and (j != 0):
                    file.write("},{")
                if ((j + 1) % total_features_emg != 0) or (j == 0):
                    file.write(", ")

        file.write("#define reduced_features_len {}\n".format(len(self.selected_features_rfe)))
        file.write("const float test_reduced_feat_mov1[number_of_used_segments ][reduced_features_len] =")
        file.write("{{")
        for i in range(0, self.used_segments):

            for j in range(0, len(self.selected_features_rfe)):
                feat_data = self.movement_1_reduced[i][j]
                file.write(str(feat_data))
                if (i == self.used_segments - 1) and (j == len(self.selected_features_rfe) - 1):
                    file.write("}};\n\n ")
                elif((j + 1) % len(self.selected_features_rfe) == 0) and (j != 0):
                    file.write( "},{")
                if((j + 1) % len(self.selected_features_rfe) != 0) or (j == 0):
                    file.write( ", ")

        # self.movement_1_reduced = np.zeros((used_segments,len(self.selected_features_rfe)))

        print("PCA movement 1 for testing:", self.movement_1_test_PCA)
        print("PCA movement 2 for testing:", self.movement_2_test_PCA)
        PCA_movement_test_1_shape = np.shape(self.movement_1_test_PCA)
        PCA_movement_test_2_shape = np.shape(self.movement_2_test_PCA)
        file.write("#define PCA_test_rows {}\n".format(PCA_movement_test_1_shape[0]))
        file.write("#define PCA_test_cols {}\n".format(PCA_movement_test_1_shape[1]))
        file.write("const float PCA_movement_test_1 [PCA_test_rows][PCA_test_cols] =")
        file.write(" {{")
        for i in range(PCA_movement_test_1_shape[0]):

            for j in range(PCA_movement_test_1_shape[1]):
                file.write(str(self.movement_1_test_PCA[i, j]))
                if(i == PCA_movement_test_1_shape[0] - 1) and (j == PCA_movement_test_1_shape[1] - 1):
                    file.write("}};\n\n ")
                elif ((j+1) % PCA_movement_test_1_shape[1] == 0) and (j != 0):
                    file.write("},{")
                if((j+1) % PCA_movement_test_1_shape[1] != 0) or (j == 0):
                    file.write( ", ")
        file.write("const float PCA_movement_test_2 [PCA_test_rows][PCA_test_cols] =")
        file.write(" {{")
        for i in range(PCA_movement_test_2_shape[0]):

            for j in range(PCA_movement_test_2_shape[1]):
                file.write(str(self.movement_2_test_PCA[i, j]))
                if(i == PCA_movement_test_2_shape[0] - 1) and (j == PCA_movement_test_2_shape[1] - 1):
                    file.write("}};\n\n ")
                elif ((j+1) % PCA_movement_test_2_shape[1] == 0) and (j != 0):
                    file.write("},{")
                if ((j+1) % PCA_movement_test_2_shape[1] != 0) or (j == 0):
                    file.write(", ")

        test_data = self.X_test.to_numpy()
        test_data_flat = test_data.reshape((1, test_data.shape[0] * test_data.shape[1]))
        test_data_flat = test_data_flat[0]
        file.write("#define test_rows {}\n".format(test_data.shape[0]))
        file.write("#define test_columns {}\n".format(test_data.shape[1]))
        file.write("const float test_data[test_rows][test_columns] = {{")
        for i in range(0, len(test_data_flat)):
            file.write(str(test_data_flat[i]))
            if i == (len(test_data_flat) - 1):
                file.write("}};\n")
            elif((i + 1) % test_data.shape[1] == 0) and (i != 0):
                file.write("},{")
            if((i + 1) % test_data.shape[1] != 0) or (i == 0):
                file.write(", ")
        file.write("typedef struct \n { \n ")
        file.write("int used_features[{}] ; \n".format(len(used_features)))
        file.write("}")
        file.write("params_mov_{}_vs_{}_t;\n".format(vs_movements[0], vs_movements[1]))
        file.write("params_mov_{}_vs_{}_t ".format(vs_movements[0], vs_movements[1]))
        file.write("params_mov_{}_vs_{}; \n".format(vs_movements[0], vs_movements[1]))
        file.write("void init_params_mov_{}_vs_{}();\n".format(vs_movements[0], vs_movements[1]))

        # ToDo: Get PCA1 and PCA2 Xmin,Xmax,min_scale and max scale
        # self.margin = 0.1  # 10% margin
        # self.scale_max = 1 + self.margin
        # self.scale_min = 1 - self.margin
        #
        # # max-min values
        # self.max_PCA_1 = self.mov[:,0].max()
        # self.max_PCA_2 = self.mov[:,1].max()
        # self.min_PCA_1 = self.mov[:,0].min()
        # self.min_PCA_2 = self.mov[:,1].min()
        # file.write("float margin_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        # file.write(str(self.margin) + "; \n")
        # file.write("float scale_max_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        # file.write(str(self.scale_max) + "; \n")
        # file.write("float scale_min_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        # file.write(str(self.scale_min) + "; \n")
        file.write("float PCA_1_max_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        file.write(str(self.max_PCA_1) + "; \n")
        file.write("float PCA_1_min_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        file.write(str(self.min_PCA_1) + "; \n")
        file.write("float PCA_2_max_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        file.write(str(self.max_PCA_2) + "; \n")
        file.write("float PCA_2_min_mov_{}_vs_{} = ".format(vs_movements[0],vs_movements[1]))
        file.write(str(self.min_PCA_2) + "; \n")
        file.write("#endif")
        file.close()

    def create_c_file(self):
        used_features = self.selected_features_rfe
        vs_movements = self.movement_pair
        file_name = "features_for_movements_{}_vs_{}.c".format(vs_movements[0], vs_movements[1])
        file = open(file_name, 'w')
        file.truncate(0)
        file.close()
        file = open(file_name, 'a+')
        file.write("#ifdef _FEATURES_FOR_MOVEMENTS_{}_vs_{}_\n".format(vs_movements[0], vs_movements[1]))
        file.write("#include ")
        file.write("\"features_for_movements_{}_vs_{}.h\" \n".format(vs_movements[0], vs_movements[1]))
        file.write("void init_params_mov_{}_vs_{}()\n".format(vs_movements[0], vs_movements[1]))
        file.write("{\n")
        for i in range(0, len(used_features)):
            file.write("params_mov_{}_vs_{}.used_features[{}] = ".format(vs_movements[0], vs_movements[1], i))
            file.write("{}_mov_{}_vs_{}".format(used_features[i], vs_movements[0], vs_movements[1]))
            file.write(";\n")
        file.write("}\n")
        file.write("#endif")
        file.close()
    def create_pickle_data(self):
        self.save_output_data("Xtrain.pkl",self.X_train)
        self.save_output_data("ytrain.pkl", pd.DataFrame(self.y_train, columns=['label']))
        self.save_output_data("Xtest.pkl",self.X_test)
        self.save_output_data("ytest.pkl", pd.DataFrame(self.y_test, columns=['label']))

    def save_output_data(self, name, dataframe):
        dataframe.to_pickle(name)

    def read_pickle_data(self):
        x_train = self.read_output_data("Xtrain.pkl")
        y_train = self.read_output_data("ytrain.pkl")
        x_test = self.read_output_data("Xtest.pkl")
        y_test = self.read_output_data("ytest.pkl")

        return x_train,y_train,x_test,y_test

    def read_output_data(self,name):
        return pd.read_pickle(name)

    def get_keys(self,dict):
        list = []
        for k in dict.keys():
            list.append(k)
        return list
    def import_nina_data(self):
        # load data
        # S1: 50% of forearm
        s1_e2_a1 = loadmat('S1_E2_A1.mat')
        # S2: 70% of forearm
        s2_e2_a1 = loadmat('S2_E2_A1.mat')
        # S3: 30% of forearm
        s3_e2_a1 = loadmat('S3_E2_A1.mat')
        # S5: 90% of forearm
        s5_e2_a1 = loadmat('S5_E2_A1.mat')

        self.raw_data = (s1_e2_a1, s2_e2_a1, s3_e2_a1, s5_e2_a1)

    def get_subject_dataframes(self):
        for i in range(0, len(self.raw_data)):
            self.raw_keys[i] = self.get_keys(self.raw_data[i])

        for i in range(0, len(self.raw_data)):
            self.keys[i] = self.raw_keys[i][3: len(self.raw_keys[i])]
        # emg data
        emg_signals = {}
        for i in range(0, len(self.raw_data)):
            emg_signals[i] = np.array(self.raw_data[i][self.keys[i][0]])
            print("emg_signals", emg_signals[i])

        # restim data (labels)
        restim = {}
        for i in range(0, len(self.raw_data)):
            restim[i] = np.array(self.raw_data[i][self.keys[i][8]])
            print("restim", restim[i])

        # rerepetition (repetitions)
        rerep = {}
        for i in range(0, len(self.raw_data)):
            rerep[i] = np.array(self.raw_data[i][self.keys[i][9]])
            print("rerep", rerep[i])
        # stack data
        data ={}
        for j in range(0, len(self.raw_data)):
            data[j] = np.vstack(([emg_signals[j][:, self.channels[i]] for i in range(0, self.channels.shape[0])], restim[j][:, 0],
                                 rerep[j][:, 0]))
            data[j] = data[j].T

        # create DataFrames
        self.Subject = {}
        column_labels = ['emg_{}'.format(i) for i in self.channels]
        column_labels.append('label')
        column_labels.append('repetition')
        for i in range(0, len(self.raw_data)):
            self.Subject[i] = pd.DataFrame(data[i], columns=column_labels)

    def segment_movement(self,movement):
        movement_segmented = {}
        min_rep = int(np.min(movement[['repetition']].to_numpy()))
        max_rep = int(np.max(movement[['repetition']].to_numpy()))

        for j in range(min_rep, max_rep + 1):
            # repetition j
            movement_segmented[j] = movement[movement.repetition == j]
            rows = len(movement_segmented[j].index)
            segments = np.array([i // self.samples_per_window for i in range(0, rows)])
            movement_segmented[j]['segment'] = segments
        return movement_segmented
    def get_segment_data(self):
        movements_data = {}

        for i in range(0, len(self.movements_to_classify)):
            for j in range(0, len(self.raw_data)):
                movements_data[i, j] = self.Subject[j][self.Subject[j].label == self.movements_to_classify[i]]

        for i in range(0, len(self.movements_to_classify)):
            for j in range(0, len(self.raw_data)):
                self.movements_data_segmented[i, j] = self.segment_movement(movements_data[i, j])
    def extract_features(self):
        for i in range(0, len(self.movements_to_classify)):
            for j in range(0, len(self.raw_data)):
                for k in range(0, len(self.features_list)):
                    self.get_feature(self.movements_data_segmented[i, j], self.channels, self.features_list[k])

    def get_feature(self,segmented_movement, channels_list, feature):

        number_of_rep = len(segmented_movement)

        # Iterate over repetition
        for rep_idx in range(1, number_of_rep + 1):
            number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
            number_of_emg_channels = channels_list.shape[0]

            # create feature columns
            for i in range(0, number_of_emg_channels):
                segmented_movement[rep_idx][feature + '_emg_{}'.format(i)] = np.empty(
                    len(segmented_movement[rep_idx]))
                segmented_movement[rep_idx][feature + '_emg_{}'.format(i)][:] = np.NaN

            # Iterate over segments
            for segment_idx in range(0, number_of_segments - 1):
                # get segment indexes
                segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
                data_segment = segmented_movement[rep_idx][segment_filter]

                # next segment
                if segment_idx < number_of_segments - 1:
                    next_segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx + 1
                    next_data_segment = segmented_movement[rep_idx][next_segment_filter]

                feature_obtained = np.zeros((number_of_segments, number_of_emg_channels))
                for i in range(0, number_of_emg_channels):
                    if feature == 'MAV':
                        # MAV per segment
                        feature_obtained[segment_idx, i] = np.mean(np.abs(data_segment['emg_{}'.format(i)]))
                    elif feature == 'ZC':
                        # ZC per segment
                        feature_obtained[segment_idx, i] = self.ZC_function(data_segment['emg_{}'.format(i)])
                    elif feature == 'WL':
                        feature_obtained[segment_idx, i] = self.WL_function(data_segment['emg_{}'.format(i)])
                    elif feature == 'RMS':
                        feature_obtained[segment_idx, i] = self.RMS_function(data_segment['emg_{}'.format(i)])
                    elif feature == 'SSC':
                        feature_obtained[segment_idx, i] = self.SSC_function(data_segment['emg_{}'.format(i)])
                    elif feature == 'MAVS':
                        if i < number_of_segments - 1:
                            feature_obtained[segment_idx, i] = np.mean(np.abs(next_data_segment['emg_{}'.format(i)])) \
                                                               - np.mean(np.abs(data_segment['emg_{}'.format(i)]))
                        else:
                            feature_obtained[segment_idx, i] = - np.mean(
                                np.abs(data_segment['emg_{}'.format(i)]))
                    # add obtained feature to dataframe
                    segmented_movement[rep_idx].loc[segment_filter, feature + '_emg_{}'.format(i)] = feature_obtained[
                        segment_idx, i]
    def ZC_function(self,data_segment):
        x_i = data_segment.to_numpy()
        x_i_p1 = np.concatenate((x_i[1:], 0), axis=None)
        threshold = 1e-5
        abs_sub = np.abs(x_i - x_i_p1)
        prod_x = np.prod((x_i, x_i_p1), axis=0)
        sign_prod = np.sign(prod_x)
        count_vector = sign_prod[np.where(abs_sub > threshold)]
        ZC_feature = np.sum(np.abs(count_vector[np.where(count_vector == -1)]))

        return ZC_feature

    def RMS_function(self, data_segment):
        x_i = data_segment.to_numpy()
        RMS = np.sqrt((1 / x_i.shape[0]) * np.dot(x_i, x_i))
        return RMS

    def WL_function(self, data_segment):
        x_i = data_segment.to_numpy()
        x_i_p1 = np.concatenate((x_i[1:], 0), axis=None)
        abs_sub = np.abs(x_i_p1 - x_i)
        WL_feature = sum(abs_sub)
        return WL_feature

    def SSC_function(self, data_segment):
        x_i = data_segment.to_numpy()
        x_i_m1 = np.concatenate((0, x_i[:-1]), axis=None)
        x_i_p1 = np.concatenate((x_i[1:], 0), axis=None)
        ones = np.array([1 for i in range(0, x_i.shape[0])])
        ones = ones[1:-1]
        threshold = 1e-5
        abs_sub_p = np.abs(x_i - x_i_p1)
        abs_sub_m = np.abs(x_i - x_i_m1)
        c1 = (x_i > x_i_m1)
        c2 = (x_i > x_i_p1)
        c3 = (x_i < x_i_m1)
        c4 = (x_i < x_i_p1)
        condition_comp = (c1 & c2) | (c3 & c4)
        c1_th = (abs_sub_m > threshold)
        c2_th = (abs_sub_p > threshold)
        condition_threshold = c1_th | c2_th
        final_condition = condition_comp & condition_threshold
        final_condition = final_condition[1:-1]
        SSC = np.sum(ones[np.where(final_condition)])
        return SSC
    def select_features(self):
        self.movement_1 = {}
        self.movement_2 = {}
        for i in range(0, len(self.raw_data)):
            self.movement_1[i] = self.concatenate_movement_repetition(self.movements_data_segmented[self.movement_pair[0], i])
            self.movement_2[i] = self.concatenate_movement_repetition(self.movements_data_segmented[self.movement_pair[1], i])

        for i in range(0, len(self.raw_data)):
            if i == 0:
                self.movement_1_complete = self.movement_1[0]
                self.movement_2_complete = self.movement_2[0]

            else:
                self.movement_1_complete = pd.concat([self.movement_1_complete, self.movement_1[i]])
                self.movement_2_complete = pd.concat([self.movement_2_complete, self.movement_2[i]])

        self.movement_1_vs_movement_2 = pd.concat([self.movement_1_complete, self.movement_2_complete])
        #ToDo: probar que el rango este bien
        movement_1_vs_movement_2_feat = self.movement_1_vs_movement_2.iloc[:,
                                        [i for i in range(-self.features_per_channel * self.channels.shape[0],
                                                          0)]]
        movement_1_vs_movement_2_feat['label'] = self.movement_1_vs_movement_2['label']

        df = movement_1_vs_movement_2_feat
        X = df.drop('label', 1)
        y = df['label']

        cols = list(X.columns)
        model = LinearRegression()
        # Initializing RFE model
        rfe = RFE(model, n_features_to_select=6)
        # Transforming data using RFE
        X_rfe = rfe.fit_transform(X, y)
        # Fitting the data to model
        model.fit(X_rfe, y)
        temp = pd.Series(rfe.support_, index=cols)
        self.selected_features_rfe = temp[temp == True].index
        print("selected features rfe:", self.selected_features_rfe)


    def concatenate_movement_repetition(self,movement_to_conc):
        # concatenate repetitions
        complete_movement = pd.concat(movement_to_conc)
        # remove NaN (segment size < window)
        complete_movement.dropna(subset=["MAV_emg_0"], inplace=True)

        return complete_movement

    def reduce_features_PCA(self):
        n_components = 2
        self.pca_rfe = PCA(n_components=n_components)
        self.pcafit_rfe = self.pca_rfe.fit(self.movement_1_vs_movement_2[self.selected_features_rfe])
        print("shape of data to transform (before pca transform)", self.movement_1_complete[self.selected_features_rfe].shape)
        # print("data to transform ",self.movement_1_complete[self.selected_features_rfe]- (np.mean(self.movement_1_complete[self.selected_features_rfe], axis=0)))
        print("data to transform (before pca transform)", self.movement_1_complete[self.selected_features_rfe] )
        self.reduced_movement_1_rfe = self.pcafit_rfe.transform(self.movement_1_complete[self.selected_features_rfe])
        self.reduced_movement_2_rfe = self.pcafit_rfe.transform(self.movement_2_complete[self.selected_features_rfe])
        # -USED FOR MICROCONTROLLER IMPLEMENTATION OF PCA-
        self.PCA_matrix = self.pca_rfe.components_
        self.PCA_mean = self.pca_rfe.mean_
        # ------------------------------------------------
        print("shape of data to transform ", self.movement_1_complete[self.selected_features_rfe].shape)
        # print("data to transform ",self.movement_1_complete[self.selected_features_rfe]- (np.mean(self.movement_1_complete[self.selected_features_rfe], axis=0)))
        print("data to transform ", self.movement_1_complete[self.selected_features_rfe] )
        data = self.movement_1_complete[self.selected_features_rfe].to_numpy()
        data_to_transform = (data - self.pca_rfe.mean_)

        print("pca matrix", self.PCA_matrix)
        print("pca matrix shape", self.PCA_matrix.shape)
        print("pca mean", self.PCA_mean)
        print("shape of data to transform (using to_numpy method)", data_to_transform.shape)
        print("data to transform (using to_numpy method)", data_to_transform)

        x_mult = np.matmul(data_to_transform, self.PCA_matrix.T)


        # comp = self.reduced_movement_1_rfe == x_mult
        # equ = comp.all()
        print("PCA*X", x_mult)
        print("PCA*X shape", x_mult.shape)
        print("PCA transform", self.reduced_movement_1_rfe)
        print("PCA transform shape", self.reduced_movement_1_rfe.shape)
        equ = np.array_equal(x_mult,self.reduced_movement_1_rfe)
        if equ:
            print("conversion por PCA y por multiplicación de matrices es igual")
        else:
            print("conversion por PCA y por multiplicación de matrices es diferente")



    def normalize_training_data_PCA(self):
        label_1 = np.array([1 for i in range(0, len(self.reduced_movement_1_rfe))])
        label_2 = np.array([0 for i in range(0, len(self.reduced_movement_2_rfe))])
        self.y = np.concatenate((label_1, label_2), axis=None)
        self.mov = np.vstack((self.reduced_movement_1_rfe, self.reduced_movement_2_rfe))
        print("mov shape", self.mov.shape)
        # margin
        # self.margin = 0.1  # 10% margin
        # self.margin = 0
        # self.scale_max = 1 + self.margin
        # self.scale_min = 1 - self.margin

        # max-min values
        self.max_PCA_1 = self.mov[:,0].max()
        self.max_PCA_2 = self.mov[:,1].max()
        self.min_PCA_1 = self.mov[:,0].min()
        self.min_PCA_2 = self.mov[:,1].min()

        self.reduced_mov = np.zeros_like(self.mov)
        # self.reduced_mov[:, 0] = (self.mov[:, 0] - (self.scale_min * self.min_PCA_1)) / (
        #             (self.scale_max * self.max_PCA_1) - (self.scale_min * self.min_PCA_1))
        # self.reduced_mov[:, 1] = (self.mov[:, 1] - (self.scale_min * self.min_PCA_2)) / (
        #             (self.scale_max * self.max_PCA_2) - (self.scale_min * self.min_PCA_2))

        self.reduced_mov[:, 0] = (self.mov[:, 0] - (self.min_PCA_1)) / (
                    (self.max_PCA_1) - (self.min_PCA_1))
        self.reduced_mov[:, 1] = (self.mov[:, 1] - (self.min_PCA_2)) / (
                    (self.max_PCA_2) - (self.min_PCA_2))

        df = pd.DataFrame(self.reduced_mov, columns=['PCA_1', 'PCA_2'])
        self.X = df[['PCA_1', 'PCA_2']]

    def split_data(self,test_size=0.004,train_size=0.007):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, train_size=train_size,
                                                            random_state=0)

    def fitting(self):
        # Create training and testing samples

        print("shape of X_train", self.X_train.shape)
        print("shape of X_test", self.X_test.shape)
        print("shape of y_train", self.y_train.shape)
        print("shape of y_test", self.y_test.shape)
        print("X_train", self.X_train, "y_train", self.y_train)
        # Fit the model
        # Note, available kernels: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        model = SVC(kernel='rbf', probability=True, C=self.C, gamma=self.gamma)
        self.clf = model.fit(self.X_train, self.y_train)
        print("values of clf", self.clf.predict_proba(self.X))

        # Predict class labels on training data
        self.pred_labels_tr = model.predict(self.X_train)
        # Predict class labels on a test data
        self.pred_labels_te = model.predict(self.X_test)

        # Use score method to get accuracy of the model
        print('----- Evaluation on Test Data -----')
        score_te = model.score(self.X_test, self.y_test)
        print('Accuracy Score: ', score_te)
        # Look at classification report to evaluate the model
        print(classification_report(self.y_test, self.pred_labels_te))
        print('--------------------------------------------------------')

        print('----- Evaluation on Training Data -----')
        score_tr = model.score(self.X_train, self.y_train)
        print('Accuracy Score: ', score_tr)
        # Look at classification report to evaluate the model
        print(classification_report(self.y_train, self.pred_labels_tr))
        print('--------------------------------------------------------')

    def Plot_3D(self):
        # Specify a size of the mesh to be used
        # mesh_size = 1e-5 # for small valued data
        mesh_size = 0.01  # for normal data X E [0,1]
        margin = 0

        # Create a mesh grid on which we will run our model
        x_min, x_max = self.X.iloc[:, 0].fillna(self.X.mean()).min() - margin, self.X.iloc[:, 0].fillna(self.X.mean()).max() + margin
        y_min, y_max = self.X.iloc[:, 1].fillna(self.X.mean()).min() - margin, self.X.iloc[:, 1].fillna(self.X.mean()).max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)
        print("xx shape", xx.shape)
        print("xx, yy", (xx, yy))
        # Calculate predictions on grid
        Z = self.clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        print("Values of Z", Z)

        # Create a 3D scatter plot with predictions
        fig = px.scatter_3d(x=self.X_test['PCA_1'], y=self.X_test['PCA_2'], z=self.y_test,
                            opacity=0.8, color_discrete_sequence=['black'])

        # Set figure title and colors
        fig.update_layout(  # title_text="Scatter 3D Plot with SVM Prediction Surface",
            paper_bgcolor='white',
            scene=dict(xaxis=dict(backgroundcolor='white',
                                  color='black',
                                  gridcolor='#f0f0f0'),
                       yaxis=dict(backgroundcolor='white',
                                  color='black',
                                  gridcolor='#f0f0f0'
                                  ),
                       zaxis=dict(backgroundcolor='lightgrey',
                                  color='black',
                                  gridcolor='#f0f0f0',
                                  )))
        # Update marker size
        fig.update_traces(marker=dict(size=1))

        # Add prediction plane
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='SVM Prediction',
                                  colorscale='RdBu', showscale=False,
                                  contours={"z": {"show": True, "start": 0.2, "end": 0.8, "size": 0.05}}))
        fig.show()
    def get_test_data(self):
        print("movements data segmented", self.movements_data_segmented)
        print("movement 1:", self.movement_1_complete)
        print("movement 1 labels:", np.unique(self.movement_1_complete['label']))
        print("movement 2:", self.movement_2_complete)
        print("movement 2 labels:", np.unique(self.movement_2_complete['label']))
        movement = 0
        repetition = 1
        electrode = 0
        subject = 0
        segment = 2
        data_emg = self.movement_1[subject]['emg_{}'.format(electrode)].to_numpy()
        data_repetition = self.movement_1[subject]['repetition'].to_numpy()
        data_segment = self.movement_1[subject]['segment'].to_numpy()
        filtered_data = self.movement_1[subject][(self.movement_1[subject]['repetition']==repetition) & (self.movement_1[subject]['segment']==segment)]
        filtered_data_emg = filtered_data['emg_{}'.format(electrode)].to_numpy()
        fig1, axs1 = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
        t_dim1 = data_emg.shape[0]
        t_1 = np.array([i for i in range(0, t_dim1)])
        axs1[0].set_title('Subject {}: EMG signal of movement {}, channel {}'.format(subject,movement,electrode))
        axs1[0].plot(t_1, data_emg)

        axs1[1].set_title('repetition')
        axs1[1].plot(t_1,data_repetition)

        axs1[2].set_title('segment')
        axs1[2].plot(t_1, data_segment)

        filtered_data_emg_fill = np.zeros(t_dim1)
        filtered_data_emg_fill[0:filtered_data_emg.shape[0]] = filtered_data_emg
        axs1[3].set_title('filtered segment')
        axs1[3].plot(t_1, filtered_data_emg_fill)
        plt.show()

        # Create test data for microcroontroller
        # movement 1
        print("Movement 1 for testing")
        subject = random.randint(0,len(self.raw_data) - 1)
        print('repetition values', np.unique(self.movement_1[subject]['repetition'].to_numpy()))
        max_rep = np.amax(np.unique(self.movement_1[subject]['repetition'].to_numpy()))
        min_rep = np.amin(np.unique(self.movement_1[subject]['repetition'].to_numpy()))
        repetition = random.randint(min_rep,max_rep)
        print("max_rep", max_rep, "min_rep", min_rep, "repetition", repetition)
        rep_dataframe = self.movement_1[subject][(self.movement_1[subject]['repetition']==repetition)]
        max_segment = np.amax(np.unique(rep_dataframe['segment']))
        min_segment = np.amin(np.unique(rep_dataframe['segment']))
        used_segments = 4
        self.used_segments = used_segments
        segment = random.randint(min_segment,max_segment-used_segments)
        print("max_segment", max_segment, "min_segment", min_segment, "segment", segment)
        self.movement_1_test = {}
        self.movement_1_features_test = {}

        number_of_emg_channels = self.channels.shape[0]
        self.number_of_emg_channels = number_of_emg_channels
        for i in range(0,used_segments):
            filtered_data = self.movement_1[subject][(self.movement_1[subject]['repetition'] == repetition) & (
                    self.movement_1[subject]['segment'] == (segment + i))]
            self.movement_1_features_test[i] = filtered_data.iloc[0,
                                          [k for k in range(-self.features_per_channel * self.channels.shape[0],
                                                            0)]]
            for j in range(0,number_of_emg_channels):
                self.movement_1_test[i,j] = filtered_data['emg_{}'.format(j)].to_numpy()

        # print("emg signals of movement 1 to test", self.movement_1_test)
        # print("its len",len(self.movement_1_test))
        print("emg features of movement 1 to test", self.movement_1_features_test)
        # movement 2
        print("Movement 2 for testing")
        subject = random.randint(0, len(self.raw_data) - 1)
        print('repetition values', np.unique(self.movement_2[subject]['repetition'].to_numpy()))
        max_rep = np.amax(np.unique(self.movement_2[subject]['repetition'].to_numpy()))
        min_rep = np.amin(np.unique(self.movement_2[subject]['repetition'].to_numpy()))
        repetition = random.randint(min_rep, max_rep)
        print("max_rep", max_rep, "min_rep", min_rep, "repetition", repetition)
        rep_dataframe = self.movement_2[subject][(self.movement_2[subject]['repetition'] == repetition)]
        max_segment = np.amax(np.unique(rep_dataframe['segment']))
        min_segment = np.amin(np.unique(rep_dataframe['segment']))
        used_segments = 4
        segment = random.randint(min_segment, max_segment - used_segments)
        print("max_segment", max_segment, "min_segment", min_segment, "segment", segment)
        self.movement_2_test = {}
        self.movement_2_features_test = {}
        number_of_emg_channels = self.channels.shape[0]
        self.movement_2_features_test = {}
        for i in range(0, used_segments):
            filtered_data = self.movement_2[subject][(self.movement_2[subject]['repetition'] == repetition) & (
                    self.movement_2[subject]['segment'] == (segment + i))]
            self.movement_2_features_test[i] = filtered_data.iloc[0,
                                          [k for k in range(-self.features_per_channel * self.channels.shape[0],
                                                            0)]]
            for j in range(0, number_of_emg_channels):
                self.movement_2_test[i, j] = filtered_data['emg_{}'.format(j)].to_numpy()
        # print("emg signals of movement 2 to test", self.movement_2_test)
        # print("its len", len(self.movement_2_test))
        print("emg features of movement 2 to test", self.movement_2_features_test)
        # PCA transform data
        #ToDo: Observar el procedimiento para obtener la transformacion PCA en Reduce_features_PCA y aplicarlo en
        # movement_1_features_test y movement_2_features_test. Almacenar lo obtenido en PCA y las señales emg en un archivo
        # *.h y comparar con lo obtenido en el lenguaje C.
        self.movement_1_features_reduced = {}
        self.movement_2_features_reduced = {}
        self.movement_1_reduced = np.zeros((used_segments,len(self.selected_features_rfe)))
        self.movement_2_reduced = np.zeros((used_segments,len(self.selected_features_rfe)))
        # print("used features:", self.selected_features_rfe)
        # print("test reduction:", self.movement_1_features_test[0])
        # print("test reduction 2:", self.movement_1_features_test[0][self.selected_features_rfe])
        # Reduce features
        for i in range(0,used_segments):
            self.movement_1_features_reduced[i] = self.movement_1_features_test[i][self.selected_features_rfe]
            self.movement_1_reduced[i] = self.movement_1_features_reduced[i].to_numpy()
            self.movement_2_features_reduced[i] = self.movement_2_features_test[i][self.selected_features_rfe]
            self.movement_2_reduced[i] = self.movement_2_features_reduced[i].to_numpy()
        print("reduced features movement 1", self.movement_1_reduced)
        print("reduced features movement 2", self.movement_2_reduced)
        self.movement_1_test_PCA = np.matmul(self.movement_1_reduced - self.pca_rfe.mean_, self.PCA_matrix.T)
        self.movement_2_test_PCA = np.matmul(self.movement_2_reduced - self.pca_rfe.mean_, self.PCA_matrix.T)
        print("PCA movement 1 for testing:", self.movement_1_test_PCA)
        print("PCA movement 2 for testing:", self.movement_2_test_PCA)
        print("Normalized PCA obtained via normalize_training_data_PCA function:\n")
        print(self.X)
        print(self.X.min())
        print(self.X.max())


    def test_program(self):
        tic = time.time()
        self.import_nina_data()
        self.get_subject_dataframes()
        self.get_segment_data()

        self.extract_features()
        self.select_features()
        self.reduce_features_PCA()
        self.normalize_training_data_PCA()
        self.split_data()
        self.fitting()
        # self.create_h_file()
        # self.create_c_file()
        print("data to write", self.X_train,self.y_train,self.X_test,self.y_test)
        self.create_pickle_data()
        x_tra, y_tra, x_te,y_te = self.read_pickle_data()
        print("read data",x_tra, y_tra, x_te,y_te)
        self.get_test_data()
        self.create_h_file()
        self.create_c_file()
        toc = time.time()
        tic_toc = toc - tic
        print("Execution time:", tic_toc)
        self.Plot_3D()
