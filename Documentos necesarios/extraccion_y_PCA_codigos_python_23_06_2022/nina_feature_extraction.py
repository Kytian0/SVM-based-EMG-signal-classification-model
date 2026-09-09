from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm

def get_keys(dict):
    list = []
    for k in dict.keys():
        list.append(k)
    return list

def segment_movement(movement):
    movement_segmented = {}
    min_rep = int(np.min(movement[['repetition']].to_numpy()))
    max_rep = int(np.max(movement[['repetition']].to_numpy()))

    for j in range(min_rep, max_rep+1):
        # repetition j
        movement_segmented[j] = movement[movement.repetition == j]
        rows = len(movement_segmented[j].index)
        segments = np.array([i // samples_per_window for i in range(0, rows)])
        movement_segmented[j]['segment'] = segments
    return movement_segmented
def normalize_min_max_(segmented_movement,feature,channels_list):
    number_of_rep = len(segmented_movement)
    number_of_emg_channels = channels_list.shape[0]
    for rep_idx in range(1, number_of_rep + 1):
        # create normalized feature columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx][feature + '_normal_emg_{}'.format(i)] = np.zeros(
                len(segmented_movement[rep_idx]))

        # calculate normalized feature
        for j in range(0, number_of_emg_channels):
            feature_data = segmented_movement[rep_idx][feature +'_emg_{}'.format(j)]
            feature_data_norm = (feature_data - feature_data.min())/(feature_data.max() - feature_data.min())
            segmented_movement[rep_idx][feature +'_normal_emg_{}'.format(j)] = feature_data_norm


def get_MAV(segmented_movement, channels_list):

    number_of_rep = len(segmented_movement)

    # Iterate over repetition
    for rep_idx in range(1, number_of_rep + 1):
        number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
        number_of_emg_channels = channels_list.shape[0]

        # create mean columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx]['MAV_emg_{}'.format(i)] = np.empty(
                len(segmented_movement[rep_idx]))
            segmented_movement[rep_idx]['MAV_emg_{}'.format(i)][:] = np.NaN

        # Iterate over segments
        for segment_idx in range(0, number_of_segments - 1):
            # get segment indexes
            segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
            data_segment = segmented_movement[rep_idx][segment_filter]

            mean = np.zeros((number_of_segments, number_of_emg_channels))
            for i in range(0, number_of_emg_channels):
                # MAV per segment
                mean[segment_idx, i] = np.mean(np.abs(data_segment['emg_{}'.format(i)]))
                # add MAV to dataframe
                segmented_movement[rep_idx].loc[segment_filter, 'MAV_emg_{}'.format(i)] = mean[
                    segment_idx, i]

def WL_function(data_segment):
    x_i = data_segment.to_numpy()
    x_i_p1 = np.concatenate((x_i[1:],0), axis=None)
    abs_sub = np.abs(x_i_p1 - x_i)
    WL_feature = sum(abs_sub)
    return WL_feature




def get_WL(segmented_movement, channels_list):

    number_of_rep = len(segmented_movement)

    # Iterate over repetition
    for rep_idx in range(1, number_of_rep + 1):
        number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
        number_of_emg_channels = channels_list.shape[0]

        # create WL columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx]['WL_emg_{}'.format(i)] = np.empty(
                len(segmented_movement[rep_idx]))
            segmented_movement[rep_idx]['WL_emg_{}'.format(i)][:] = np.NaN
        # Iterate over segments
        for segment_idx in range(0, number_of_segments - 1):
            # get segment indexes
            segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
            data_segment = segmented_movement[rep_idx][segment_filter]

            WL = np.zeros((number_of_segments, number_of_emg_channels))
            for i in range(0, number_of_emg_channels):
                # WL per segment
                WL[segment_idx,i] = WL_function(data_segment['emg_{}'.format(i)])
                # add WL to dataframe
                segmented_movement[rep_idx].loc[segment_filter, 'WL_emg_{}'.format(i)] = WL[
                    segment_idx, i]
def ZC_function(data_segment):
    x_i = data_segment.to_numpy()
    x_i_p1 = np.concatenate((x_i[1:],0), axis=None)
    threshold = 1e-5
    abs_sub = np.abs(x_i - x_i_p1)
    prod_x = np.prod((x_i, x_i_p1), axis=0)
    sign_prod = np.sign(prod_x)
    count_vector = sign_prod[np.where(abs_sub > threshold)]
    ZC_feature = np.sum(np.abs(count_vector[np.where(count_vector == -1)]))

    return ZC_feature


def get_ZC(segmented_movement, channels_list):

    number_of_rep = len(segmented_movement)

    # Iterate over repetition
    for rep_idx in range(1, number_of_rep + 1):
        number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
        number_of_emg_channels = channels_list.shape[0]

        # create ZC columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx]['ZC_emg_{}'.format(i)] = np.empty(
                len(segmented_movement[rep_idx]))
            segmented_movement[rep_idx]['ZC_emg_{}'.format(i)][:] = np.NaN

        # Iterate over segments
        for segment_idx in range(0, number_of_segments - 1):
            # get segment indexes
            segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
            data_segment = segmented_movement[rep_idx][segment_filter]

            ZC = np.zeros((number_of_segments, number_of_emg_channels))
            for i in range(0, number_of_emg_channels):
                # ZC per segment
                ZC[segment_idx,i] = ZC_function(data_segment['emg_{}'.format(i)])
                # add ZC to dataframe
                segmented_movement[rep_idx].loc[segment_filter, 'ZC_emg_{}'.format(i)] = ZC[
                    segment_idx, i]
def RMS_function(data_segment):
    x_i = data_segment.to_numpy()
    RMS = np.sqrt((1/x_i.shape[0])*np.dot(x_i,x_i))
    return RMS

def get_RMS(segmented_movement, channels_list):

    number_of_rep = len(segmented_movement)

    # Iterate over repetition
    for rep_idx in range(1, number_of_rep + 1):
        number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
        number_of_emg_channels = channels_list.shape[0]

        # create RMS columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx]['RMS_emg_{}'.format(i)] = np.empty(
                len(segmented_movement[rep_idx]))
            segmented_movement[rep_idx]['RMS_emg_{}'.format(i)][:] = np.NaN

        # Iterate over segments
        for segment_idx in range(0, number_of_segments - 1):
            # get segment indexes
            segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
            data_segment = segmented_movement[rep_idx][segment_filter]

            RMS = np.zeros((number_of_segments, number_of_emg_channels))
            for i in range(0, number_of_emg_channels):
                # RMS per segment
                RMS[segment_idx,i] = RMS_function(data_segment['emg_{}'.format(i)])
                # add RMS to dataframe
                segmented_movement[rep_idx].loc[segment_filter, 'RMS_emg_{}'.format(i)] = RMS[
                    segment_idx, i]

def SSC_function(data_segment):
    x_i = data_segment.to_numpy()
    x_i_m1 = np.concatenate((0,x_i[:-1]), axis=None)
    x_i_p1 = np.concatenate((x_i[1:], 0), axis=None)
    ones = np.array([1 for i in range(0,x_i.shape[0])])
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
    SSC = np.sum(ones[np.where(final_condition)])
    return SSC


def get_SSC(segmented_movement, channels_list):

    number_of_rep = len(segmented_movement)

    # Iterate over repetition
    for rep_idx in range(1, number_of_rep + 1):
        number_of_segments = len(np.unique(segmented_movement[rep_idx]['segment']))
        number_of_emg_channels = channels_list.shape[0]

        # create SSC columns
        for i in range(0, number_of_emg_channels):
            segmented_movement[rep_idx]['SSC_emg_{}'.format(i)] = np.empty(
                len(segmented_movement[rep_idx]))
            segmented_movement[rep_idx]['SSC_emg_{}'.format(i)][:] = np.NaN

        # Iterate over segments
        for segment_idx in range(0, number_of_segments - 1):
            # get segment indexes
            segment_filter = segmented_movement[rep_idx]['segment'] == segment_idx
            data_segment = segmented_movement[rep_idx][segment_filter]

            SSC = np.zeros((number_of_segments, number_of_emg_channels))
            for i in range(0, number_of_emg_channels):
                # SSC per segment
                SSC[segment_idx,i] = SSC_function(data_segment['emg_{}'.format(i)])
                # add SSC to dataframe
                segmented_movement[rep_idx].loc[segment_filter, 'SSC_emg_{}'.format(i)] = SSC[
                    segment_idx, i]
def get_correlation(movement_to_corr,channels,features_per_channel):
    # concatenate repetitions
    complete_movement = pd.concat(movement_to_corr)
    # remove NaN (segment size < window)
    complete_movement.dropna(subset=["MAV_emg_0"], inplace=True)
    # get correlation
    cor = complete_movement.iloc[:, [i for i in range(-2 * features_per_channel * channels.shape[0],
                                                      -features_per_channel * channels.shape[0])]].corr()
    return (cor, complete_movement)


# load databases
#S1: 50% of forearm
s1_e2_a1 = loadmat('S1_E2_A1.mat')
#S2: 70% of forearm
s2_e2_a1 = loadmat('S2_E2_A1.mat')
#S3: 30% of forearm
s3_e2_a1 = loadmat('S3_E2_A1.mat')
#S5: 90% of forearm
s5_e2_a1 = loadmat('S5_E2_A1.mat')
#S7: 0% of forearm
# s7_e2_a1 = loadmat('S7_E2_A1.mat')



#MATLAB fields in databases
raw_keys = get_keys(s1_e2_a1)
raw_keys2 = get_keys(s2_e2_a1)
raw_keys3 = get_keys(s3_e2_a1)
raw_keys5 = get_keys(s5_e2_a1)
# raw_keys7 = get_keys(s7_e2_a1)

#fields without MATLAB headers
keys = raw_keys[3: len(raw_keys)]
keys2 = raw_keys2[3: len(raw_keys2)]
keys3 = raw_keys3[3: len(raw_keys3)]
keys5 = raw_keys5[3: len(raw_keys5)]
# keys7 = raw_keys5[3: len(raw_keys5)]

print("keys 1:", keys)
print("keys 2:", keys2)
print("keys 3:", keys3)
print("keys 5:", keys5)
# print("keys 7:", keys7)

# plot control flags
plot_all = False
plot_raw = False
plot_segmented = False
plot_seg_and_features = False
plot_complete_movement_features = False
plot_total_correlation = False

# TODO: 1. Añadir los datos de S3,S5 y S7 hasta el procedimiento de matriz de correlacion -> ok
# TODO: 2. HALLAR UNA MATRIZ GENERAL DE CORRELACION Y SELECCIONAR CARACTERISCAS -> en proceso
# TODO: 3. PCA de 2 ejes con las características seleccionadas
#emg data
emg_s1_e2 = np.array(s1_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre) - S1
emg_s2_e2 = np.array(s2_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre) - S2
emg_s3_e2 = np.array(s3_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre) - S3
emg_s5_e2 = np.array(s5_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre) - S5
# emg_s7_e2 = np.array(s7_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre) - S7

#labels (restimulus)
restim_s1= np.array(s1_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre) - S1
restim_s2= np.array(s2_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre) - S2
restim_s3= np.array(s3_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre) - S3
restim_s5= np.array(s5_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre) - S5
# restim_s7= np.array(s7_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre) - S7

# repetitions (rerepetition)
rerep_s1= np.array(s1_e2_a1[keys[9]]) # exercise 2 - grasping movements (agarre) - S1
rerep_s2= np.array(s2_e2_a1[keys[9]]) # exercise 2 - grasping movements (agarre) - S2
rerep_s3= np.array(s3_e2_a1[keys[9]]) # exercise 2 - grasping movements (agarre) - S3
rerep_s5= np.array(s5_e2_a1[keys[9]]) # exercise 2 - grasping movements (agarre) - S5
# rerep_s7= np.array(s7_e2_a1[keys[9]]) # exercise 2 - grasping movements (agarre) - S7

# channels = np.array([0,2,4,6,10,11]) # sensor band, biceps brachi and triceps brachi sensors - original
channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
# plot raw emg
if plot_all | plot_raw:
    #subject 1 plots
    fig, axs = plt.subplots(nrows=18, ncols=1, figsize=(20, 10))
    t_dim= emg_s1_e2.shape[0]
    # channels = np.array([0,2,4,6,10,11]) # sensor band, biceps brachi and triceps brachi sensors
    for i in range(0,channels.shape[0]):
        print('emg signal: ', emg_s1_e2[:,channels[i]])
        t = np.array([i for i in range(0,t_dim)])

        axs[3*i].set_title('Subject 1: EMG signal exercise 2 (all movements), channel {}' .format(channels[i]))
        axs[3*i].plot(t,emg_s1_e2[:,channels[i]])

        axs[(3*i) + 1].set_title('movement label')
        axs[(3*i) + 1].plot(t, restim_s1[:,0])

        axs[(3*i) + 2].set_title('repetition label')
        axs[(3*i) + 2].plot(t, rerep_s1[:,0])


    # subject 2 plots
    fig2, axs2 = plt.subplots(nrows=18, ncols=1, figsize=(20, 10))
    t_dim_2= emg_s2_e2.shape[0]
    # channels = np.array([0,2,4,6,10,11]) # sensor band, biceps brachi and triceps brachi sensors
    for i in range(0,channels.shape[0]):
        print('emg signal: ', emg_s2_e2[:,channels[i]])
        t2 = np.array([i for i in range(0,t_dim_2)])

        axs2[3*i].set_title('Subject 2: EMG signal exercise 2 (all movements), channel {}' .format(channels[i]))
        axs2[3*i].plot(t2,emg_s2_e2[:,channels[i]])

        axs2[(3*i) + 1].set_title('movement label')
        axs2[(3*i) + 1].plot(t2, restim_s2[:,0])

        axs2[(3*i) + 2].set_title('repetition label')
        axs2[(3*i) + 2].plot(t2, rerep_s2[:,0])

    #plt.show()


data_s1 = np.vstack(([emg_s1_e2[:, channels[i]] for i in range(0,channels.shape[0])],restim_s1[:,0], rerep_s1[:,0]))
data_s1 = data_s1.T
data_s2 = np.vstack(([emg_s2_e2[:, channels[i]] for i in range(0,channels.shape[0])],restim_s2[:,0], rerep_s2[:,0]))
data_s2 = data_s2.T
data_s3 = np.vstack(([emg_s3_e2[:, channels[i]] for i in range(0,channels.shape[0])],restim_s3[:,0], rerep_s3[:,0]))
data_s3 = data_s3.T
data_s5 = np.vstack(([emg_s5_e2[:, channels[i]] for i in range(0,channels.shape[0])],restim_s5[:,0], rerep_s5[:,0]))
data_s5 = data_s5.T
# data_s7 = np.vstack(([emg_s7_e2[:, channels[i]] for i in range(0,channels.shape[0])],restim_s7[:,0], rerep_s7[:,0]))
# data_s7 = data_s7.T

print("data_s1 shape: ",data_s1.shape)
print("data_s2 shape: ",data_s2.shape)
print("movement labels s1:", np.unique(restim_s1))
print("max movement label in S1:", np.max(restim_s1))
print("max movement label in S2:", np.max(restim_s2))

Subject_1 = pd.DataFrame(data_s1, columns= ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7','emg_8','emg_9','emg_10','emg_11','label', 'repetition'])
Subject_2 = pd.DataFrame(data_s2, columns= ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7','emg_8','emg_9','emg_10','emg_11','label', 'repetition'])
Subject_3 = pd.DataFrame(data_s3, columns= ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7','emg_8','emg_9','emg_10','emg_11','label', 'repetition'])
Subject_5 = pd.DataFrame(data_s5, columns= ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7','emg_8','emg_9','emg_10','emg_11','label', 'repetition'])
# Subject_7 = pd.DataFrame(data_s7, columns= ['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','label', 'repetition'])

print("Subject 1 dataframe", Subject_1)

# Subject 1 movements
# Finger abduction movement (restim= 22)
abduction_s1 = Subject_1[Subject_1.label == 22]
# Finger flexion movement (restim= 23)
flexion_s1 = Subject_1[Subject_1.label == 27]
# Wrist supination movement (restim= 26)
supination_s1 = Subject_1[Subject_1.label == 28]
# Wrist pronation movement (restim= 27)
pronation_s1 = Subject_1[Subject_1.label == 29]

# Subject 2 movements
# Finger abduction movement (restim= 22)
abduction_s2 = Subject_2[Subject_2.label == 22]
# Finger flexion movement (restim= 23)
flexion_s2 = Subject_2[Subject_2.label == 27]
# Wrist supination movement (restim= 26)
supination_s2 = Subject_2[Subject_2.label == 28]
# Wrist pronation movement (restim= 26)
pronation_s2 = Subject_2[Subject_2.label == 29]

# Subject 3 movements
# Finger abduction movement (restim= 22)
abduction_s3 = Subject_3[Subject_3.label == 22]
# Finger flexion movement (restim= 23)
flexion_s3 = Subject_3[Subject_3.label == 27]
# Wrist supination movement (restim= 26)
supination_s3 = Subject_3[Subject_3.label == 28]
# Wrist pronation movement (restim= 26)
pronation_s3 = Subject_3[Subject_3.label == 29]

# Subject 5 movements
# Finger abduction movement (restim= 22)
abduction_s5 = Subject_5[Subject_5.label == 22]
# Finger flexion movement (restim= 23)
flexion_s5 = Subject_5[Subject_5.label == 27]
# Wrist supination movement (restim= 26)
supination_s5 = Subject_5[Subject_5.label == 28]
# Wrist pronation movement (restim= 26)
pronation_s5 = Subject_5[Subject_5.label == 29]

# # Subject 7 movements
# # Finger abduction movement (restim= 22)
# abduction_s7 = Subject_7[Subject_7.label == 22]
# # Finger flexion movement (restim= 23)
# flexion_s7 = Subject_7[Subject_7.label == 23]
# # Wrist supination movement (restim= 26)
# supination_s7 = Subject_7[Subject_7.label == 26]
# # Wrist pronation movement (restim= 26)
# pronation_s7 = Subject_7[Subject_7.label == 27]



if plot_all | plot_segmented:

    # movement_s1 and movement_s2 represents the move is going to be plotted
    movement_s1 = abduction_s1
    movement_s2 = abduction_s2
    # pandas to numpy
    emg_s1_e2 = movement_s1[['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5']].to_numpy()
    # emg_s1_e2 = emg_s1_e2.T
    emg_s2_e2 = movement_s2[['emg_0','emg_1','emg_2','emg_3','emg_4','emg_5']].to_numpy()
    # emg_s2_e2 = emg_s2_e2.T

    restim_s1 = movement_s1[['label']].to_numpy()
    # restim_s1 = restim_s1.T

    restim_s2 = movement_s2[['label']].to_numpy()
    # restim_s2 = restim_s2.T

    rerep_s1 = movement_s1[['repetition']].to_numpy()
    # rerep_s1 = rerep_s1.T

    rerep_s2 = movement_s2[['repetition']].to_numpy()
    # rerep_s2 = rerep_s2.T

    #subject 1 movement plots
    fig3, axs3 = plt.subplots(nrows=18, ncols=1, figsize=(20, 10))
    t_dim3= emg_s1_e2.shape[0]
    # channels = np.array([0,2,4,6,10,11]) # sensor band, biceps brachi and triceps brachi sensors
    for i in range(0,emg_s1_e2.shape[1]):
        print('emg signal: ', emg_s1_e2[:,i])
        t3 = np.array([i for i in range(0,t_dim3)])

        axs3[3*i].set_title('Subject 1: EMG signal exercise 2 (ONE MOVEMENT), channel {}' .format(i))
        axs3[3*i].plot(t3,emg_s1_e2[:,i])

        axs3[(3*i) + 1].set_title('movement label')
        axs3[(3*i) + 1].plot(t3, restim_s1[:,0])

        axs3[(3*i) + 2].set_title('repetition label')
        axs3[(3*i) + 2].plot(t3, rerep_s1[:,0])


    # subject 2 plots
    fig4, axs4 = plt.subplots(nrows=18, ncols=1, figsize=(20, 10))
    t_dim4= emg_s2_e2.shape[0]
    # channels = np.array([0,2,4,6,10,11]) # sensor band, biceps brachi and triceps brachi sensors
    for i in range(0,emg_s2_e2.shape[1]):
        print('emg signal: ', emg_s2_e2[:,i])
        t4 = np.array([i for i in range(0,t_dim4)])

        axs4[3*i].set_title('Subject 2: EMG signal exercise 2 (ONE MOVEMENT), channel {}' .format(i))
        axs4[3*i].plot(t4,emg_s2_e2[:,i])

        axs4[(3*i) + 1].set_title('movement label')
        axs4[(3*i) + 1].plot(t4, restim_s2[:,0])

        axs4[(3*i) + 2].set_title('repetition label')
        axs4[(3*i) + 2].plot(t4, rerep_s2[:,0])

    # plt.show()

#Segmentation of movements
# f_s = 2 kHz -> T_s = 0.0005 s ; window time = 250 ms
f_s = 2000
T_s = 1/f_s
window_time = 200e-3
samples_per_window = window_time*f_s

# Subject 1 segmentation

# movement 1: abduction
abduction_s1_repetition_segmented = segment_movement(abduction_s1)

# movement 2: flexion
flexion_s1_repetition_segmented = segment_movement(flexion_s1)

# movement 3: supination
supination_s1_repetition_segmented = segment_movement(supination_s1)

# movement 4: pronation
pronation_s1_repetition_segmented = segment_movement(pronation_s1)

# Subject 2 segmentation

# movement 1: abduction
abduction_s2_repetition_segmented = segment_movement(abduction_s2)

# movement 2: flexion
flexion_s2_repetition_segmented = segment_movement(flexion_s2)

# movement 3: supination
supination_s2_repetition_segmented = segment_movement(supination_s2)

# movement 4: pronation
pronation_s2_repetition_segmented = segment_movement(pronation_s2)


# Subject 3 segmentation

# movement 1: abduction
abduction_s3_repetition_segmented = segment_movement(abduction_s3)

# movement 2: flexion
flexion_s3_repetition_segmented = segment_movement(flexion_s3)

# movement 3: supination
supination_s3_repetition_segmented = segment_movement(supination_s3)

# movement 4: pronation
pronation_s3_repetition_segmented = segment_movement(pronation_s3)


# Subject 5 segmentation

# movement 1: abduction
abduction_s5_repetition_segmented = segment_movement(abduction_s5)

# movement 2: flexion
flexion_s5_repetition_segmented = segment_movement(flexion_s5)

# movement 3: supination
supination_s5_repetition_segmented = segment_movement(supination_s5)

# movement 4: pronation
pronation_s5_repetition_segmented = segment_movement(pronation_s5)

# Subject 7 segmentation
#
# # movement 1: abduction
# abduction_s7_repetition_segmented = segment_movement(abduction_s7)
#
# # movement 2: flexion
# flexion_s7_repetition_segmented = segment_movement(flexion_s7)
#
# # movement 3: supination
# supination_s7_repetition_segmented = segment_movement(supination_s7)
#
# # movement 4: pronation
# pronation_s7_repetition_segmented = segment_movement(pronation_s7)

# Feature Extraction

# Subject 1
# movement 1
# get features
get_MAV(abduction_s1_repetition_segmented,channels)
get_WL(abduction_s1_repetition_segmented,channels)
get_ZC(abduction_s1_repetition_segmented,channels)
get_RMS(abduction_s1_repetition_segmented,channels)
get_SSC(abduction_s1_repetition_segmented,channels)
normalize_min_max_(abduction_s1_repetition_segmented,'MAV',channels)
normalize_min_max_(abduction_s1_repetition_segmented,'WL',channels)
normalize_min_max_(abduction_s1_repetition_segmented,'ZC',channels)
normalize_min_max_(abduction_s1_repetition_segmented,'RMS',channels)
normalize_min_max_(abduction_s1_repetition_segmented,'SSC',channels)

# movement 2
# get features
get_MAV(flexion_s1_repetition_segmented,channels)
get_WL(flexion_s1_repetition_segmented,channels)
get_ZC(flexion_s1_repetition_segmented,channels)
get_RMS(flexion_s1_repetition_segmented,channels)
get_SSC(flexion_s1_repetition_segmented,channels)
normalize_min_max_(flexion_s1_repetition_segmented,'MAV',channels)
normalize_min_max_(flexion_s1_repetition_segmented,'WL',channels)
normalize_min_max_(flexion_s1_repetition_segmented,'ZC',channels)
normalize_min_max_(flexion_s1_repetition_segmented,'RMS',channels)
normalize_min_max_(flexion_s1_repetition_segmented,'SSC',channels)

# movement 3
# get features
get_MAV(supination_s1_repetition_segmented,channels)
get_WL(supination_s1_repetition_segmented,channels)
get_ZC(supination_s1_repetition_segmented,channels)
get_RMS(supination_s1_repetition_segmented,channels)
get_SSC(supination_s1_repetition_segmented,channels)
normalize_min_max_(supination_s1_repetition_segmented,'MAV',channels)
normalize_min_max_(supination_s1_repetition_segmented,'WL',channels)
normalize_min_max_(supination_s1_repetition_segmented,'ZC',channels)
normalize_min_max_(supination_s1_repetition_segmented,'RMS',channels)
normalize_min_max_(supination_s1_repetition_segmented,'SSC',channels)


# movement 4
# get features
get_MAV(pronation_s1_repetition_segmented,channels)
get_WL(pronation_s1_repetition_segmented,channels)
get_ZC(pronation_s1_repetition_segmented,channels)
get_RMS(pronation_s1_repetition_segmented,channels)
get_SSC(pronation_s1_repetition_segmented,channels)
normalize_min_max_(pronation_s1_repetition_segmented,'MAV',channels)
normalize_min_max_(pronation_s1_repetition_segmented,'WL',channels)
normalize_min_max_(pronation_s1_repetition_segmented,'ZC',channels)
normalize_min_max_(pronation_s1_repetition_segmented,'RMS',channels)
normalize_min_max_(pronation_s1_repetition_segmented,'SSC',channels)

# Subject 2
# movement 1
# get Mean Absolute Value (MAV)
get_MAV(abduction_s2_repetition_segmented,channels)
get_WL(abduction_s2_repetition_segmented,channels)
get_ZC(abduction_s2_repetition_segmented,channels)
get_RMS(abduction_s2_repetition_segmented,channels)
get_SSC(abduction_s2_repetition_segmented,channels)
normalize_min_max_(abduction_s2_repetition_segmented,'MAV',channels)
normalize_min_max_(abduction_s2_repetition_segmented,'WL',channels)
normalize_min_max_(abduction_s2_repetition_segmented,'ZC',channels)
normalize_min_max_(abduction_s2_repetition_segmented,'RMS',channels)
normalize_min_max_(abduction_s2_repetition_segmented,'SSC',channels)

# movement 2
# get features
get_MAV(flexion_s2_repetition_segmented,channels)
get_WL(flexion_s2_repetition_segmented,channels)
get_ZC(flexion_s2_repetition_segmented,channels)
get_RMS(flexion_s2_repetition_segmented,channels)
get_SSC(flexion_s2_repetition_segmented,channels)
normalize_min_max_(flexion_s2_repetition_segmented,'MAV',channels)
normalize_min_max_(flexion_s2_repetition_segmented,'WL',channels)
normalize_min_max_(flexion_s2_repetition_segmented,'ZC',channels)
normalize_min_max_(flexion_s2_repetition_segmented,'RMS',channels)
normalize_min_max_(flexion_s2_repetition_segmented,'SSC',channels)

# movement 3
# get Mean Absolute Value (MAV)
get_MAV(supination_s2_repetition_segmented,channels)
get_WL(supination_s2_repetition_segmented,channels)
get_ZC(supination_s2_repetition_segmented,channels)
get_RMS(supination_s2_repetition_segmented,channels)
get_SSC(supination_s2_repetition_segmented,channels)
normalize_min_max_(supination_s2_repetition_segmented,'MAV',channels)
normalize_min_max_(supination_s2_repetition_segmented,'WL',channels)
normalize_min_max_(supination_s2_repetition_segmented,'ZC',channels)
normalize_min_max_(supination_s2_repetition_segmented,'RMS',channels)
normalize_min_max_(supination_s2_repetition_segmented,'SSC',channels)

# movement 4
# get Mean Absolute Value (MAV)
get_MAV(pronation_s2_repetition_segmented,channels)
get_WL(pronation_s2_repetition_segmented,channels)
get_ZC(pronation_s2_repetition_segmented,channels)
get_RMS(pronation_s2_repetition_segmented,channels)
get_SSC(pronation_s2_repetition_segmented,channels)
normalize_min_max_(pronation_s2_repetition_segmented,'MAV',channels)
normalize_min_max_(pronation_s2_repetition_segmented,'WL',channels)
normalize_min_max_(pronation_s2_repetition_segmented,'ZC',channels)
normalize_min_max_(pronation_s2_repetition_segmented,'RMS',channels)
normalize_min_max_(pronation_s2_repetition_segmented,'SSC',channels)

# Subject 3
# movement 1
# get Mean Absolute Value (MAV)
get_MAV(abduction_s3_repetition_segmented,channels)
get_WL(abduction_s3_repetition_segmented,channels)
get_ZC(abduction_s3_repetition_segmented,channels)
get_RMS(abduction_s3_repetition_segmented,channels)
get_SSC(abduction_s3_repetition_segmented,channels)
normalize_min_max_(abduction_s3_repetition_segmented,'MAV',channels)
normalize_min_max_(abduction_s3_repetition_segmented,'WL',channels)
normalize_min_max_(abduction_s3_repetition_segmented,'ZC',channels)
normalize_min_max_(abduction_s3_repetition_segmented,'RMS',channels)
normalize_min_max_(abduction_s3_repetition_segmented,'SSC',channels)

# movement 2
# get features
get_MAV(flexion_s3_repetition_segmented,channels)
get_WL(flexion_s3_repetition_segmented,channels)
get_ZC(flexion_s3_repetition_segmented,channels)
get_RMS(flexion_s3_repetition_segmented,channels)
get_SSC(flexion_s3_repetition_segmented,channels)
normalize_min_max_(flexion_s3_repetition_segmented,'MAV',channels)
normalize_min_max_(flexion_s3_repetition_segmented,'WL',channels)
normalize_min_max_(flexion_s3_repetition_segmented,'ZC',channels)
normalize_min_max_(flexion_s3_repetition_segmented,'RMS',channels)
normalize_min_max_(flexion_s3_repetition_segmented,'SSC',channels)

# movement 3
# get Mean Absolute Value (MAV)
get_MAV(supination_s3_repetition_segmented,channels)
get_WL(supination_s3_repetition_segmented,channels)
get_ZC(supination_s3_repetition_segmented,channels)
get_RMS(supination_s3_repetition_segmented,channels)
get_SSC(supination_s3_repetition_segmented,channels)
normalize_min_max_(supination_s3_repetition_segmented,'MAV',channels)
normalize_min_max_(supination_s3_repetition_segmented,'WL',channels)
normalize_min_max_(supination_s3_repetition_segmented,'ZC',channels)
normalize_min_max_(supination_s3_repetition_segmented,'RMS',channels)
normalize_min_max_(supination_s3_repetition_segmented,'SSC',channels)

# movement 4
# get Mean Absolute Value (MAV)
get_MAV(pronation_s3_repetition_segmented,channels)
get_WL(pronation_s3_repetition_segmented,channels)
get_ZC(pronation_s3_repetition_segmented,channels)
get_RMS(pronation_s3_repetition_segmented,channels)
get_SSC(pronation_s3_repetition_segmented,channels)
normalize_min_max_(pronation_s3_repetition_segmented,'MAV',channels)
normalize_min_max_(pronation_s3_repetition_segmented,'WL',channels)
normalize_min_max_(pronation_s3_repetition_segmented,'ZC',channels)
normalize_min_max_(pronation_s3_repetition_segmented,'RMS',channels)
normalize_min_max_(pronation_s3_repetition_segmented,'SSC',channels)

# Subject 5
# movement 1
# get Mean Absolute Value (MAV)
get_MAV(abduction_s5_repetition_segmented,channels)
get_WL(abduction_s5_repetition_segmented,channels)
get_ZC(abduction_s5_repetition_segmented,channels)
get_RMS(abduction_s5_repetition_segmented,channels)
get_SSC(abduction_s5_repetition_segmented,channels)
normalize_min_max_(abduction_s5_repetition_segmented,'MAV',channels)
normalize_min_max_(abduction_s5_repetition_segmented,'WL',channels)
normalize_min_max_(abduction_s5_repetition_segmented,'ZC',channels)
normalize_min_max_(abduction_s5_repetition_segmented,'RMS',channels)
normalize_min_max_(abduction_s5_repetition_segmented,'SSC',channels)

# movement 2
# get features
get_MAV(flexion_s5_repetition_segmented,channels)
get_WL(flexion_s5_repetition_segmented,channels)
get_ZC(flexion_s5_repetition_segmented,channels)
get_RMS(flexion_s5_repetition_segmented,channels)
get_SSC(flexion_s5_repetition_segmented,channels)
normalize_min_max_(flexion_s5_repetition_segmented,'MAV',channels)
normalize_min_max_(flexion_s5_repetition_segmented,'WL',channels)
normalize_min_max_(flexion_s5_repetition_segmented,'ZC',channels)
normalize_min_max_(flexion_s5_repetition_segmented,'RMS',channels)
normalize_min_max_(flexion_s5_repetition_segmented,'SSC',channels)

# movement 3
# get Mean Absolute Value (MAV)
get_MAV(supination_s5_repetition_segmented,channels)
get_WL(supination_s5_repetition_segmented,channels)
get_ZC(supination_s5_repetition_segmented,channels)
get_RMS(supination_s5_repetition_segmented,channels)
get_SSC(supination_s5_repetition_segmented,channels)
normalize_min_max_(supination_s5_repetition_segmented,'MAV',channels)
normalize_min_max_(supination_s5_repetition_segmented,'WL',channels)
normalize_min_max_(supination_s5_repetition_segmented,'ZC',channels)
normalize_min_max_(supination_s5_repetition_segmented,'RMS',channels)
normalize_min_max_(supination_s5_repetition_segmented,'SSC',channels)

# movement 4
# get Mean Absolute Value (MAV)
get_MAV(pronation_s5_repetition_segmented,channels)
get_WL(pronation_s5_repetition_segmented,channels)
get_ZC(pronation_s5_repetition_segmented,channels)
get_RMS(pronation_s5_repetition_segmented,channels)
get_SSC(pronation_s5_repetition_segmented,channels)
normalize_min_max_(pronation_s5_repetition_segmented,'MAV',channels)
normalize_min_max_(pronation_s5_repetition_segmented,'WL',channels)
normalize_min_max_(pronation_s5_repetition_segmented,'ZC',channels)
normalize_min_max_(pronation_s5_repetition_segmented,'RMS',channels)
normalize_min_max_(pronation_s5_repetition_segmented,'SSC',channels)

# # Subject 7
# # movement 1
# # get Mean Absolute Value (MAV)
# get_MAV(abduction_s7_repetition_segmented,channels)
# get_WL(abduction_s7_repetition_segmented,channels)
# get_ZC(abduction_s7_repetition_segmented,channels)
# get_RMS(abduction_s7_repetition_segmented,channels)
# get_SSC(abduction_s7_repetition_segmented,channels)
# normalize_min_max_(abduction_s7_repetition_segmented,'MAV',channels)
# normalize_min_max_(abduction_s7_repetition_segmented,'WL',channels)
# normalize_min_max_(abduction_s7_repetition_segmented,'ZC',channels)
# normalize_min_max_(abduction_s7_repetition_segmented,'RMS',channels)
# normalize_min_max_(abduction_s7_repetition_segmented,'SSC',channels)
#
# # movement 2
# # get features
# get_MAV(flexion_s7_repetition_segmented,channels)
# get_WL(flexion_s7_repetition_segmented,channels)
# get_ZC(flexion_s7_repetition_segmented,channels)
# get_RMS(flexion_s7_repetition_segmented,channels)
# get_SSC(flexion_s7_repetition_segmented,channels)
# normalize_min_max_(flexion_s7_repetition_segmented,'MAV',channels)
# normalize_min_max_(flexion_s7_repetition_segmented,'WL',channels)
# normalize_min_max_(flexion_s7_repetition_segmented,'ZC',channels)
# normalize_min_max_(flexion_s7_repetition_segmented,'RMS',channels)
# normalize_min_max_(flexion_s7_repetition_segmented,'SSC',channels)
#
# # movement 3
# # get Mean Absolute Value (MAV)
# get_MAV(supination_s7_repetition_segmented,channels)
# get_WL(supination_s7_repetition_segmented,channels)
# get_ZC(supination_s7_repetition_segmented,channels)
# get_RMS(supination_s7_repetition_segmented,channels)
# get_SSC(supination_s7_repetition_segmented,channels)
# normalize_min_max_(supination_s7_repetition_segmented,'MAV',channels)
# normalize_min_max_(supination_s7_repetition_segmented,'WL',channels)
# normalize_min_max_(supination_s7_repetition_segmented,'ZC',channels)
# normalize_min_max_(supination_s7_repetition_segmented,'RMS',channels)
# normalize_min_max_(supination_s7_repetition_segmented,'SSC',channels)
#
# # movement 4
# # get Mean Absolute Value (MAV)
# get_MAV(pronation_s7_repetition_segmented,channels)
# get_WL(pronation_s7_repetition_segmented,channels)
# get_ZC(pronation_s7_repetition_segmented,channels)
# get_RMS(pronation_s7_repetition_segmented,channels)
# get_SSC(pronation_s7_repetition_segmented,channels)
# normalize_min_max_(pronation_s7_repetition_segmented,'MAV',channels)
# normalize_min_max_(pronation_s7_repetition_segmented,'WL',channels)
# normalize_min_max_(pronation_s7_repetition_segmented,'ZC',channels)
# normalize_min_max_(pronation_s7_repetition_segmented,'RMS',channels)
# normalize_min_max_(pronation_s7_repetition_segmented,'SSC',channels)
#


if plot_all | plot_seg_and_features:
    # Movement in Subject 1: segmented first repetition
    index = 1
    electrode = 0
    emg_rep1 = abduction_s1_repetition_segmented[index][['emg_{}'.format(electrode)]].to_numpy()
    mav_rep1 = abduction_s1_repetition_segmented[index]['MAV_emg_{}'.format(electrode)].to_numpy()
    wl_rep1 = abduction_s1_repetition_segmented[index]['WL_emg_{}'.format(electrode)].to_numpy()
    zc_rep1 = abduction_s1_repetition_segmented[index]['ZC_emg_{}'.format(electrode)].to_numpy()
    rms_rep1 = abduction_s1_repetition_segmented[index]['RMS_emg_{}'.format(electrode)].to_numpy()
    ssc_rep1 = abduction_s1_repetition_segmented[index]['SSC_emg_{}'.format(electrode)].to_numpy()
    normal_mav_rep1 = abduction_s1_repetition_segmented[index]['MAV_normal_emg_{}'.format(electrode)].to_numpy()
    normal_wl_rep1 = abduction_s1_repetition_segmented[index]['WL_normal_emg_{}'.format(electrode)].to_numpy()
    normal_zc_rep1 = abduction_s1_repetition_segmented[index]['ZC_normal_emg_{}'.format(electrode)].to_numpy()
    normal_rms_rep1 = abduction_s1_repetition_segmented[index]['RMS_normal_emg_{}'.format(electrode)].to_numpy()
    normal_ssc_rep1 = abduction_s1_repetition_segmented[index]['SSC_normal_emg_{}'.format(electrode)].to_numpy()
    rep_label_rep1 = abduction_s1_repetition_segmented[index][['repetition']].to_numpy()
    segment_rep1 = abduction_s1_repetition_segmented[index][['segment']].to_numpy()


    fig5, axs5 = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
    t_dim5 = emg_rep1.shape[0]
    t5 = np.array([i for i in range(0, t_dim5)])

    axs5[0].set_title('Subject 1: EMG signal of a movement, channel {}'.format(index))
    axs5[0].plot(t5, emg_rep1)

    axs5[1].set_title('Normal features ')
    axs5[1].plot(t5, normal_mav_rep1, label='MAV')
    axs5[1].plot(t5, normal_wl_rep1,label='WL')
    axs5[1].plot(t5, normal_zc_rep1, label='ZC')
    axs5[1].plot(t5, normal_rms_rep1, label='RMS')
    axs5[1].plot(t5, normal_ssc_rep1, label='SSC')
    axs5[1].legend()


    axs5[2].set_title('repetition label')
    axs5[2].plot(t5, rep_label_rep1)

    axs5[3].set_title('segmentation label')
    axs5[3].plot(t5, segment_rep1)

# get correlation

#Subject 1
features_per_channel = 5
abduction_s1_cor, abduction_s1_complete = get_correlation(abduction_s1_repetition_segmented,channels,features_per_channel)
flexion_s1_cor, flexion_s1_complete = get_correlation(flexion_s1_repetition_segmented,channels,features_per_channel)
supination_s1_cor, supination_s1_complete = get_correlation(supination_s1_repetition_segmented,channels,features_per_channel)
pronation_s1_cor, pronation_s1_complete = get_correlation(pronation_s1_repetition_segmented,channels,features_per_channel)

#Subject 2
abduction_s2_cor, abduction_s2_complete = get_correlation(abduction_s2_repetition_segmented,channels,features_per_channel)
flexion_s2_cor, flexion_s2_complete = get_correlation(flexion_s2_repetition_segmented,channels,features_per_channel)
supination_s2_cor, supination_s2_complete = get_correlation(supination_s2_repetition_segmented,channels,features_per_channel)
pronation_s2_cor, pronation_s2_complete = get_correlation(pronation_s2_repetition_segmented,channels,features_per_channel)

#Subject 3
abduction_s3_cor, abduction_s3_complete = get_correlation(abduction_s3_repetition_segmented,channels,features_per_channel)
flexion_s3_cor, flexion_s3_complete = get_correlation(flexion_s3_repetition_segmented,channels,features_per_channel)
supination_s3_cor, supination_s3_complete = get_correlation(supination_s3_repetition_segmented,channels,features_per_channel)
pronation_s3_cor, pronation_s3_complete = get_correlation(pronation_s3_repetition_segmented,channels,features_per_channel)

#Subject 5
abduction_s5_cor, abduction_s5_complete = get_correlation(abduction_s5_repetition_segmented,channels,features_per_channel)
flexion_s5_cor, flexion_s5_complete = get_correlation(flexion_s5_repetition_segmented,channels,features_per_channel)
supination_s5_cor, supination_s5_complete = get_correlation(supination_s5_repetition_segmented,channels,features_per_channel)
pronation_s5_cor, pronation_s5_complete = get_correlation(pronation_s5_repetition_segmented,channels,features_per_channel)

# #Subject 7
# abduction_s7_cor, abduction_s7_complete = get_correlation(abduction_s7_repetition_segmented,channels,features_per_channel)
# flexion_s7_cor, flexion_s7_complete = get_correlation(flexion_s7_repetition_segmented,channels,features_per_channel)
# supination_s7_cor, supination_s7_complete = get_correlation(supination_s7_repetition_segmented,channels,features_per_channel)
# pronation_s7_cor, pronation_s7_complete = get_correlation(pronation_s7_repetition_segmented,channels,features_per_channel)


#Concatenate all features
all_features = pd.concat([abduction_s1_complete,flexion_s1_complete,supination_s1_complete,pronation_s1_complete,
                          abduction_s2_complete,flexion_s2_complete,supination_s2_complete,pronation_s2_complete,
                          abduction_s3_complete,flexion_s3_complete,supination_s3_complete,pronation_s3_complete,
                          abduction_s5_complete,flexion_s5_complete,supination_s5_complete,pronation_s5_complete])

# get correlation
total_cor = all_features.iloc[:, [i for i in range(-2 * features_per_channel * channels.shape[0],
                                                  -features_per_channel * channels.shape[0])]].corr()

#get absolute correlation
cor_target = abs(total_cor)

#get relevant features
relevant_features = cor_target[cor_target<0.2]
print ("matrix with relevant features (p<0.2) ", relevant_features)

mean_relevant_features = relevant_features.mean()
print ("vector with mean p in relevant features", mean_relevant_features)


sort_features = mean_relevant_features.sort_values()
print("sorted relevant features", sort_features)

count_condition_in_feature= relevant_features.count()
print("count", count_condition_in_feature)

weight_mean = (relevant_features.count()/(features_per_channel * channels.shape[0]))/(relevant_features.mean())
weight_mean = weight_mean.sort_values()

print("weight mean p:", weight_mean)

significant_keys= (weight_mean[-int(np.ceil((features_per_channel * channels.shape[0])/2)):]).keys()
print("significant features:", significant_keys)


# Pearson

#abduction vs flexion
abduction_complete = pd.concat([abduction_s1_complete,abduction_s2_complete,
                                         abduction_s3_complete,abduction_s5_complete])
label_1 = np.array([1 for i in range(0,len(abduction_complete))])

flexion_complete = pd.concat([flexion_s1_complete,flexion_s2_complete,
                              flexion_s3_complete,flexion_s5_complete])
label_0 = np.array([0 for i in range(0,len(flexion_complete))])
label = np.concatenate((label_1,label_0),axis=None)


abduction_vs_flexion = pd.concat([abduction_complete,flexion_complete])
# abduction_vs_flexion.insert(abduction_vs_flexion.shape[1] ,"label", label)


# Not normalized
abduction_vs_flexion_feat = abduction_vs_flexion.iloc[:, [i for i in range(-2 * features_per_channel * channels.shape[0],
                                                  -features_per_channel * channels.shape[0])]]
#abduction_vs_flexion_feat = abduction_vs_flexion.iloc[:, [i for i in range(- features_per_channel * channels.shape[0],-1)]]
# Normalized
abduction_vs_flexion_feat["label"] = abduction_vs_flexion["label"]

abduction_vs_flexion_corr = abduction_vs_flexion_feat.corr()
abduction_vs_flexion_corr_class = abs(abduction_vs_flexion_corr["label"])
ax_abd = plt.axes()
sns.heatmap(abduction_vs_flexion_corr, annot=True, cmap=plt.cm.Reds, ax=ax_abd)
ax_abd.set_title("Correlation Matrix (Pearson) for abduction vs flexion")
plt.show()
abduction_vs_flexion_relevant_features = abduction_vs_flexion_corr_class[abduction_vs_flexion_corr_class>0.15]
print("abduction_vs_flexion_relevant_features",abduction_vs_flexion_relevant_features)
sort_abd_vs_flexion = abduction_vs_flexion_relevant_features.sort_values()
abd_vs_flexion_keys = sort_abd_vs_flexion[-int(np.ceil(0.5*len(sort_abd_vs_flexion))):-1].keys()
abduction_vs_flexion_relevant_features_corr = abduction_vs_flexion_feat[abd_vs_flexion_keys].corr()
mean_abd_vs_flexion_relev_features_corr = abduction_vs_flexion_relevant_features_corr.mean()
mean_abd_vs_flexion_relev_features_corr = mean_abd_vs_flexion_relev_features_corr.sort_values()
print("ordered mean correlation:", mean_abd_vs_flexion_relev_features_corr)
significant_keys_abd_flex = mean_abd_vs_flexion_relev_features_corr[0: int(np.ceil(0.4*len(mean_abd_vs_flexion_relev_features_corr)))].keys()
# significant_keys_abd_flex = mean_abd_vs_flexion_relev_features_corr.keys()
print("significant features abduction vs flexion", significant_keys_abd_flex)
ax_abd2 = plt.axes()
sns.heatmap(abduction_vs_flexion_relevant_features_corr, annot=True, cmap=plt.cm.Reds, ax=ax_abd2)
ax_abd2.set_title("Correlation Matrix (Pearson) for abduction vs flexion relevant features")
plt.show()
#PCA abduction vs flexion (using selected features)
pca = PCA(n_components=2)
pcafit = pca.fit(abduction_vs_flexion[significant_keys_abd_flex])
reduced_abduction = pcafit.transform(abduction_complete[significant_keys_abd_flex])
reduced_flexion = pcafit.transform(flexion_complete[significant_keys_abd_flex])
# PCA abduction vs flexion (using all features)
pca2 = PCA(n_components=2)
all_features_keys = abduction_vs_flexion_feat.iloc[:,[i for i in range(0, abduction_vs_flexion_feat.shape[1]-1)]].keys()
pcafit2 = pca2.fit(abduction_vs_flexion_feat[all_features_keys])
reduced_abduction2 = pcafit2.transform(abduction_complete[all_features_keys])
reduced_flexion2 = pcafit2.transform(flexion_complete[all_features_keys])
print("reduced abduction", reduced_abduction)
print("reduced flexion", reduced_flexion)

if plot_complete_movement_features | plot_all:
    electrode = 0
    mov = flexion_s2_complete


    fig6, axs6 = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
    emg = mov['emg_{}'.format(electrode)].to_numpy()
    mav = mov['MAV_normal_emg_{}'.format(electrode)].to_numpy()
    wl = mov['WL_normal_emg_{}'.format(electrode)].to_numpy()
    rms = mov['RMS_normal_emg_{}'.format(electrode)].to_numpy()
    zc = mov['ZC_normal_emg_{}'.format(electrode)].to_numpy()
    ssc = mov['SSC_normal_emg_{}'.format(electrode)].to_numpy()
    segment = mov['segment'].to_numpy()
    t_dim6 = emg.shape[0]
    t6 = np.array([i for i in range(0, t_dim6)])
    axs6[0].plot(t6, emg)
    axs6[1].plot(t6, mav)
    axs6[2].plot(t6, segment)
if plot_total_correlation | plot_all:
    cor,mov = (flexion_s2_cor, flexion_s2_complete)
    ax_cor = plt.axes()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, ax=ax_cor)
    ax_cor.set_title("Correlation Matrix (Pearson) for 1 movement")
    plt.show()
    ax_total_cor = plt.axes()

    sns.heatmap(total_cor, annot=True, cmap=plt.cm.Reds, ax=ax_total_cor)
    ax_total_cor.set_title("Correlation Matrix (Pearson) for all training dataset")
    plt.show()
    ax_filter_cor = plt.axes()

    sns.heatmap(relevant_features, annot=True, cmap=plt.cm.Reds, ax=ax_filter_cor)
    ax_filter_cor.set_title("Correlation Matrix (Pearson) for all training dataset (filtered)")
    plt.show()
    # ax_mean_filter = plt.axes()
    # sns.heatmap(mean_relevant_features, annot=True, cmap=plt.cm.Reds, ax=ax_mean_filter)
    # ax_filter_cor.set_title("Correlation Matrix (Pearson) for all training dataset (filter-mean)")

fig = plt.figure()
ax_feat = fig.add_subplot(111, projection='3d')
idx1,idx2,idx3 = (0,1,2)
ax_feat.scatter(abduction_complete['ZC_normal_emg_11'], abduction_complete['RMS_normal_emg_4'], abduction_complete['WL_normal_emg_4'], c='orange')
ax_feat.scatter(flexion_complete['ZC_normal_emg_11'], flexion_complete['RMS_normal_emg_4'], flexion_complete['WL_normal_emg_4'], c='green')

#ax_feat.scatter(abduction_s3_complete[significant_keys_abd_flex[idx1]], abduction_s3_complete[significant_keys_abd_flex[idx2]], abduction_s3_complete[significant_keys[idx3]], c='orange')
#ax_feat.scatter(flexion_s3_complete[significant_keys_abd_flex[idx1]], flexion_s3_complete[significant_keys_abd_flex[idx2]], flexion_s3_complete[significant_keys[idx3]], c='green')
# ax_feat.scatter(supination_s5_complete[significant_keys[idx1]], supination_s5_complete[significant_keys[idx2]], supination_s5_complete[significant_keys[idx3]], c='pink')
# ax_feat.scatter(pronation_s5_complete[significant_keys[idx1]], pronation_s5_complete[significant_keys[idx2]], pronation_s5_complete[significant_keys[idx3]], c='blue')
ax_feat.set_xlabel('ZC_normal_emg_11')
ax_feat.set_ylabel('RMS_normal_emg_4')
ax_feat.set_zlabel('WL_normal_emg_4')
plt.show()

fig_pca = plt.figure()
ax_pca = fig_pca.add_subplot()
ax_pca.scatter(reduced_abduction[:,0],reduced_abduction[:,1], c='red')
ax_pca.scatter(reduced_flexion[:,0],reduced_flexion[:,1], c='blue')
ax_pca.set_xlabel('PC 1')
ax_pca.set_ylabel('PC 2')



plt.show()

# show plots (end of algorithm)
if plot_all | plot_raw | plot_segmented | plot_seg_and_features | plot_complete_movement_features | plot_total_correlation:
    plt.show()















