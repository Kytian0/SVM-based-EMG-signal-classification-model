# libraries
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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

# flags
#method
filter_method = False
PCA_all_feat = False
RFE_method = True

#plot
plot = True
plot_segmented = False
plot_features = False
plot_corr_class = False
plot_complete_movement = False
plot_PCA = False
plot_PCA_all_feat = False
plot_PCA_RFE = True
plot_SVM_RFE = True

# functions
def get_keys(dict):
    list = []
    for k in dict.keys():
        list.append(k)
    return list

def segment_movement(movement, samples_per_window):
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


def RMS_function(data_segment):
    x_i = data_segment.to_numpy()
    RMS = np.sqrt((1/x_i.shape[0])*np.dot(x_i,x_i))
    return RMS


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


def WL_function(data_segment):
    x_i = data_segment.to_numpy()
    x_i_p1 = np.concatenate((x_i[1:],0), axis=None)
    abs_sub = np.abs(x_i_p1 - x_i)
    abs_sub = abs_sub[0:-1]
    WL_feature = sum(abs_sub)
    return WL_feature


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

def get_feature(segmented_movement, channels_list, feature):

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

            #next segment
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
                    feature_obtained[segment_idx,i] = ZC_function(data_segment['emg_{}'.format(i)])
                elif feature == 'WL':
                    feature_obtained[segment_idx, i] = WL_function(data_segment['emg_{}'.format(i)])
                elif feature == 'RMS':
                    feature_obtained[segment_idx, i] = RMS_function(data_segment['emg_{}'.format(i)])
                elif feature == 'SSC':
                    feature_obtained[segment_idx, i] = SSC_function(data_segment['emg_{}'.format(i)])
                elif feature == 'MAVS':
                    if i< number_of_segments - 1:
                        feature_obtained[segment_idx, i] = np.mean(np.abs(next_data_segment['emg_{}'.format(i)])) \
                                                           - np.mean(np.abs(data_segment['emg_{}'.format(i)]))
                    else:
                        feature_obtained[segment_idx, i] = - np.mean(
                            np.abs(data_segment['emg_{}'.format(i)]))
                # add obtained feature to dataframe
                segmented_movement[rep_idx].loc[segment_filter, feature +'_emg_{}'.format(i)] = feature_obtained[
                    segment_idx, i]
def concatenate_movement_repetition(movement_to_conc):
    # concatenate repetitions
    complete_movement = pd.concat(movement_to_conc)
    # remove NaN (segment size < window)
    complete_movement.dropna(subset=["MAV_emg_0"], inplace=True)

    return complete_movement

def fitting(X, y, C, gamma):
    # Create training and testing samples
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,train_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.004, train_size=0.007,
                                                        random_state=0)
    print("shape of X_train", X_train.shape)
    print("shape of X_test", X_test.shape)
    print("shape of y_train", y_train.shape)
    print("shape of y_test", y_test.shape)
    print("X_train", X_train, "y_train", y_train)
    # Fit the model
    # Note, available kernels: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    model = SVC(kernel='rbf', probability=True, C=C, gamma=gamma)
    clf = model.fit(X_train, y_train)
    print("values of clf", clf.predict_proba(X))

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    # Use score method to get accuracy of the model
    print('----- Evaluation on Test Data -----')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')

    print('----- Evaluation on Training Data -----')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf


def Plot_3D(X, X_test, y_test, clf):
    # Specify a size of the mesh to be used
    #mesh_size = 1e-5 # for small valued data
    mesh_size = 0.01 # for normal data X E [0,1]
    margin = 0

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
    print("xx shape", xx.shape)
    print("xx, yy", (xx,yy))
    # Calculate predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    print("Values of Z", Z)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(x=X_test['PCA_1'], y=X_test['PCA_2'], z=y_test,
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


tic = time.time()
# load data
#S1: 50% of forearm
s1_e2_a1 = loadmat('S1_E2_A1.mat')
#S2: 70% of forearm
s2_e2_a1 = loadmat('S2_E2_A1.mat')
#S3: 30% of forearm
s3_e2_a1 = loadmat('S3_E2_A1.mat')
#S5: 90% of forearm
s5_e2_a1 = loadmat('S5_E2_A1.mat')

raw_data = (s1_e2_a1,s2_e2_a1,s3_e2_a1,s5_e2_a1)
#MATLAB fields in databases
raw_keys = {}
for i in range(0, len(raw_data)):
    raw_keys[i] = get_keys(raw_data[i])
    # print("raw keys", raw_keys[i])
keys ={}
for i in range(0,len(raw_data)):
    keys[i] = raw_keys[i][3: len(raw_keys[i])]
    # print("keys", keys[i])


# get subject data

# emg data
emg_signals = {}
for i in range(0, len(raw_data)):
    emg_signals[i] = np.array(raw_data[i][keys[i][0]])
    print("emg_signals", emg_signals[i])

# restim data (labels)
restim = {}
for i in range(0, len(raw_data)):
    restim[i] = np.array(raw_data[i][keys[i][8]])
    print("restim", restim[i])

# rerepetition (repetitions)
rerep = {}
for i in range(0, len(raw_data)):
    rerep[i] = np.array(raw_data[i][keys[i][9]])
    print("rerep", rerep[i])

# Electrode channels
# channel 0 - channel 7 -> forearm electrodes
# channel 8 -> Flexor digitorum
# channel 9 -> Extensor digitorum
# channel 10 -> Biceps Brachii
# channel 11 -> Triceps Brachii

channels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

# order data
data = {}

for j in range(0,len(raw_data)):
    data[j] = np.vstack(([emg_signals[j][:, channels[i]] for i in range(0,channels.shape[0])],restim[j][:,0], rerep[j][:,0]))
    data[j] = data[j].T
# print("data",data)

# create DataFrames
Subject = {}
column_labels = ['emg_{}'.format(i) for i in channels]
column_labels.append('label')
column_labels.append('repetition')
for i in range(0,len(raw_data)):
    Subject[i] = pd.DataFrame(data[i], columns=column_labels)
print("Subject 0", Subject[0])

# movements to classify

movements_to_classify = (22, 27, 28, 29)

movements_data = {}

for i in range(0, len(movements_to_classify)):
    for j in range(0,len(raw_data)):
        movements_data[i,j] = Subject[j][Subject[j].label == movements_to_classify[i]]
print("movement 0 in subject 0", movements_data[0,0])

# segment movements data

# f_s = 2 kHz -> T_s = 0.0005 s ; window time = 200 ms
f_s = 2000
T_s = 1/f_s
window_time = 200e-3
samples_per_window = window_time*f_s

movements_data_segmented = {}

for i in range(0, len(movements_to_classify)):
    for j in range(0,len(raw_data)):
        movements_data_segmented[i,j] = segment_movement(movements_data[i,j],samples_per_window)



# get features

features_list = ['MAV','WL', 'ZC', 'RMS', 'SSC', 'MAVS']
for i in range(0, len(movements_to_classify)):
    for j in range(0,len(raw_data)):
        for k in range(0, len(features_list)):
            get_feature(movements_data_segmented[i,j],channels, features_list[k])

for i in range(0, len(movements_to_classify)):
    for j in range(0,len(raw_data)):
        for k in range(0, len(features_list)):
            normalize_min_max_(movements_data_segmented[i,j],features_list[k],channels)

# select features
features_per_channel = len(features_list)

# select features for movement 1 vs movement 2

# movement pair represents indexes in "movements_to_classify" array
movement_pair = [0,1]
#ToDo: revisar la funcion get_correlation() del código previo para concatenar las repeticiones en un solo diccionario
movement_1 = {}
movement_2 = {}
for i in range(0,len(raw_data)):
    movement_1[i] = concatenate_movement_repetition(movements_data_segmented[movement_pair[0],i])
    movement_2[i] = concatenate_movement_repetition(movements_data_segmented[movement_pair[1],i])

for i in range(0,len(raw_data)):
    if i == 0:
        movement_1_complete = movement_1[0]
        movement_2_complete = movement_2[0]
        print("movement 1 complete, i = 0", movement_1_complete)
    else:
        movement_1_complete = pd.concat([movement_1_complete, movement_1[i]])
        movement_2_complete = pd.concat([movement_2_complete, movement_2[i]])

movement_1_vs_movement_2 = pd.concat([movement_1_complete,movement_2_complete])
movement_1_vs_movement_2_feat = movement_1_vs_movement_2.iloc[:, [i for i in range(-2 * features_per_channel * channels.shape[0],
                                                  -features_per_channel * channels.shape[0])]]
movement_1_vs_movement_2_feat['label'] = movement_1_vs_movement_2['label']


if filter_method:
    # Find correlated features with class
    movement_1_vs_movement_2_corr = movement_1_vs_movement_2_feat.corr()
    # movement_1_vs_movement_2_corr_class = abs(movement_1_vs_movement_2_corr['label'])
    movement_1_vs_movement_2_corr_class = abs(movement_1_vs_movement_2_corr['ZC_emg_3'])

    threshold_p = 0.75
    movement_1_vs_movement_2_relevant_features = movement_1_vs_movement_2_corr_class[movement_1_vs_movement_2_corr_class>threshold_p]
    print("correlation features vs class: ", movement_1_vs_movement_2_relevant_features)
    # sort features
    sort_movement_1_vs_movement_2 = movement_1_vs_movement_2_relevant_features.sort_values()

    # first feature reduction
    reduction_factor_1 = 0.7
    relevant_features_keys = sort_movement_1_vs_movement_2[-int(np.ceil(reduction_factor_1*len(sort_movement_1_vs_movement_2))):-1].keys()

    # Find uncorrelated features

    # correlation matrix between features
    movement_1_vs_movement_2_relevant_features_corr = movement_1_vs_movement_2_feat[relevant_features_keys].corr()
    # mean correlation per feature
    mean_mov1_vs_mov2_relevant_features_corr = movement_1_vs_movement_2_relevant_features_corr.mean()
    # sort mean correlations
    mean_mov1_vs_mov2_relevant_features_corr = mean_mov1_vs_mov2_relevant_features_corr.sort_values()

    #second feature reduction
    reduction_factor_2 = 0.5
    uncorrelated_features_keys = mean_mov1_vs_mov2_relevant_features_corr[0: int(np.ceil(reduction_factor_2*len(mean_mov1_vs_mov2_relevant_features_corr)))].keys()
    print("uncorrelated features movement 1 vs movement 2", uncorrelated_features_keys)


    # reduce features (PCA)
    pca = PCA(n_components=2)
    pcafit = pca.fit(movement_1_vs_movement_2[uncorrelated_features_keys])
    reduced_movement_1 = pcafit.transform(movement_1_complete[uncorrelated_features_keys])
    reduced_movement_2 = pcafit.transform(movement_2_complete[uncorrelated_features_keys])

# use PCA with all features


all_features_list = ['MAV_emg_0','MAV_emg_1','MAV_emg_2','MAV_emg_3','MAV_emg_4','MAV_emg_5','MAV_emg_6','MAV_emg_7','MAV_emg_8','MAV_emg_9','MAV_emg_10','MAV_emg_11',
                 'WL_emg_0','WL_emg_1','WL_emg_2','WL_emg_3','WL_emg_4','WL_emg_5','WL_emg_6','WL_emg_7','WL_emg_8','WL_emg_9','WL_emg_10','WL_emg_11',
                 'ZC_emg_0','ZC_emg_1','ZC_emg_2','ZC_emg_3','ZC_emg_4','ZC_emg_5','ZC_emg_6','ZC_emg_7','ZC_emg_8','ZC_emg_9','ZC_emg_10','ZC_emg_11',
                 'RMS_emg_0','RMS_emg_1','RMS_emg_2','RMS_emg_3','RMS_emg_4','RMS_emg_5','RMS_emg_6','RMS_emg_7','RMS_emg_8','RMS_emg_9','RMS_emg_10','RMS_emg_11',
                 'SSC_emg_0','SSC_emg_1','SSC_emg_2','SSC_emg_3','SSC_emg_4','SSC_emg_5','SSC_emg_6','SSC_emg_7','SSC_emg_8','SSC_emg_9','SSC_emg_10','SSC_emg_11',
                 'MAVS_emg_0', 'MAVS_emg_1','MAVS_emg_2','MAVS_emg_3','MAVS_emg_4','MAVS_emg_5','MAVS_emg_6','MAVS_emg_7','MAVS_emg_8','MAVS_emg_9','MAVS_emg_10','MAVS_emg_11']
if PCA_all_feat:
    pca2 = PCA(n_components=2)
    pcafit_2 = pca2.fit(movement_1_vs_movement_2[all_features_list])
    reduced_movement_1_all = pcafit_2.transform(movement_1_complete[all_features_list])
    reduced_movement_2_all = pcafit_2.transform(movement_2_complete[all_features_list])
# Conclusion: se muestra menos separable utilizando todas las características crudas


# RFE method
if RFE_method:
    df = movement_1_vs_movement_2_feat
    X = df.drop('label',1)
    y = df['label']

    cols = list(X.columns)
    model = LinearRegression()
    #Initializing RFE model
    rfe = RFE(model, n_features_to_select= 6)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)
    #Fitting the data to model
    model.fit(X_rfe,y)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    print("selected features rfe:", selected_features_rfe)

    pca_rfe = PCA(n_components=2)
    pcafit_rfe = pca_rfe.fit(movement_1_vs_movement_2[selected_features_rfe])
    reduced_movement_1_rfe = pcafit_rfe.transform(movement_1_complete[selected_features_rfe])
    reduced_movement_2_rfe = pcafit_rfe.transform(movement_2_complete[selected_features_rfe])



    #train SVM with RFE
    label_1 = np.array([1 for i in range(0,len(reduced_movement_1_rfe))])
    label_2 = np.array([0 for i in range(0, len(reduced_movement_2_rfe))])
    y = np.concatenate((label_1,label_2),axis=None)
    mov = np.vstack((reduced_movement_1_rfe,reduced_movement_2_rfe))
    df= pd.DataFrame(mov,columns=['PCA_1','PCA_2'])
    df2=df.copy(deep=True)
    # normalize PCA
    margin = 0.1 # 10% margin
    scale_max = 1 + margin
    scale_min = 1 - margin

    # min_max normalization with margin
    # normalize PCA 1
    df2['PCA_1'] = (df['PCA_1'] - (scale_min * df['PCA_1'].min())) / (
    (scale_max * (df['PCA_1'].max()) - (scale_min * df['PCA_1'].min())))
    # normalize PCA 2
    df2['PCA_2'] = (df['PCA_2'] - (scale_min * df['PCA_2'].min())) / (
    (scale_max * (df['PCA_2'].max()) - (scale_min * df['PCA_2'].min())))

    print("df2 PCA 1 max", df2['PCA_1'].max())
    print("df2 PCA 1 min", df2['PCA_1'].min())
    print("df2 PCA 2 max", df2['PCA_2'].max())
    print("df2 PCA 2 min", df2['PCA_2'].min())


    reduced_movement_1_rfe_normal = np.zeros_like(reduced_movement_1_rfe)
    reduced_movement_1_rfe_normal[:,0] = (reduced_movement_1_rfe[:,0] - (scale_min * df['PCA_1'].min())) / (
    (scale_max * (df['PCA_1'].max()) - (scale_min * df['PCA_1'].min())))
    reduced_movement_1_rfe_normal[:,1] = (reduced_movement_1_rfe[:,1] - (scale_min * df['PCA_2'].min())) / (
    (scale_max * (df['PCA_2'].max()) - (scale_min * df['PCA_2'].min())))

    reduced_movement_2_rfe_normal = np.zeros_like(reduced_movement_2_rfe)
    reduced_movement_2_rfe_normal[:,0] = (reduced_movement_2_rfe[:,0] - (scale_min * df['PCA_1'].min())) / (
    (scale_max * (df['PCA_1'].max()) - (scale_min * df['PCA_1'].min())))
    reduced_movement_2_rfe_normal[:,1] = (reduced_movement_2_rfe[:,1] - (scale_min * df['PCA_2'].min())) / (
    (scale_max * (df['PCA_2'].max()) - (scale_min * df['PCA_2'].min())))


    X = df2[['PCA_1','PCA_2']]

    # original (Soft Margin)

    # X_train, X_test, y_train, y_test, clf = fitting(X, y, 1, 'scale')

    # similar to RBF RNS

    X_train, X_test, y_train, y_test, clf = fitting(X, y, 1e7, 'auto')


toc = time.time()
tic_toc = toc - tic
print("Elapsed time for extraction-selection-reduction-train classifier:", tic_toc)
# plot functions
if plot:
    if plot_segmented:
        movement = 0
        repetition = 1
        subject = 1
        electrode = 0
        emg = movements_data_segmented[movement,subject][repetition][['emg_{}'.format(electrode)]].to_numpy()
        repetition_label = movements_data_segmented[movement,subject][repetition][['repetition']].to_numpy()
        segment = movements_data_segmented[movement,subject][repetition][['segment']].to_numpy()
        fig1, axs1 = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
        t_dim1 = emg.shape[0]
        t_1 = np.array([i for i in range(0, t_dim1)])
        axs1[0].set_title('Subject {}: EMG signal of movement {}, channel {}'.format(subject,movement,electrode))
        axs1[0].plot(t_1, emg)

        axs1[1].set_title('repetition')
        axs1[1].plot(t_1,repetition_label)

        axs1[2].set_title('segment')
        axs1[2].plot(t_1, segment)
    if plot_features:
        movement = 1
        repetition = 2
        subject = 2
        electrode = 4
        features_to_plot = ('MAV','ZC','RMS')
        emg = movements_data_segmented[movement, subject][repetition][
            ['emg_{}'.format(electrode)]].to_numpy()

        feature_1 = movements_data_segmented[movement,subject][repetition][
            ['{}_emg_{}'.format(features_to_plot[0], electrode)]].to_numpy()

        feature_2 = movements_data_segmented[movement, subject][repetition][
            ['{}_emg_{}'.format(features_to_plot[1], electrode)]].to_numpy()

        feature_3 = movements_data_segmented[movement, subject][repetition][
            ['{}_emg_{}'.format(features_to_plot[2], electrode)]].to_numpy()

        fig2, axs2 = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
        t_dim2 = emg.shape[0]
        t_2 = np.array([i for i in range(0, t_dim2)])

        axs2[0].set_title('Subject {}: EMG signal of movement {}, channel {}'.format(subject, movement, electrode))
        axs2[0].plot(t_2, emg)

        axs2[1].set_title('Feature {}'.format(features_to_plot[0]))
        axs2[1].plot(t_2, feature_1)

        axs2[2].set_title('Feature {}'.format(features_to_plot[1]))
        axs2[2].plot(t_2, feature_2)

        axs2[3].set_title('Feature {}'.format(features_to_plot[2]))
        axs2[3].plot(t_2, feature_3)

    if plot_corr_class:
        cor = movement_1_vs_movement_2_corr
        axs_cor = plt.axes()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, ax=axs_cor)
        axs_cor.set_title("Correlation Matrix (Pearson) for movement 1 vs movement 2 ")
    if plot_complete_movement:
        subject = 0
        electrode = 3
        movement_to_plot = movement_1[subject]['emg_{}'.format(electrode)].to_numpy()
        segment = movement_1[subject]['segment'].to_numpy()
        label = movement_1[subject]['label'].to_numpy()
        repetition = movement_1[subject]['repetition'].to_numpy()
        fig3, axs3 = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
        t_dim3 = movement_to_plot.shape[0]
        t_3 = np.array([i for i in range(0, t_dim3)])

        axs3[0].set_title('Subject {}: EMG signal of movement {}, channel {}'.format(subject,np.unique(label), electrode))
        axs3[0].plot(t_3, movement_to_plot)

        axs3[1].set_title('segment')
        axs3[1].plot(t_3, segment)

        axs3[2].set_title('repetition')
        axs3[2].plot(t_3, repetition)

        axs3[3].set_title('label')
        axs3[3].plot(t_3, label)
    if plot_PCA:
        fig_pca = plt.figure()
        ax_pca = fig_pca.add_subplot()
        ax_pca.scatter(reduced_movement_1[:, 0], reduced_movement_1[:, 1], c='red')
        ax_pca.scatter(reduced_movement_2[:, 0], reduced_movement_2[:, 1], c='blue')
        ax_pca.set_xlabel('PC 1')
        ax_pca.set_ylabel('PC 2')
    if plot_PCA_all_feat:
        fig_pca_all = plt.figure()
        ax_pca_all = fig_pca_all.add_subplot()
        ax_pca_all.scatter(reduced_movement_1_all[:, 0], reduced_movement_1_all[:, 1], c='red')
        ax_pca_all.scatter(reduced_movement_2_all[:, 0], reduced_movement_2_all[:, 1], c='blue')
        ax_pca_all.set_xlabel('PC 1')
        ax_pca_all.set_ylabel('PC 2')
    if plot_PCA_RFE:
        fig_pca_rfe = plt.figure()
        ax_pca_rfe = fig_pca_rfe.add_subplot()
        ax_pca_rfe.scatter(reduced_movement_1_rfe_normal[:, 0], reduced_movement_1_rfe_normal[:, 1], c='red')
        ax_pca_rfe.scatter(reduced_movement_2_rfe_normal[:, 0], reduced_movement_2_rfe_normal[:, 1], c='blue')
        ax_pca_rfe.set_xlabel('PC 1')
        ax_pca_rfe.set_ylabel('PC 2')
    if plot_SVM_RFE:
        Plot_3D(X, X_test, y_test, clf)


    plt.show()