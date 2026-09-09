from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def get_keys(dict):
    list = []
    for k in dict.keys():
        list.append(k)
    return list

#load databases
s1_e1_a1 = loadmat('S1_E1_A1.mat')
s1_e2_a1 = loadmat('S1_E2_A1.mat')
s1_e3_a1 = loadmat('S1_E3_A1.mat')

#MATLAB fields in databases
raw_keys = get_keys(s1_e1_a1)
raw_keys2 = get_keys(s1_e2_a1)
#fields without MATLAB headers
keys = raw_keys[3: len(raw_keys)]
keys2 = raw_keys2[3: len(raw_keys2)]

print("keys 1:", keys)
print("keys 2:", keys2)

#emg data

emg1 = np.array(s1_e1_a1[keys[0]]) #exercise 1 - finger and wrist movements (dedos)
emg2 = np.array(s1_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre)
emg3 = np.array(s1_e3_a1[keys[0]]) # exercise 3 - force movements (fuerza)
number_emg_sensors = emg1.shape[1]

#exercise data

exe1 = np.array(s1_e1_a1[keys[6]]) #exercise 1 - finger and wrist movements (dedos)
exe2 = np.array(s1_e2_a1[keys[6]]) # exercise 2 - grasping movements (agarre)
exe3 = np.array(s1_e3_a1[keys[6]]) # exercise 3 - force movements (fuerza)

#restimulus data - verified stimulus data

restim1 = np.array(s1_e1_a1[keys[8]]) #exercise 1 - finger and wrist movements (dedos)
restim2 = np.array(s1_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre)
restim3 = np.array(s1_e3_a1[keys[8]]) # exercise 3 - force movements (fuerza)

#repetition data

rep1 = np.array(s1_e1_a1[keys[7]]) #exercise 1 - finger and wrist movements (dedos)
rep2 = np.array(s1_e2_a1[keys[7]]) # exercise 2 - grasping movements (agarre)
rep3 = np.array(s1_e3_a1[keys[7]]) # exercise 3 - force movements (fuerza)

print('size of emg signal:', emg1.shape)
print('number of emg electrodes:', number_emg_sensors)

print('exercise 1:', exe1)
print('exercise 2:', exe2)
print('exercise 3:', exe3)

print('restimulus 1:', restim1)
print('restimulus 1 label example, must be in range 0 to 17:', restim1[223420][0])
print('restimulus 2:', restim2)
print('restimulus 2 label example, must be in range 0 to 17:', restim2[223420][0])
print('restimulus 3:', restim3)
print('restimulus 3 label example:', restim3[223420][0])


print('possible labels\n')
print('restimulus field in subject 1: possible labels from exercise 1 :', np.unique(restim1))
print('restimulus field in subject 1: possible labels from exercise 2 :', np.unique(restim2))
print('restimulus field in subject 1: possible labels from exercise 3 :', np.unique(restim3))

#load databases
s2_e1_a1 = loadmat('S2_E1_A1.mat')
s2_e2_a1 = loadmat('S2_E2_A1.mat')
s2_e3_a1 = loadmat('S2_E3_A1.mat')
#MATLAB fields in databases
raw_keys_s2 = get_keys(s2_e1_a1)
raw_keys2_s2 = get_keys(s2_e2_a1)
# emg signals
emg1_s2 = np.array(s2_e1_a1[keys[0]]) #exercise 1 - finger and wrist movements (dedos)
emg2_s2 = np.array(s2_e2_a1[keys[0]]) # exercise 2 - grasping movements (agarre)
emg3_s2 = np.array(s2_e3_a1[keys[0]]) # exercise 3 - force movements (fuerza)

#restimulus data - verified stimulus data

restim1_s2 = np.array(s2_e1_a1[keys[8]]) #exercise 1 - finger and wrist movements (dedos)
restim2_s2 = np.array(s2_e2_a1[keys[8]]) # exercise 2 - grasping movements (agarre)
restim3_s2 = np.array(s2_e3_a1[keys[8]]) # exercise 3 - force movements (fuerza)
print('restimulus field in subject 2: possible labels from exercise 1 :', np.unique(restim1_s2))
print('restimulus field in subject 2: possible labels from exercise 2 :', np.unique(restim2_s2))
print('restimulus field in subject 2: possible labels from exercise 3 :', np.unique(restim3_s2))


print('repetition 1:', rep1)
print('repetition 2:', rep2)
print('repetition 3:', rep3)



fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))
t_dim= emg2.shape[0]
channel = 10
print('emg signal: ', emg2[:,channel])
t = np.array([i for i in range(0,t_dim)])

axs[0].set_title('Subject 1: EMG signal exercise 2 (all movements), channel {}' .format(channel))
axs[0].plot(t,emg2[:,0])

axs[1].set_title('movement label')
axs[1].plot(t, restim2[:,0])

fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))
t_dim_2= emg2_s2.shape[0]
channel = 10
print('emg signal: ', emg2_s2[:,channel])
t2 = np.array([i for i in range(0,t_dim_2)])

axs2[0].set_title('Subject 2: EMG signal exercise 2 (all movements), channel {}' .format(channel))
axs2[0].plot(t2,emg2_s2[:,0])

axs2[1].set_title('movement label')
axs2[1].plot(t2, restim2_s2[:,0])


plt.show()


sample_rows = (emg1.shape[0]) + (emg2.shape[0]) + (emg3.shape[0]) # number of samples
emg_label_1 = np.hstack((emg1, restim1)) # emg and movement label from dataset E1
emg_label_2 = np.hstack((emg2, restim2)) # emg and movement label from dataset E2
emg_label_3 = np.hstack((emg3, restim3)) # emg and movement label from dataset E3

# concatenate labeled E1, E2 and E3
dataset_ordered = np.vstack((emg_label_1, emg_label_2))
dataset_ordered = np.vstack((dataset_ordered, emg_label_3))

print("sample rows:", sample_rows)
print("dataset size", dataset_ordered.shape)


