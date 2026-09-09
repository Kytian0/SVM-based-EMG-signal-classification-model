from scipy.io import loadmat
import numpy as np


def get_keys(dict):
    list = []
    for k in dict.keys():
        list.append(k)
    return list

#  import subject samples
female_1 = loadmat('female_1.mat')
female_2 = loadmat('female_2.mat')
female_3 = loadmat('female_3.mat')
male_1 = loadmat('male_1.mat')
male_2 = loadmat('male_2.mat')
subject_number = 5

raw_movements = get_keys(female_1)  # raw movements names
channels_number = 2  # number of electrodes

movements = raw_movements[3:len(raw_movements)]  # movements names without MATLAB headers (first three positions)
movements_number = len(movements)/channels_number
print("raw movements: ", raw_movements)
print("movements without MATLAB headers: ", movements)
print("number of movements: ", movements_number)

# obtain dataset - ordered by channels
dataset = []
movement_names = []
label_size = 0
for i in range(0, len(movements)):

    # subject 1
    dataset.append(female_1[movements[i]]) # i_th movement samples from subject 1
    label_size = len(female_1[movements[i]]) # movement label size
    # subject 2
    dataset.append(female_2[movements[i]]) # i_th movement samples from subject 2
    label_size = label_size + len(female_2[movements[i]]) # movement label size
    # subject 3
    dataset.append(female_3[movements[i]]) # i_th movement samples from subject 3
    label_size = label_size + len(female_3[movements[i]]) # movement label size
    # subject 4
    dataset.append(male_1[movements[i]]) # i_th movement samples from subject 4
    label_size = label_size + len(male_1[movements[i]]) # movement label size
    # subject 5
    dataset.append(male_2[movements[i]]) # i_th movement samples from subject 5
    label_size = label_size + len(male_2[movements[i]]) # movement label size

    actual_name = [movements[i] for j in range(0, label_size)] #create label vector for actual movement

    movement_names.append(actual_name)# append movement labels

# print(female_1[movements[0]])

print(movement_names)
print('dataset size: ', len(dataset))
print(dataset)

# obtain dataset - ordered by pair of channels for each movement type- REVISAR

movement_names_ordered = []
label_size = 0
subject_set = [female_1, female_2, female_3, male_1, male_2]
subj = subject_set[0] #sujeto 0
r = subj[movements[0]] # movimiento 0
sample_rows = len(r[0]) # cantidad de datos tomada en cada muestra de movimiento
dataset_ordered = np.zeros((1, sample_rows)) # valor dummy para lograr el uso del np.append
a = 0
for i in range(0, len(movements) - 1, 2):

    for j in range(0, len(subject_set)):
        # subject j with i_th movement
        subject = subject_set[j]
        # ch 1
        mat_ch_1 = np.array(subject[movements[i]])

        # ch 2
        mat_ch_2 = np.array(subject[movements[i + 1]])

        #order in pair of channels

        for k in range(0, mat_ch_1.shape[0]):
            # append channel 1 and after channel 2
            #ch1
            ch1 = mat_ch_1[k]
            dataset_ordered = np.vstack((dataset_ordered, ch1))
            #ch2
            ch2 = mat_ch_2[k]
            dataset_ordered = np.vstack((dataset_ordered, ch2))

            # append channel 1 and after channel 2 movement names
            #ch1
            movement_names_ordered.append(movements[i])
            #ch2
            movement_names_ordered.append(movements[i+1])
            a = mat_ch_1.shape[0]
dataset_ordered = np.delete(dataset_ordered,0,0)
print(movement_names_ordered)
print('ordered dataset size: ', dataset_ordered.shape)
print('ordered dataset: ', dataset_ordered)
print(a)

