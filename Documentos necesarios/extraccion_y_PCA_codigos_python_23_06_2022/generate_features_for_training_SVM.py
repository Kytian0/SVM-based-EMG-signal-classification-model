import pickle
import numpy as np
import pandas as pd

def save_output_data(name,dataframe):
    dataframe.to_pickle(name)
def read_output_data(name):
    return pd.read_pickle(name)

data = pd.DataFrame({'MAV':[0.12,11.43,34.5],'ZC':[1,2,3]})
print("data",data)
save_output_data("features_pickle.pkl",data)
data_2 = read_output_data("features_pickle.pkl")
print("data_2",data_2)