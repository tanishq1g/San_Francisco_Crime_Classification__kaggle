import pandas as pd 
import numpy as np
from sklearn import preprocessing

class Data_Preprocessing:

    def form_labels(self, data, label_col):
        le = preprocessing.LabelEncoder()
        data[label_col] = le.fit_transform(data[label_col])
    
    def drop_labels(self, data, label_col):
        return data.drop(columns = [label_col])

    def X_Y_remove_outliers(self, data):
        data = data[data['X'] < -122.25]
        data = data[data['Y'] < 40]