import pandas as pd 
import numpy as np
from sklearn import preprocessing

class Feature_Engineering:

    def __init__(self):
        self.features = []

    def X_Y_remove_outliers(self, data):
        data = data[data['X'] < -122.25]
        data = data[data['Y'] < 40]

    def extract_dt_time(self, data):
        data['Hour'] = data.Dates.dt.hour
        data['Year'] = data.Dates.dt.year
        data['Month'] = data.Dates.dt.month
        data['Minute'] = data.Dates.dt.minute
    
    def onehot(self, data, columns, add_feature = True):
        for col in columns:
            onehot = pd.get_dummies(data[col])
            onehot.columns = [col + '_' + str(x) for x in onehot.columns]
            data = pd.concat([data, onehot], axis = 1)
            if(add_feature):
                self.features += onehot.columns.tolist()
        return data        
    
    def add_seasons(self, data, add_feature = True):
        data['Summer'] = data['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
        data['Winter'] = data['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        data['Autumn'] = data['Month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
        data['Spring'] = data['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
        if(add_feature):
            self.features += ['Summer', 'Winter', 'Autumn', 'Spring']

    def add_crossing(self, data, add_feature = True):
        data['crossing'] = data['Address'].apply(lambda x: 1 if (x.find('/') != -1) else 0)
        if(add_feature):
            self.features += ['crossing']

    def Hour_bins(self, data, nbins = 4, add_feature = True):
        if(nbins == 2):
            data['morning'] = data['Hour'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 6,7, 8, 9, 10, 11] else 0)
            data['night'] = data['Hour'].apply(lambda x: 1 if x in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0] else 0)
            if(add_feature):
                self.features += ['morning', 'night']
        else if(nbins == 3):
            data['night'] = data['Hour'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 6,7] else 0)
            data['morning'] = data['Hour'].apply(lambda x: 1 if x in [8, 9, 10, 11] else 0)
            data['evening'] = data['Hour'].apply(lambda x: 1 if x in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0] else 0)
            if(add_feature):
                self.features += ['morning', 'night', 'evening']
        else if(nbins == 4):
            data['morning'] = data['Hour'].apply(lambda x: 1 if x in [7, 8, 9, 10, 11] else 0)
            data['evening'] = data['Hour'].apply(lambda x: 1 if x in [17, 18, 19, 20, 21, 22] else 0)
            data['night'] = data['Hour'].apply(lambda x: 1 if x in [23, 0, 1, 2, 3, 4, 5, 6] else 0)
            data['afternoon'] = data['Hour'].apply(lambda x: 1 if x in [12, 13, 14, 15, 16] else 0)
            if(add_feature):
                self.features += ['morning', 'night', 'evening', 'afternoon']

    

        
