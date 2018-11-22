import pandas as pd 
import numpy as np
from sklearn import preprocessing
import pygeohash as pgh

class Feature_Engineering:

    def __init__(self):
        self.features = []

    def extract_dt_time(self, data):
        data['Hour'] = data.Dates.dt.hour
        data['Year'] = data.Dates.dt.year
        data['Month'] = data.Dates.dt.month
        data['Minute'] = data.Dates.dt.minute - 30
    
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
        elif(nbins == 3):
            data['night'] = data['Hour'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 6,7] else 0)
            data['morning'] = data['Hour'].apply(lambda x: 1 if x in [8, 9, 10, 11] else 0)
            data['evening'] = data['Hour'].apply(lambda x: 1 if x in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0] else 0)
            if(add_feature):
                self.features += ['morning', 'night', 'evening']
        elif(nbins == 4):
            data['morning'] = data['Hour'].apply(lambda x: 1 if x in [7, 8, 9, 10, 11] else 0)
            data['evening'] = data['Hour'].apply(lambda x: 1 if x in [17, 18, 19, 20, 21, 22] else 0)
            data['night'] = data['Hour'].apply(lambda x: 1 if x in [23, 0, 1, 2, 3, 4, 5, 6] else 0)
            data['afternoon'] = data['Hour'].apply(lambda x: 1 if x in [12, 13, 14, 15, 16] else 0)
            if(add_feature):
                self.features += ['morning', 'night', 'evening', 'afternoon']

    def geohashing(self, train, test, precision = 8, pivot_col = 'Resolution', add_feature = True):
        geo1 = train.apply(lambda x: pgh.encode(x.X, x.Y, precision = precision), axis = 1)
        train = pd.concat([train, pd.get_dummies(geo1)], axis = 1)
        geo2 = test.apply(lambda x: pgh.encode(x.X, x.Y, precision = precision), axis = 1)
        test = pd.concat([test, pd.get_dummies(geo2)], axis = 1)
        if(add_feature):
            self.features += geo1.unique().tolist()
        c1 = 0
        c2 = 0
        for i in np.asarray(geo1.unique()):
            flag = 0
            for j in np.asarray(geo2.unique()):
                if(i == j):
                    flag = 1
                    c1 += 1
            if(flag == 0):
                c2 += 1
                print('unique',i)
                test[j] = test[pivot_col].apply(lambda x: 0)
                if(add_feature):
                    self.features += [j]
        print('count',c1,c2)
        c1 = 0
        c2 = 0
        for i in np.asarray(geo2.unique()):
            flag = 0
            for j in np.asarray(geo1.unique()):
                if(i == j):
                    flag = 1
                    c1 += 1
            if(flag == 0):
                c2 += 1
                print('unique',i)
                train[j] = train[pivot_col].apply(lambda x: 0)
                if(add_feature):
                    self.features += [j]
        print('count',c1,c2)
        return train, test

    def X_Y_rot(self, data, add_feature = True):
        sc = preprocessing.StandardScaler()
        sc.fit(data[['X', 'Y']])
        data[['new_X'], ['new_Y']] = sc.transform([['X', 'Y']])
        data["rot45_X"], data["rot45_Y"] = .707 * data["new_Y"] + .707 * data["new_X"], .707 * data["new_Y"] - .707 * data["new_X"]
        data["rot30_X"], data["rot30_Y"] = (1.732/2) * data["new_X"] + (1./2) * data["new_Y"], (1.732/2) * train["Y"] - (1./2) * data["new_X"]
        data["rot60_X"], data["rot60_Y"] = (1./2) * data["new_X"] + (1.732/2) * data["new_Y"], (1./2)* data["new_Y"] - (1.732/2) * data["new_X"]
        data["radial_r"] = np.sqrt( np.power(data["new_Y"], 2) + np.power(data["new_X"], 2))
        if(add_feature):
            self.features += ['rot60_X', 'rot60_Y', 'rot30_X', 'rot30_Y', 'rot45_X', 'rot45_Y', 'radial_r']

            
    

        
