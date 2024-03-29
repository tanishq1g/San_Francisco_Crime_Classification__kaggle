import pandas as pd 
import numpy as np
from sklearn import preprocessing
import pygeohash as pgh
from copy import deepcopy
from sklearn.decomposition import PCA
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
    
    def add_feature(self, columns):
        self.features += columns

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

    def geohashing(self, train, test, precision = 3, pivot_col = 'Resolution', add_feature = True, is_onehot = True):
        train['geohashes'] = train.apply(lambda x: pgh.encode(x.X, x.Y, precision = precision), axis = 1)
        test['geohashes'] = test.apply(lambda x: pgh.encode(x.X, x.Y, precision = precision), axis = 1)
        # if(add_feature):
        #     self.features += geo1.unique().tolist()
        # for i in np.asarray(geo1.unique()):
        #     flag = 0
        #     for j in np.asarray(geo2.unique()):
        #         if(i == j):
        #             flag = 1
        #     if(flag == 0):
        #         test[i] = test[pivot_col].apply(lambda x: 0)
        #         if(add_feature):
        #             self.features += [j]

        # for i in np.asarray(geo2.unique()):
        #     flag = 0
        #     for j in np.asarray(geo1.unique()):
        #         if(i == j):
        #             flag = 1
        #     if(flag == 0):
        #         train[i] = train[pivot_col].apply(lambda x: 0)
        #         if(add_feature):
        #             self.features += [j]
        
        return train, test

    def X_Y_rot(self, data, add_feature = True):
        data[['new_X', 'new_Y']] = data[['X', 'Y']]
        sc = preprocessing.StandardScaler()
        data[['new_X', 'new_Y']] = sc.fit_transform(data[['X', 'Y']])
        data["rot45_X"], data["rot45_Y"] = .707 * data["new_Y"] + .707 * data["new_X"], .707 * data["new_Y"] - .707 * data["new_X"]
        data["rot30_X"], data["rot30_Y"] = (1.732/2) * data["new_X"] + (1./2) * data["new_Y"], (1.732/2) * data["new_Y"] - (1./2) * data["new_X"]
        data["rot60_X"], data["rot60_Y"] = (1./2) * data["new_X"] + (1.732/2) * data["new_Y"], (1./2)* data["new_Y"] - (1.732/2) * data["new_X"]
        data["radial_r"] = np.sqrt( np.power(data["new_Y"], 2) + np.power(data["new_X"], 2))
        if(add_feature):
            self.features += ['rot60_X', 'rot60_Y', 'rot30_X', 'rot30_Y', 'rot45_X', 'rot45_Y', 'radial_r']
 
    def odds_base_target(self, train, test, base, target, col_name_prefix, add_base_odds = False, add_feature = True): #odds of target given base
        bas_sort = sorted(train[base].unique())
        tar_sort = sorted(train[target].unique())
        tar_counts = train.groupby([target]).size()
        bas_tar_counts = train.groupby([base, target]).size()
        bas_counts = train.groupby([base]).size()
        logodds = {}
        logoddsPA = {}
        MIN_CAT_COUNTS = 2
        tar_logodds = np.log(tar_counts / len(train)) - np.log(1.0 - tar_counts / float(len(train)))
        for bas in bas_sort:
            PA = bas_counts[bas] / float(len(train))
            logoddsPA[bas] = np.log(PA) - np.log(1.- PA)
            logodds[bas] = deepcopy(tar_logodds)
            for tar in bas_tar_counts[bas].keys():
                if (bas_tar_counts[bas][tar] > MIN_CAT_COUNTS) and bas_tar_counts[bas][tar] < bas_counts[bas]:
                    PA = bas_tar_counts[bas][tar] / float(bas_counts[bas])
                    logodds[bas][tar_sort.index(tar)] = np.log(PA) - np.log(1.0 - PA)
            logodds[bas] = pd.Series(logodds[bas])
            logodds[bas].index = range(len(tar_sort))
        bas_features = train[base].apply(lambda x: logodds[x])
        bas_features.columns = [col_name_prefix + "_odds" + str(x) for x in range(len(bas_features.columns))]
        train = pd.concat([train, bas_features], axis = 1)
        if(add_base_odds):
            train[base + '_odds'] = train[base].apply(lambda x: logoddsPA[x])
        if(add_feature):
            self.features += bas_features.columns.tolist()
        
        new_bas_sort = sorted(test[base].unique())
        new_bas_counts = test.groupby(base).size()
        only_new = set(new_bas_sort + bas_sort) - set(bas_sort)
        only_old = set(new_bas_sort + bas_sort) - set(new_bas_sort)
        in_both = set(new_bas_sort).intersection(bas_sort)
        for bas in only_new:
            PA = new_bas_counts[bas] / float(len(test) + len(train))
            logoddsPA[bas] = np.log(PA) - np.log(1.- PA)
            logodds[bas] = deepcopy(tar_logodds)
            logodds[bas].index = range(len(tar_sort))
        for bas in in_both:
            PA = (bas_counts[bas] + new_bas_counts[bas]) / float(len(test) + len(train))
            logoddsPA[bas] = np.log(PA) - np.log(1.- PA)
        bas_features_te = test[base].apply(lambda x: logodds[x])
        bas_features_te.columns = [col_name_prefix + "_odds" + str(x) for x in range(len(bas_features_te.columns))]
        test = pd.concat([test, bas_features_te], axis = 1)
        if(add_base_odds):
            test[base + '_odds'] = test[base].apply(lambda x: logoddsPA[x])

        return train, test

    def bc_wc_oc(self, data, add_feature = False):
        white_crime = ["FRAUD", "FORGERY/COUNTERFEITING", "BAD CHECKS" , "EXTORTION", "EMBEZZLEMENT", "SUSPICIOUS OCC", "BRIBERY", "GAMBLING"]
        blue_crime = ["VANDALISM", "LARCENY/THEFT", "STOLEN PROPERTY", "ROBBERY", "DRIVING UNDER THE INFLUENCE", "DISORDERLY CONDUCT", "LIQUOR LAWS", "VEHICLE THEFT", "ASSAULT", "KIDNAPPING", "TRESPASS", "ARSON", "RECOVERED VEHICLE",  "SEX OFFENSES FORCIBLE","WEAPON LAWS", "DRUG/NARCOTIC", "FAMILY OFFENSES", "BURGLARY"]
        other_crime = ["MISSING PERSON", "RUNAWAY",  'PROSTITUTION', "DRUNKENNESS", "SUICIDE",  "LOITERING", "OTHER OFFENSES", "NON-CRIMINAL", "WARRANTS", "SECONDARY CODES"]

        data['bc_wc_oc'] = data['Category'].apply(lambda x: 'b' if x in blue_crime else ('w' if x in white_crime else ('o' if x in other_crime else 'error')))
        if 'error' in data.train.bc_wc_oc.unique():
            print('all categories not found')
        if(add_feature):
            self.features += ['bc_wc_oc']

    def apply_PCA(self, data, n_components):
        pca = PCA(n_components = n_components)
        pca.fit(data)
        new_data = pca.transform(data)
        return new_data

    def add_cols(self, data, col1, col2, col_type, new_col_name):
        data[new_col_name] = data[col1].astype(col_type) + data[col2].astype(col_type) 

    def correlated_columns(self, data, lim):
        corr = data.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= lim or corr.iloc[i,j] <= -lim:
                    if columns[j]:
                        columns[j] = False
        return columns