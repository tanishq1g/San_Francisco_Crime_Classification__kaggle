import pandas as pd

class Data:

    def __init__(self, file_train, file_test = None):
        self.file_train = file_train
        self.file_test = file_test

    def data_import(self):
        self.train = pd.read_csv(self.file_train, parse_dates = ['Dates'])
        self.test = pd.read_csv(self.file_test, parse_dates = ['Dates'])

    def form_labels(self, label_col):
        le = preprocessing.LabelEncoder()
        self.train_labels = le.fit_transform(self.train[label_col])
   
    def drop_labels(self, label_col):
        self.train = self.train.drop(columns = [label_col])