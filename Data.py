import pandas as pd

class Data:

    def __init__(self, file_train, file_test = None):
        self.file_train = file_train
        self.file_test = file_test

    def data_import(self):
        self.train = pd.read_csv(self.file_train, parse_dates = ['Dates'])
        self.test = pd.read_csv(self.file_test, parse_dates = ['Dates'])
