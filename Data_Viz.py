import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Data_Viz:

    def value_counts_graphs(self, train, columns):
        for col in columns:

    def chk_corr(self, data):
        plt.figure(figsize = (20, 15))
        sns.heatmap(data.corr(), annot = True)
        plt.show()  

    def twoval_jointplot(self, column1, column2, data, kind = "scatter"):
        plt.figure(figsize = (20, 15))
        sns.jointplot(x = column1, y = column2, data = data, kind = kind)
        plt.show()     

    def plot_hists(self, data):
        data.hist(bins = 200, figsize = (20, 15))

    def map_kde_plot(self, x, y, category, data, map_path):
        map = np.loadtxt(map_path)
        plt.figure(figsize = (20, 15))
        kde = sns.kdeplot(x = data[data["Category"] = category][x], y = data[data["Category"] = category][y])
        kde.imshow(map, cmap=plt.get_cmap('gray'))

    