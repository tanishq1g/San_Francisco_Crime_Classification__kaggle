import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Data_Viz:

    def value_counts_graphs(self, train, test, columns):
        for col in columns:
            
