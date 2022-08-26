import numpy as np
import pandas as pd
import scipy.sparse as sp
import numpy.linalg as LA
from scipy.signal import convolve2d
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta, MO
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial import distance
from pySankey.sankey import sankey
import math
import datetime
import subprocess, os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from prettytable import PrettyTable
import glob
from scipy.stats import norm


def transform(df):
    data0 = np.array(df)
    data = np.array(df)
    nr = len(data)
    nc = len(data[0])
    
    indexList = [np.any(i) for i in pd.isna(data)]
    data = np.delete(data, indexList, axis=0)

    data = np.concatenate([data0[:2,:], data])
    
    lam = 14400
    nr = len(data)
    nc = len(data[0])
    data2 = np.array([[0.0 for i in range(nc)] for j in range(nr - 2)])
    data3 = np.array([[0.0 for i in range(nc)] for j in range(nr - 2)])
    data4 = np.array([[0.0 for i in range(nc)] for j in range(nr - 2)])
    
    for j in range(2, nc):
        if data[0,j] == 1:
            for i in range(nr - 2):
                data2[i,j] = data[i+2, j]
            data3[:,j] = hpfilt(data2[:,j], lam)
            for i in range(nr - 2):
                data[i+2,j] = data2[i,j] - data3[i,j]
        elif data[0,j] == 2:
            for i in range(nr - 2):
                data2[i,j] = np.log(data[i + 2,j])*100
            data3[:,j] = hpfilt(data2[:,j], lam)
            for i in range(nr - 2):
                data[i+2,j] = data2[i,j] - data3[i,j]
        elif data[0,j] == 3:
            for i in range(nr - 2):
                data2[i,j] = np.log(data[i + 2,j])*100
            dats = data2[:,j]
            data4[:,j] = sadj(dats)
            data3[:,j] = hpfilt(data4[:,j], lam)
            for i in range(nr - 2):
                data[i+2,j] = data4[i,j] - data3[i,j]
    
    
       
    #df = pd.DataFrame(data)
    col_names = df.columns
    df=pd.DataFrame(data = data, columns=col_names)
    return df

