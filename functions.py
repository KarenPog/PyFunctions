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

def myadd(a,b):
    c = a+b
    return c


def delete_figures():
    removing_files = glob.glob(r'C:\Users\Karen\Desktop\AlertPy\*.png')
    for i in removing_files:
        os.remove(i)

def delete_datfile():
    removing_files = glob.glob(r'C:\Users\Karen\Desktop\AlertPy\*.dat')
    for i in removing_files:
        os.remove(i)
        
def impute(df):
    
    for i in range(2, len(df.columns)):
        df.iloc[:,i] = df.iloc[:,i].interpolate(method='spline', order=1, limit_direction = 'forward')
    return df

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

def hpfilt(y, w):    
   
    #y = np.array(y)
    
    #y = y[~np.isnan(y)]
    m = len(y)
    d = np.tile(np.array([w, -4*w ,((6*w+1)/2)]),(m,1))
    d[0,1] = -2*w
    d[0,2] = (1+w)/2
    d[1,2] = (5*w+1)/2
    d[m-2,1]=-2*w
    d[m-1,2]=(1+w)/2
    d[m-2,2]=(5*w+1)/2
    B = np.array(sp.spdiags(np.transpose(d), [-2,-1,0], m, m).todense())
    B = B+np.transpose(B)
    smooth_hp = [[]]
    smooth_hp = np.matmul(LA.inv(B),y)
    smooth_final = np.transpose(smooth_hp)
    return smooth_final
     
def sadj(Data):
    y = Data
    T = len(Data)
    sW13 = np.tile(1/12,(11)).tolist()
    sW13.insert(0,1/24)
    sW13.append(1/24)
    yS = np.convolve(y,sW13,'same')
    yS[0:6] = yS[6]
    yS[T-6:T] = yS[T-7]
    np.seterr(divide='ignore', invalid='ignore')
    xt = np.divide(y, yS).astype(float)
    
    s = 12
    sidx = []
    for i in range(0, s):
        sidx.append(np.arange(i,T,s))


    sW3 = np.array([1/9,2/9,1/3,2/9,1/9])
    aW3 = np.array([[.259, .407],[.37, .407], [.259, .185], [.111, 0]])

    shat = np.array([0.0 for k in range(T)])

    for i in range(s):
        ns = len(sidx[i])
        first = np.arange(0,4,1)
        ind = np.arange(0,ns,1)
        last = ind[-4:]
        cr = sidx[i]
        dat = np.array([0.0 for k in range(ns)])
        for ii in range(ns):
            dat[ii] = xt[cr[ii]]

        sd = np.convolve(dat,sW3,'same')
        dat1 = np.array([[0.0 for i in range(1)] for j in range(4)])
        for i in first:
            dat1[i,0] = dat[i]
        fd = convolve2d(dat1, np.rot90(aW3,2), 'valid')
        sd[0] = fd[0,0]
        sd[1] = fd[0,1]
        for i in range(4):
            dat1[i,0] = dat[last[i]]
        fd = convolve2d(dat1, aW3, 'valid')
        sd[ns-2] = fd[0,0]
        sd[ns-1] = fd[0,1]
        for ii in range(ns):
            shat[cr[ii]] = sd[ii]


    sb = np.convolve(shat,sW13,'same');
    sb[0:6] = sb[s:s+6]
    sb[T-6:T] = sb[T-s-6:T-s]
    s33 = shat/sb

    dt = y/s33
    sWH = np.array([-0.019, -0.028, 0, 0.066, 0.147, 0.214, 0.24, 0.214, 0.147,0.066, 0, -0.028, -0.019])
    aWH = np.array([[-0.034, -0.017,   0.045,   0.148,   0.279,   0.421],
           [-.005,  .051,   .130,   .215,   .292,   .353],
            [.061,   .135,   .201,   .241,   .254,   .244],
            [.144,   .205,   .230,   .216,   .174,   .120],
            [.211,   .233,   .208,   .149,   .080,   .012],
            [.238,   .210,   .144,   .068,   .002,  -.058],
            [.213,   .146,   .066,   .003,  -.039,  -.092],
            [.147,   .066,   .004,  -.025,  -.042,  0    ],
            [.066,   .003,  -.020,  -.016,  0,      0    ],
            [0.001,  -0.022,  -0.008,  0,      0,      0],
            [-.026,  -.011,   0,     0,      0,      0  ],
            [-.016,   0,      0,     0,      0,      0 ]]);
    first = np.arange(0,12,1)
    last = np.arange(T-12,T,1)
    h13 = np.convolve(dt,sWH,'same')

    dat1 = np.array([[0.0 for i in range(1)] for j in range(len(last))])
    for i in range(len(last)):
        dat1[i,0] = dt[last[i]]
    h13[T-6:T] = convolve2d(dat1, aWH, 'valid')


    for i in range(len(first)):
        dat1[i,0] = dt[first[i]]
    h13[0:6] = convolve2d(dat1, np.rot90(aWH,2), 'valid')
    np.seterr(divide='ignore', invalid='ignore')
    xt = np.divide(y, h13)
                 
    sW5 = np.tile(1/5,(3)).tolist()    
    sW5.insert(0,1/15)
    sW5.insert(1,2/15)
    sW5.append(2/15)
    sW5.append(1/15)

    aW5 = np.array([[0.150,0.250, 0.293],
           [.217, .250, .283],
           [.217 ,.250, .283],
           [.217 ,.183, .150],
           [.133, .067,    0],
           [.067 ,  0,     0]]);

    for i in range(s):
        ns = len(sidx[i])
        first = np.arange(0,6,1)
        ind = np.arange(0,ns,1)
        last = ind[-6:]
        cr = sidx[i]
        dat = np.array([0.0 for k in range(ns)])
        for ii in range(ns):
            dat[ii] = xt[cr[ii]]

        sd = np.convolve(dat,sW5,'same')
        dat1 = np.array([[0.0 for i in range(1)] for j in range(6)])
        for i in first:
            dat1[i,0] = dat[i]
        fd = convolve2d(dat1, np.rot90(aW5,2), 'valid')
        sd[0] = fd[0,0]
        sd[1] = fd[0,1]
        sd[2] = fd[0,2]
        for i in range(6):
            dat1[i,0] = dat[last[i]]
        fd = convolve2d(dat1, aW5, 'valid')
        sd[ns-3] = fd[0,0]
        sd[ns-2] = fd[0,1]
        sd[ns-1] = fd[0,2]
        for ii in range(ns):
            shat[cr[ii]] = sd[ii]
    sb = np.convolve(shat,sW13,'same');
    sb[0:6] = sb[s:s+6]
    sb[T-6:T] = sb[T-s-6:T-s]
    s35 = shat/sb
    dt = y/s35
    return dt

def signal_to_noise_ratio(df, prds, param):
    
    str_num = len(prds)
    lst = np.zeros(str_num).astype(str)
    for i in range(str_num):
        if len(prds[i]) <= 10:
            lst[i] =  df[df['date'] == prds[i]].index[0]
        else:
            lst2 = prds[i].split(':')
            p1 = df[df['date'] == lst2[0]].index[0]
            p2 = df[df['date'] == lst2[1]].index[0]
            p = str(p1)+'-'+str(p2)
            lst[i] = p
    
    f1 = lst[0].split('-')
    f2 = lst[len(lst)-1].split('-')
    if len(f1) == 1 and int(f1[0]) - 30 >=2:
        start_index = int(f1[0]) - 30
    elif len(f1) == 2 and int(f1[0]) - 30 >=2:
        start_index = int(f1[0]) - 30
    else:
        start_index = 2

    if len(f2) == 1 and int(f2[0]) + 24 <=len(df):
        end_index = int(f2[0]) + 24
    elif len(f2) == 2 and int(f2[1]) + 24 <=len(df):
        end_index = int(f2[1]) + 24    
    else:
        end_index = len(df)
   
        
    data0 = np.array(df)
    data = np.array(df)
    data = data[start_index:end_index,1].astype(float)
    avr = np.mean(data, axis = 0)
    sd = np.std(data, axis = 0)
    if data0[1,1] < 0:
        threshold = avr - param*sd
    else:
        threshold = avr + param*sd

    data1 = np.array([[0 for i in range(1)] for j in range(len(data))])

    data2 = np.array([[0 for i in range(len(data))] for j in range(1)])
    data1_1 = np.array([[0 for i in range(1)] for j in range(1)])
    
    for i in range(len(data)):
        if threshold < 0 and data[i] > threshold:
            data1[i,:] = 0
        elif threshold < 0 and data[i] < threshold:
            data1[i,:] = 1

        if threshold > 0 and data[i] < threshold:
            data1[i,:] = 0
        elif threshold > 0 and data[i] > threshold:
            data1[i,:] = 1

    for i in range(str_num):
        lst1 = lst[i].split('-')
        if len(lst1) == 1:
            ind = int(lst[i])
            data1_1 = np.concatenate((data1_1, data1[ind-start_index-6:ind-start_index,:]), axis = 0)
            
        if len(lst1) == 2:
            p = lst[i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            data1_1 = np.concatenate((data1_1, data1[p1-start_index-6:p1-start_index,:]), axis = 0)
    data1_1 = np.delete(data1_1, 0, 0)
    data1_1 = np.sum(data1_1.reshape(int(len(data1_1)/6),6), axis = 1)
    for i in range(len(data1_1)):
        if data1_1[i] == 0:
            data1_1[i] = 1
        else:
            data1_1[i] = 0
    
    for i in range(str_num):
        lst1 = lst[str_num-1-i].split('-')
        if len(lst1) == 1:
            ind = int(lst[str_num-1-i])
            data1 = np.delete(data1, np.s_[ind-start_index-6:ind-start_index+1], axis = 0)
        if len(lst1) == 2:
            p = lst[str_num-1-i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            data1 = np.delete(data1, np.s_[p1-start_index-6:p2-start_index], axis = 0)



    fn = np.sum(data1_1)
    tp = str_num - fn
    fp = np.sum(data1)
    tn = end_index - start_index
    StN = ((fp/(fp+tn))/(tp/(tp+fn)))

    return threshold, StN, start_index, end_index, lst

def optimal_StN(df, prds, bounds):
    nc = df.shape[1]
    f = np.linspace(bounds[0], bounds[1], 1000, endpoint=True)
    arr = np.array([0.0 for i in range(len(f))])
    arr2 = np.array([0.0 for i in range(len(f))])
    #results = np.array([[0.0 for i in range(len(f))] for j in range(2)])
    results2 = np.array([[0.0 for i in range(nc-2)] for j in range(2)])
    
    for i in range(2, nc):
        results = np.array([[0.0 for i in range(len(f))] for j in range(2)])
        for j in range(len(f)):
            res = signal_to_noise_ratio(df.iloc[:,[1,i]], prds, f[j])
            results[0, j] = res[0]
            results[1, j] = res[1]
        results = results[:,(results[1,:] > 0.01)]
        f1 = np.amin(results[1,:], axis = 0)
        min_index = np.argmin(results[1,:],axis=0)
        f2 = results[0, min_index]
        results2[0, i - 2] = f2
        results2[1, i - 2] = f1
    start_index = res[2]
    end_index = res[3]
    df1 = df.iloc[start_index:,:]
    df1 = np.array(df1)

    for i in range(2,len(df1[0])):
        if df.iloc[1,i] == -1:
            for j in range(len(df1)):
                if df1[j,i] < results2[0, i-2]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0
        if df.iloc[1,i] == 1:
            for j in range(len(df1)):
                if df1[j,i] > results2[0, i-2]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0

    col_names = df.columns
    
    df1 = pd.DataFrame(df1)
    df1 = pd.DataFrame(data = df1.values, columns=col_names)
    results2=pd.DataFrame(data = results2)
    results2.insert(0, "Показатели", " ")
    results2.loc[0, "Показатели"] = "Пороговый уровень"
    results2.loc[1, "Показатели"] = "Сигнал/Шум отношение"

    return results2, df1, res[2], res[4]

def correl(x0, prds, start_index, stn):
    
    x = np.array(x0)
    x = x[:,2:]
    y = np.array([[0 for i in range(1)] for j in range(len(x))])
    str_num = prds
    for i in range(len(prds)):
        if len(str_num[i].split('-')) == 1:
            y[int(str_num[i]) - start_index, :] = 1
        elif len(str_num[i].split('-')) == 2:
            f = str_num[i].split('-')
            f1 = int(f[0])
            f2 = int(f[1])
            y[f1 - start_index:f2-start_index, :] = 1
    
    init = 0
    nobs = len(x)
   # 
    nvar = len(x[0])
    arr2 = np.array([[0.0 for i in range(12)] for j in range(nvar)] )
    for i in range(nvar):
        x1 = x[:,i]
        xlag = np.ones((nobs,12))*init
        icnt = 0
        for j in range(12):
            xlag[list(range(j+1, nobs)), icnt +j] = x1[list(range(0, nobs - (j+1)))]
      # icnt = icnt+n
        nr = len(xlag)
        nc = len(xlag[0])
        xlag = xlag[12:nr,:]

        y1 = y[12:nr,:]
        arr = np.hstack((y1, xlag))
        

    
    
        for j in range(12):
            coeff = np.corrcoef(arr[:,0], arr[:,j+1])
            arr2[i,j] = coeff[0,1]
    arr2=pd.DataFrame(arr2)
    
    arr2 = arr2.transpose()
    
    col_names = x0.columns[2:x0.shape[1]]
    arr2=pd.DataFrame(data = arr2.values, columns=col_names)
    arr2.insert(0, "Лаги", " ")
    
    for i in range(12):
        arr2.loc[i, "Лаги"] = i+1
    arr2_max = np.array([np.max(arr2.iloc[:,1:], axis = 0)]).transpose()
    signal_to_noise = np.array(stn.iloc[1:,1:]).transpose()
    
    arr3 = np.concatenate((signal_to_noise, arr2_max), axis = 1)
    
    x = arr3[:,0]
    y = arr3[:,1]
    
    arr3=pd.DataFrame(arr3)
    col_names1 =["Отношение шум/сигнал", "Коэффициент корреляции"]
    аrr3 = pd.DataFrame(arr3)
    arr3 = pd.DataFrame(data = arr3.values, columns=col_names1)
        
    g = sns.jointplot(data=pd.DataFrame({'Отношение шум/сигнал':x, 'Коэффициент корреляции':y}), x='Отношение шум/сигнал', y='Коэффициент корреляции')
    g.ax_joint.axvline(x=0.5,linestyle='--', color = 'r')
    g.ax_joint.axhline(y=0.5,linestyle='--', color = 'r')
    #g.ax_joint.annotate(f'Зона потенциальных \n предикторов ',xy=(0.1, 0.8), xycoords='axes fraction',ha='right', va='center',fontsize=10 )
    
    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure1.png')
    plt.show(block = False)
    
    return arr2

def potential_predictors(x, y, z):
    y = np.array(y)
    y = y[:,1:]
    arr1 = y[1:,:]
    max_elements = z.max(axis=0)
    arr2 =np.array( [np.array(max_elements.iloc[1:])])
    arr = np.concatenate([arr1,arr2])
    indlist = [0 for i in range(len(arr[0]))]
    for i in range(len(arr[0])):
        if arr[0,i] <=0.5 and arr[1,i] >=0.5:
                ind = i
                indlist[i] = ind+2
    indlist_no_zeros = [x for x in indlist if x != 0]
    rng = np.arange(2, x.shape[1], 1).tolist()
    final_index= list(set(rng) - set(indlist_no_zeros))
    x = x.drop(x.columns[final_index], axis=1)
    x =  x.iloc[2:,:]
                       
    x_names = np.array([x.columns[2:]]).transpose()
    number_of_names = len(x_names)
    int_number_of_tables = int(number_of_names/15)
    number_of_tables = math.ceil(len(x_names)/15)
    #if number_of_names <=15:
    x2_names = np.array([[str for i in range(1)] for j in range(len(x_names))])
    for i in range(len(x_names)):
        x2_names[i] = "П" + str(i+1)
    x_names = np.concatenate((x_names, x2_names), axis = 1)
    x_names = pd.DataFrame(x_names)
    col_names1 =["Наименование показателя", "Обозначение"]
    set_pandas_display_options()
    table = pd.DataFrame(data = x_names.values, columns=col_names1)
    r"""
    if number_of_names > 15:
        
        table={}
        for i in range(int_number_of_tables):
            x1_names = np.array([[str for i in range(1)] for j in range(15)])
            x2_names = np.array([[str for i in range(1)] for j in range(15)])
            
            for j in range(15):
                x1_names[j] = x_names[j + 15*i]
                x2_names[j] = "П" + str((j+1)+15*i)
            x_names_new = np.concatenate((x1_names, x2_names), axis = 1)
            x_names_new = pd.DataFrame(x_names_new)
            col_names1 =["Наименование показателя", "Обозначение"]
            set_pandas_display_options()
            
            table[i+1] =  pd.DataFrame(data = x_names_new.values, columns=col_names1)
        for i in range(1):
            x1_names = np.array([[str for i in range(1)] for j in range(number_of_names - 15*int_number_of_tables)])
            x2_names = np.array([[str for i in range(1)] for j in range(number_of_names - 15*int_number_of_tables)])
            for j in range(number_of_names - 15*int_number_of_tables):
                x1_names[j] = x_names[j + 15*int_number_of_tables]
                x2_names[j] = "П" + str((j+1)+15*int_number_of_tables)
            x_names_new = np.concatenate((x1_names, x2_names), axis = 1)
            x_names_new = pd.DataFrame(x_names_new)
            col_names1 =["Наименование показателя", "Обозначение"]
            set_pandas_display_options()
            table[i+1+int_number_of_tables] =  pd.DataFrame(data = x_names_new.values, columns=col_names1)
            
    """    
    df = np.array(x)
    
    df = df[:,2:].astype(float)
    col_names = [str for i in range(len(df[0]))]
    for i in range(len(df[0])):
        col_names[i] = "П" + str(i+1)
    df = pd.DataFrame(data = df, columns = col_names)
    cor_matrix = df.corr()
    stocks = cor_matrix.index.values


    cor_matrix = np.asmatrix(cor_matrix)
    np.fill_diagonal(cor_matrix, 0)
    G = nx.from_numpy_matrix(cor_matrix)

    #relabels the nodes to match the  stocks names
    G = nx.relabel_nodes(G,lambda x: stocks[x])

    #shows the edges with their corresponding weights
    G.edges(data=True)
    fig = plt.figure()
    H = G.copy()
    corr_direction = "positive"
    min_correlation = 0.5
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] <min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    
    
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)

    #fig, ax = plt.subplots(1, 1, num=1)

    ax1 = fig.add_subplot(111)
    ax1.set_title('А. Коэффициент корреляции > 0.5',fontsize=10)
    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,node_size=tuple([x**4 for x in node_sizes]),alpha=1 )
    nx.draw_networkx_labels(H, positions, font_size=10, font_family='sans-serif' )
    if corr_direction == "positive":
        edge_colour = plt.cm.PuRd
    else:
        edge_colour = plt.cm.PuRd
        
  
    nx.draw_networkx_edges(H, positions, edgelist=edges,style='solid',width=weights, edge_color = weights, edge_cmap = edge_colour,edge_vmin = min(weights), edge_vmax=max(weights)) 
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.axis('off')
    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure2.png')
    plt.show(block = False)
    fig = plt.figure()
    H = G.copy()
    corr_direction = "negative"
    min_correlation = -0.5
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    
    
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)

    
    ax2 = fig.add_subplot(111)
    ax2.set_title('Б. Коэффициент корреляции < -0.5',fontsize=10)
    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,node_size=tuple([x**4 for x in node_sizes]),alpha=1 )
    nx.draw_networkx_labels(H, positions, font_size=10, font_family='sans-serif' )
    if corr_direction == "positive":
        edge_colour = plt.cm.PuRd
    else:
        edge_colour = plt.cm.PuRd
        

    nx.draw_networkx_edges(H, positions, edgelist=edges,style='solid',width=weights, edge_color = weights, edge_cmap = edge_colour,edge_vmin = min(weights), edge_vmax=max(weights))
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.axis('off')
    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure3.png')
    plt.show(block = False)


    correlations = df.corr()
    correlations_array = np.asarray(df.corr())

    row_linkage = hierarchy.linkage(
    distance.pdist(correlations_array), method='average')
    col_linkage = hierarchy.linkage(
    distance.pdist(correlations_array.T), method='average')
    g = sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=col_linkage)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 12)
    plt.xticks(rotation = 45)
    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure4.png')
    
    plt.show(block = False)
    
    return x, table
    
def princomp(x):
    data0 = np.array(x)
    data = data0[:,3:].astype(float)
    data = data.transpose()
    arr = np.corrcoef(data)
    w, v = LA.eig(arr)
    n = len(w)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    w = w.tolist()
    w[:] = [x for x in w if x>=1]
    v = v[:,:len(w)]
    for i in range(len(v)):
        for j in range(len(v[0])):
            v[i,j] = v[i,j]*np.sqrt(w[j])

    data = data.transpose()
    data_mean = np.mean(data, axis = 0)
    data_std = np.std(data, axis = 0)
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i,j] = (data[i,j] - data_mean[j])/data_std[j]

    comps = np.matmul(data, v)
    num_comps = len(comps[0])
    comps2 = np.concatenate((data0[:,0:2], comps), axis = 1)
    comps=pd.DataFrame(comps2)
    col_names = [str for i in range(len(v[0]))]
    for i in range(len(v[0])):
        col_names[i] = "PC"+str(i+1)

    col_names.insert(0, " ")
    col_names.insert(1, "date")
        
    comps=pd.DataFrame(data = comps.values, columns=col_names)
    names = [str for i in range(len(data[0]))]
    for i in range(len(data[0])):
        names[i] = "П" + str(i+1)
    ds = np.array([[str for i in range(2)] for j in range(num_comps*len(data[0]))])
    ds2 = np.array([[0.0 for i in range(1)] for j in range(num_comps*len(data[0]))])
    rat = v**2/w
    for i in range(num_comps):
        for j in range(len(data[0])):
            ds[j + i*len(data[0]),0] = "ГK" + " " + str(i+1)
            ds[j + i*len(data[0]),1] = names[j]
            ds2[j +i*len(data[0]),0] = rat[j,i]
    ds_whole = np.concatenate((ds,ds2), axis = 1)
    col_names1 = [str for i in range(3)]
    col_names1[0] = "PC"
    col_names1[1] = "VAR"
    col_names1[2] = "CORR"
    df = pd.DataFrame(data = ds_whole, columns=col_names1)
    sankey(
    left=df["PC"], right=df["VAR"], 
    leftWeight= df["CORR"], rightWeight=df["CORR"], 
    aspect=20, fontsize=10)
    
    # Get current figure
    fig = plt.gcf()

    # Set size in inches
    fig.set_size_inches(7, 7)
    
    # Set the color of the background to white
    fig.set_facecolor("w")
    
    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure5.png' ) 
    plt.show(block = False)
            
    

    return comps

def signal_to_noise_ratio2(df, prds, param):
    
    str_num = len(prds)
    lst = np.zeros(str_num).astype(str)
    for i in range(str_num):
        if len(prds[i]) <= 10:
            lst[i] =  df[df['date'] == prds[i]].index[0]
        else:
            lst2 = prds[i].split(':')
            p1 = df[df['date'] == lst2[0]].index[0]
            p2 = df[df['date'] == lst2[1]].index[0]
            p = str(p1)+'-'+str(p2)
            lst[i] = p
    
    f1 = lst[0].split('-')
    f2 = lst[len(lst)-1].split('-')
    if len(f1) == 1 and int(f1[0]) - 30 >=2:
        start_index = int(f1[0]) - 30
    elif len(f1) == 2 and int(f1[0]) - 30 >=2:
        start_index = int(f1[0]) - 30
    else:
        start_index = 2

    if len(f2) == 1 and int(f2[0]) + 24 <=len(df):
        end_index = int(f2[0]) + 24
    elif len(f2) == 2 and int(f2[1]) + 24 <=len(df):
        end_index = int(f2[1]) + 24    
    else:
        end_index = len(df)
   
        
    data0 = np.array(df)
    data = np.array(df)
    data = data[start_index:end_index,1].astype(float)
    avr = np.mean(data, axis = 0)
    sd = np.std(data, axis = 0)
    
    threshold = avr + param*sd

    data1 = np.array([[0 for i in range(1)] for j in range(len(data))])

    data2 = np.array([[0 for i in range(len(data))] for j in range(1)])
    data1_1 = np.array([[0 for i in range(1)] for j in range(1)])
    
    for i in range(len(data)):
        if threshold < 0 and data[i] > threshold:
            data1[i,:] = 0
        elif threshold < 0 and data[i] < threshold:
            data1[i,:] = 1

        if threshold > 0 and data[i] < threshold:
            data1[i,:] = 0
        elif threshold > 0 and data[i] > threshold:
            data1[i,:] = 1

    for i in range(str_num):
        lst1 = lst[i].split('-')
        if len(lst1) == 1:
            ind = int(lst[i])
            data1_1 = np.concatenate((data1_1, data1[ind-start_index-6:ind-start_index,:]), axis = 0)
            
        if len(lst1) == 2:
            p = lst[i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            data1_1 = np.concatenate((data1_1, data1[p1-start_index-6:p1-start_index,:]), axis = 0)
    data1_1 = np.delete(data1_1, 0, 0)
    data1_1 = np.sum(data1_1.reshape(int(len(data1_1)/6),6), axis = 1)
    for i in range(len(data1_1)):
        if data1_1[i] == 0:
            data1_1[i] = 1
        else:
            data1_1[i] = 0
    
    for i in range(str_num):
        lst1 = lst[str_num-1-i].split('-')
        if len(lst1) == 1:
            ind = int(lst[str_num-1-i])
            data1 = np.delete(data1, np.s_[ind-start_index-6:ind-start_index+1], axis = 0)
        if len(lst1) == 2:
            p = lst[str_num-1-i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            data1 = np.delete(data1, np.s_[p1-start_index-6:p2-start_index], axis = 0)



    fn = np.sum(data1_1)
    tp = str_num - fn
    fp = np.sum(data1)
    tn = end_index - start_index
    StN = ((fp/(fp+tn))/(tp/(tp+fn)))

    return threshold, StN, start_index, end_index, lst

def optimal_StN2(df, prds, bounds):
    nc = df.shape[1]
    f = np.linspace(bounds[0], bounds[1], 2000, endpoint=True)
    arr = np.array([0.0 for i in range(len(f))])
    arr2 = np.array([0.0 for i in range(len(f))])
    results = np.array([[0.0 for i in range(len(f))] for j in range(2)])
    results2 = np.array([[0.0 for i in range(nc-2)] for j in range(2)])
    
    for i in range(2, nc):
        results = np.array([[0.0 for ii in range(len(f))] for jj in range(2)])
        for j in range(len(f)):
            res = signal_to_noise_ratio2(df.iloc[:,[1,i]], prds, f[j])
            results[0, j] = res[0]
            results[1, j] = res[1]
        results = results[:,(results[1,:] > 0.01)]
        f1 = np.amin(results[1,:])
        min_index = np.argmin(results[1,:],axis=0)
        f2 = results[0, min_index]
        results2[0, i - 2] = f2
        results2[1, i - 2] = f1
    start_index = res[2]
    end_index = res[3]
    df1 = df.iloc[start_index:,:]
    df1 = np.array(df1)

    for i in range(2,len(df1[0])):
        for j in range(len(df1)):
            if results2[0, i-2] < 0:
                if df1[j,i] < results2[0, i-2]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0
        for j in range(len(df1)):
            if results2[0, i-2] > 0:
                if df1[j,i] > results2[0, i-2]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0

    df1 = pd.DataFrame(df1)
    col_names = [str for i in range(df1.shape[1]-2)]
    for i in range(2, df1.shape[1]):
        col_names[i-2] = "PC"+str(i-2+1)

    col_names.insert(0, " ")
    col_names.insert(1, "date")
        
    df1=pd.DataFrame(data = df1.values, columns=col_names)
    comps2 = np.array(df)
    comps2_rows = len(comps2)
    num_comps = len(results2[0])
    thrs_pc = np.array([[0.0 for i in range(num_comps)] for j in range(comps2_rows)])
    for i in range(comps2_rows):
        for j in range(num_comps):
            thrs_pc[i,j] = results2[0,j]
    comps2_whole = np.concatenate((comps2, thrs_pc), axis = 1)
    
    results2=pd.DataFrame(data = results2)
    
    results2.insert(0, "Показатели", " ")
    results2.loc[0, "Показатели"] = "Пороговый уровень"
    results2.loc[1, "Показатели"] = "Сигнал/Шум отношение"
    
    
    
    
    num_figs = math.ceil(num_comps/4)
    
    if num_figs == 1:
        fig =plt.figure()
        for i in range(num_figs):
            for j in range(num_comps):
                ax = fig.add_subplot(2,2,j+1)
                ax.plot(comps2_whole[:,1], comps2_whole[:,j+2],linewidth=2 ,color="black")
                ax.plot(comps2_whole[:,1], comps2_whole[:,j+2+num_comps],linewidth=2 ,color="red",linestyle='dashed' )
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                ax.margins(0)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.xticks(rotation=45)
                ax.set_title("Главная компонента" + "" + str(j+1), fontdict={'size':8})
                plt.grid(color='b', linestyle='--', linewidth=0.25)
                date_start = str(comps2_whole[0,1])
                date_end = str(comps2_whole[len(comps2_whole)-1, 1])
                if thrs_pc[0,j] > 0:
                    plt.fill_between([date_start, date_end], [np.max(comps2_whole[:,j+2]), np.max(comps2_whole[:,j+2])], [np.max(comps2_whole[:,j+2+num_comps]), np.max(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')
                else:
                    plt.fill_between([date_start, date_end], [np.min(comps2_whole[:,j+2]), np.min(comps2_whole[:,j+2])], [np.min(comps2_whole[:,j+2+num_comps]), np.min(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')    
                plt.gca().spines["top"].set_alpha(.0)
                plt.gca().spines["right"].set_alpha(.0)
                plt.subplots_adjust(hspace = 0.50)
                ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
                ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
                ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)
        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure6.png')
        plt.show(block = False)

    if num_figs > 1:
        intpart = int(num_comps/4)
        for i in range(intpart):
            fig = plt.figure()
            for j in range(4):
                ax = fig.add_subplot(2,2,j+1)
                ax.plot(comps2_whole[:,1], comps2[:,j+2],linewidth=2 ,color="black")
                ax.plot(comps2_whole[:,1], comps2_whole[:,j+2+num_comps],linewidth=2 ,color="red",linestyle='dashed' )
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                ax.margins(0)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.xticks(rotation=45)
                ax.set_title("Главная компонента" + "" + str(j+1), fontdict={'size':8})
                plt.grid(color='b', linestyle='--', linewidth=0.25)
                date_start = str(comps2_whole[0,1])
                date_end = str(comps2_whole[len(comps2_whole)-1, 1])
                if thrs_pc[0,j] > 0:
                    plt.fill_between([date_start, date_end], [np.max(comps2_whole[:,j+2]), np.max(comps2_whole[:,j+2])], [np.max(comps2_whole[:,j+2+num_comps]), np.max(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')
                else:
                    plt.fill_between([date_start, date_end], [np.min(comps2_whole[:,j+2]), np.min(comps2_whole[:,j+2])], [np.min(comps2_whole[:,j+2+num_comps]), np.min(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')    
                plt.gca().spines["top"].set_alpha(.0)
                plt.gca().spines["right"].set_alpha(.0)
                plt.subplots_adjust(hspace = 0.50)
                ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
                ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
                ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)
        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+i+1)+'.png')
        plt.show(block = False)

        
        fig = plt.figure()
        for i in range(num_comps - intpart*4):
            ax = fig.add_subplot(2,2,i+1)
            ax.plot(comps2_whole[:,1], comps2_whole[:,i+intpart*4+2],linewidth=2 ,color="black")
            ax.plot(comps2_whole[:,1], comps2_whole[:,i+2+intpart*4 + num_comps],linewidth=2 ,color="red",linestyle='dashed' )
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            ax.margins(0)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.xticks(rotation=45)
            ax.set_title("Главная компонента" + "" + str(intpart*4+i+1), fontdict={'size':8})
            plt.grid(color='b', linestyle='--', linewidth=0.25)
            date_start = str(comps2_whole[0,1])
            date_end = str(comps2_whole[len(comps2_whole)-1, 1])
            if thrs_pc[0,j] > 0:
                plt.fill_between([date_start, date_end], [np.max(comps2_whole[:,j+2]), np.max(comps2_whole[:,j+2])], [np.max(comps2_whole[:,j+2+num_comps]), np.max(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')
            else:
                plt.fill_between([date_start, date_end], [np.min(comps2_whole[:,j+2]), np.min(comps2_whole[:,j+2])], [np.min(comps2_whole[:,j+2+num_comps]), np.min(comps2_whole[:,j+2+num_comps])], hatch='/', facecolor='w')
            plt.gca().spines["top"].set_alpha(.0)
            plt.gca().spines["right"].set_alpha(.0)
            plt.subplots_adjust(hspace = 0.50)
            ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
            ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
            ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)

            plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+intpart + 1)+'.png')
            plt.show(block = False)
    return results2, df1, res[2], res[4], num_figs

def compind(x, y):
    x = np.array(x)
    coef = x[1,1:]
    coef = 1/coef
    y1 = np.array(y)
    y2 = y1[:,2:]
    compind = (np.matmul(y2,coef))/coef.sum()
    compind = (np.array([compind])).transpose()
    compind = np.concatenate((y1[:,0:2], compind), axis = 1)
    
    col_names = [str for i in range(1)]
    col_names.insert(0, " ")
    col_names.insert(1, "date")
    col_names[2] = "Composite Indicator"
    compind=pd.DataFrame(data = compind, columns=col_names)
    return compind

def threshold(df, prds, num_figs):
    x = np.array(df)
    x_rows = len(x)
    x1 = np.array([x[:,2]])
    x1 = x1.transpose()

    str_num = len(prds)
    lst = np.zeros(str_num).astype(str)
    for i in range(str_num):
        if len(prds[i]) <= 10:
            lst[i] =  df[df['date'] == prds[i]].index[0]
        else:
            lst2 = prds[i].split(':')
            p1 = df[df['date'] == lst2[0]].index[0]
            p2 = df[df['date'] == lst2[1]].index[0]
            p = str(p1)+'-'+str(p2)
            lst[i] = p
    f1 = lst[0].split('-')
    f2 = lst[len(lst)-1].split('-')
    if len(f1) == 1 and int(f1[0]) - 30 >=0:
        start_index = int(f1[0]) - 30
    elif len(f1) == 2 and int(f1[0]) - 30 >=0:
        start_index = int(f1[0]) - 30
    else:
        start_index = 0

    if len(f2) == 1 and int(f2[0]) + 24 <=len(df):
        end_index = int(f2[0]) + 24
    elif len(f2) == 2 and int(f2[1]) + 24 <=len(df):
        end_index = int(f2[1]) + 24    
    else:
        end_index = len(df)

    data1_1 = np.array([[0 for i in range(1)] for j in range(1)])   
    for i in range(str_num):
        lst1 = lst[i].split('-')
        if len(lst1) == 1:
            ind = int(lst[i])
            data1_1 = np.concatenate((data1_1, x1[ind-start_index-6:ind-start_index,:]), axis = 0)
            
        if len(lst1) == 2:
            p = lst[i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            data1_1 = np.concatenate((data1_1, x1[p1-start_index-6:p1-start_index,:]), axis = 0)
    data1_1 = np.delete(data1_1, 0, 0)
    for i in range(str_num):
        lst1 = lst[str_num-1-i].split('-')
        if len(lst1) == 1:
            ind = int(lst[str_num-1-i])
            x1 = np.delete(x1, np.s_[ind-start_index-6:ind-start_index+1], axis = 0)
        if len(lst1) == 2:
            p = lst[str_num-1-i].split('-')
            p1 = int(p[0])
            p2 = int(p[1])
            x1 = np.delete(x1, np.s_[p1-start_index-6:p2-start_index], axis = 0)

    x2 = x[:,2].tolist()
    bn = []
    for i in x2:
        if i not in bn:
            bn.append(i)
    
    bn.sort()
    num_bn = len(bn)
    r1 = [0 for i in range(num_bn)] 
    r2 = [0 for i in range(num_bn)]
    r1_1 = [0 for i in range(num_bn-1)] 
    r2_1 = [0 for i in range(num_bn-1)]
    prob = [0 for i in range(num_bn)]
    for i in range(num_bn - 1):
        r1[i+1] = np.count_nonzero(data1_1 <= bn[i+1]) 
        r2[i+1] = np.count_nonzero(x1 <= bn[i+1]) 
        

    r1_1 = np.diff(r1)
    r2_1 = np.diff(r2)
    r = r1_1 + r2_1
    

    for i in range(num_bn - 1):
        
        prob[i+1] = r1_1[i]/(r1_1[i]+r2_1[i])
    
    prob = [round(item,3) for item in prob]
    prob1 = [x for x in prob if x != 1.0 ]
    max_value = max(prob1)
    max_index = prob.index(max_value)
    thrs0 = bn[max_index-1]
    prob2 = np.array([prob]).transpose()
    col_names = []
    bn2 = np.round(bn,3)
    for i in range(len(prob)-1):
        col_names.insert(i+1, "<=" + str(bn2[i+1]))

    col_names.insert(0, "")
    col_names2 = np.array([col_names]).transpose()
    cols = [str for i in range(3)]
    cols[0] = "Интервал"
    cols[1] = "Вероятность"
    cols[2] = "Порог"
    
    ds = np.concatenate((col_names2, prob2), axis = 1)
    tr = np.array([np.zeros(len(ds))]).transpose()
    tr[max_index-1] = round(thrs0,3)
    ds = np.concatenate((ds, tr), axis = 1)
    ds = ds[1:,:]
    
    #prob[0] = "Вероятность"
    #prob = np.array([prob]).transpose()
    #result=pd.DataFrame(data = np.array(prob),columns=col_names )
    result=pd.DataFrame(data = ds,columns=cols)

    thrs = ["Пороговый уровень", thrs0]
    thrs = np.array([thrs])
    result1=pd.DataFrame(data = np.array(thrs))
    table = PrettyTable()
    table.field_names = result.columns.to_list()
    table.add_row(list(result.iloc[0,:]))


    table2 = PrettyTable()
    table2.field_names = ["Показатель", "Величина"]
    table2.add_row(list(result1.iloc[0,:]))
    thrs_arr = np.array([[0.0 for i in range(1)] for j in range(x_rows)])
    for i in range(x_rows):
        thrs_arr[i,:] = thrs0
    x_whole = np.concatenate((x,thrs_arr), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_whole[:,1], x_whole[:,2],linewidth=2 ,color="black", label='Композитный индикатор' )
    ax.plot(x_whole[:,1], x_whole[:,3],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.margins(0)
    ax.set_yticks(np.arange(0, 1, 0.99999))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    years = mdates.YearLocator(base = 1)
    ax.xaxis.set_major_locator(years)
    plt.xticks(rotation=45)
    
    ax.set_ylim(ymin=0, ymax = 1)
    #ax.set_title("Сигнальный подход", fontdict={'size':12})
    ax.get_xaxis().tick_bottom()
    plt.xticks(fontsize=6)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    ax.legend();
    ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)

    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure'+str(5+num_figs + 1)+'.png')
    plt.show(block = False)
    return result, result1, table, table2
    
def ar(y,nlag, *args):
    varargin = args
    nargin = 2 + len(varargin)
    n = len(y)
    if nargin == 2:
        const = 1
        x = np.hstack((np.ones((n,1)), mlag(y,nlag)))
    else:
        x = mlag(y,nlag)

    x = trimr(x,nlag,0)
    y = y.reshape((-1, 1))
    y = trimr(y,nlag,0)
    nadj = len(y)
    b0 = np.matmul(np.matmul(LA.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)
    yhat = np.matmul(x,b0)
    resid = np.subtract(y,yhat)
        
    return b0, yhat, resid

def mlag(x,*args):
    varargin = args
    nargin = 1+len(varargin)
    if nargin == 1:
        n = 1
        init = 0
    elif nargin == 2:
        n=args[0]
        init =0
    nobs = len(x)
    #nvar = len(x[0])
    xlag = np.ones((nobs,n))*init
    icnt = 0
    for i in range(1):
        for j in range(n):
            xlag[list(range(j+1, nobs)), icnt +j] = x[list(range(0, nobs - (j+1)))]
       # icnt = icnt+n
    nr = len(xlag)
    nc = len(xlag[0])
    return xlag

def trimr(x,n1,n2):
    n = len(x)
    #junk = len(x[0])
    if (n1+n2) >= n:
        raise ValueError("a should be nonzero")
    h1 = n1
    h2 = n-n2
    z = x[h1:h2,:]
    return z
    
def arf(y, nlag, nper):
    r = len(y)
    mat0 = np.array([[0.0 for i in range(1)] for j in range(nlag+nper)])
    mat1 = np.array([[0.0 for i in range(1)] for j in range(nlag)])
    beta = np.array([[0.0 for i in range(1)] for j in range(nlag+1)])
    betaf = np.array([[0.0 for i in range(1)] for j in range(nlag)])
    mat0f = np.array([[0.0 for i in range(1)] for j in range(nper)])

    for i in range(nlag):
        for j in range(1):
            mat0[i,j] = y[r-(nlag-i)]

    beta, yhat, resid = ar(y,nlag)
    for i in range(nlag):
        for j in range(1):
            betaf[i,j] = beta[i+1,j]

    betaf_tr = np.transpose(betaf)
    for i in range(nper):
        for j in range(nlag):
            for l in range(1):
                for ll in range(1):
                    mat1[l+j,0] = mat0[i+nlag-j-1,l]
                    mat2 = np.matmul(betaf_tr, mat1)
                    mat0[i+nlag,ll] = np.add(beta[0,ll],mat2[ll,0])

    for i in range(nper):
        for j in range(1):
            mat0f[i,j] = mat0[nlag+i,j]
            
    return mat0f
    
def fcast_pc(df, nlag, nper, num_iter):
    x0 = np.array(df)
    x = x0[:,2:]
    x1 = x[:nlag,:]
    boot_fcast = np.array([[[0.0 for col in range(num_iter)]for row in range(nper)] for k in range(len(x[0]))])
    avr = np.array([[0.0 for col in range(1)]for row in range(nper)])
    x_add = np.array([[0.0 for col in range(len(x[0]))]for row in range(nper)])
    minmax = np.array([[0.0 for col in range(2)]for row in range(nper)])
    bn = np.array([[0.0 for col in range(1)]for row in range(nper)])
    ivals = np.array([[[0.0 for col in range(8)]for row in range(nper)] for k in range(len(x[0]))])
    part1 = np.array([[[0.0 for col in range(8)]for row in range(len(x))] for k in range(len(x[0]))])
    whole_ds = np.array([[[0.0 for col in range(8)]for row in range(len(x) + nper)] for k in range(len(x[0]))])
    ds = np.array([[[str for col in range(9)]for row in range(len(x) + nper)] for k in range(len(x[0]))])
    for i in range(len(x[0])):
        beta, yhat, resid = ar(x[:,i], nlag)
        yhat = yhat.flatten()
        resid = resid.flatten()
        for j in range(num_iter):
            resid1 = np.random.choice(resid, replace = True, size = len(resid))
            yb = np.add(yhat,resid1)
            yb = np.concatenate((x1[:,i], yb), axis = 0)
            fcast = arf(yb,nlag,nper)
            for k in range(nper):
                boot_fcast[i,k,j] = fcast[k,0]
                avr = np.mean(boot_fcast[i,:,:], axis = 1)
                x_add[k,i] = avr[k]

    for i in range(len(x[0])):
        for j in range(nper):
            minmax[j,0] = min(boot_fcast[i,j,:])
            minmax[j,1] = max(boot_fcast[i,j,:])
            bn[j,0] = (minmax[j,1] - minmax[j,0])/7
            ivals[i,j,0] = minmax[j,0]
            for k in range(7):
                ivals[i,j,k+1] = ivals[i,j,k] + bn[j,0]
    
    for i in range(len(x[0])):
        for j in range(len(x)):
            for k in range(8):
                part1[i,j,k] = x[j, i]

    
    for i in range(len(x[0])):
        whole_ds[i,:,:] = np.concatenate((part1[i,:,:], ivals[i,:,:]), axis = 0)
    x_new = np.concatenate((x, x_add), axis = 0)
    time_arr = np.array([str for i in range(nper)])
    dt = x0[len(x0)- 1,1]
    for i in range(nper):
        dt = dt+ relativedelta(months=1)
        time_arr[i] = dt
    whole_time = np.concatenate((x0[:,1], time_arr), axis = 0)
    whole_time = (np.array([whole_time])).transpose()
    for i in range(len(x[0])):
        ds[i,:,:] = np.concatenate((whole_time, whole_ds[i,:,:]), axis = 1)
    ds2 = ds[:,len(ds[0])-nper-7:len(ds[0]),:]    
    num_figs = math.ceil(len(x[0])/4)
    whole_data = np.concatenate((whole_time, x_new), axis = 1)
    
    if num_figs == 1:
        fig =plt.figure()
        for i in range(num_figs):
            for j in range(len(x[0])):
                ax = fig.add_subplot(2,2,j+1)
                plt.subplots_adjust(hspace = 0.50)
                for k in range(8):
                    ax.plot(ds2[j,:,0], ds2[j,:,k+1], color ='red', alpha = 0.01)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,1].astype(float),ds2[j,:,2].astype(float), color = 'red', alpha = 0.01)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,2].astype(float),ds2[j,:,3].astype(float), color = 'red', alpha = 0.05)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,3].astype(float),ds2[j,:,4].astype(float), color = 'red', alpha = 0.1)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,4].astype(float),ds2[j,:,5].astype(float), color = 'red', alpha = 0.3)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,5].astype(float),ds2[j,:,6].astype(float), color = 'red', alpha = 0.1)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,6].astype(float),ds2[j,:,7].astype(float), color = 'red', alpha = 0.05)
                    ax.fill_between(ds2[j,:,0], ds2[j,:,7].astype(float),ds2[j,:,8].astype(float), color = 'red', alpha = 0.01)
                    plt.xticks(fontsize=7)
                    plt.yticks(fontsize=7)
                    ax.margins(0)
                    plt.xticks(rotation = 45)
                    plt.grid(color='black', linestyle='-.', linewidth=0.25, alpha = 0.3)
                    plt.gca().spines["top"].set_alpha(.0)
                    plt.gca().spines["right"].set_alpha(.0)
                    date_start = str(time_arr[0])
                    date_start2 = date_start.split("-")
                    dats = str(int(date_start2[1]) - 1)
                    date_start2[1] = dats
                    date_start = "-".join(date_start2)
                    date_end = str(time_arr[len(time_arr)-1])
                    ax.axvspan(*mdates.datestr2num([date_start, date_end]), color='gray', alpha=0.05)
                    



        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+num_figs + 4)+'.png')
        plt.show(block = False)
    if num_figs > 1:
        intpart = int(len(x[0])/4)
        for i in range(intpart):
            fig = plt.figure()
            for j in range(4):
                ax = fig.add_subplot(2,2,j+1)
                plt.subplots_adjust(hspace = 0.50)
                for k in range(8):
                    ax.plot(ds2[j,:,0], ds2[j,:,k+1], color ='red', alpha = 0.01)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,1].astype(float),ds2[j+4*i,:,2].astype(float), color = 'red', alpha = 0.01)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,2].astype(float),ds2[j+4*i,:,3].astype(float), color = 'red', alpha = 0.05)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,3].astype(float),ds2[j+4*i,:,4].astype(float), color = 'red', alpha = 0.1)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,4].astype(float),ds2[j+4*i,:,5].astype(float), color = 'red', alpha = 0.3)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,5].astype(float),ds2[j+4*i,:,6].astype(float), color = 'red', alpha = 0.1)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,6].astype(float),ds2[j+4*i,:,7].astype(float), color = 'red', alpha = 0.05)
                    ax.fill_between(ds2[j+4*i,:,0], ds2[j+4*i,:,7].astype(float),ds2[j+4*i,:,8].astype(float), color = 'red', alpha = 0.01)
                    plt.xticks(fontsize=7)
                    plt.yticks(fontsize=7)
                    plt.margins(0)
                    plt.xticks(rotation = 45)
                    plt.grid(color='black', linestyle='-.', linewidth=0.25, alpha = 0.3)
                    plt.gca().spines["top"].set_alpha(.0)
                    plt.gca().spines["right"].set_alpha(.0)
                    date_start = str(time_arr[0])
                    date_start2 = date_start.split("-")
                    dats = str(int(date_start2[1]) - 1)
                    date_start2[1] = dats
                    date_start = "-".join(date_start2)
                    date_end = str(time_arr[len(time_arr)-1])
                    ax.axvspan(*mdates.datestr2num([date_start, date_end]), color='gray', alpha=0.05)

                    
        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+num_figs+i+2+num_figs)+'.png')
        plt.show(block = False)

        
        fig = plt.figure()
        for j in range(len(x[0]) - intpart*4):
            ax = fig.add_subplot(2,2,j+1)
            plt.subplots_adjust(hspace = 0.50)
            for k in range(8):
                ax.plot(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,k+1], color ='red', alpha = 0.01)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,1].astype(float),ds2[j+4*intpart,:,2].astype(float), color = 'red', alpha = 0.01)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,2].astype(float),ds2[j+4*intpart,:,3].astype(float), color = 'red', alpha = 0.05)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,3].astype(float),ds2[j+4*intpart,:,4].astype(float), color = 'red', alpha = 0.1)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,4].astype(float),ds2[j+4*intpart,:,5].astype(float), color = 'red', alpha = 0.3)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,5].astype(float),ds2[j+4*intpart,:,6].astype(float), color = 'red', alpha = 0.1)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,6].astype(float),ds2[j+4*intpart,:,7].astype(float), color = 'red', alpha = 0.05)
                ax.fill_between(ds2[j+4*intpart,:,0], ds2[j+4*intpart,:,7].astype(float),ds2[j+4*intpart,:,8].astype(float), color = 'red', alpha = 0.01)
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                plt.margins(0)
                plt.xticks(rotation = 45)
                plt.grid(color='black', linestyle='-.', linewidth=0.25, alpha = 0.3)
                plt.gca().spines["top"].set_alpha(.0)
                plt.gca().spines["right"].set_alpha(.0)
                date_start = str(time_arr[0])
                date_start2 = date_start.split("-")
                dats = str(int(date_start2[1]) - 1)
                date_start2[1] = dats
                date_start = "-".join(date_start2)
                date_end = str(time_arr[len(time_arr)-1])
                ax.axvspan(*mdates.datestr2num([date_start, date_end]), color='gray', alpha=0.05)

        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+num_figs+intpart + 2+num_figs)+'.png')
        plt.show(block = False)

    
    whole_data = pd.DataFrame(whole_data)
    col_names = [str for i in range(whole_data.shape[1]-1)]
    for i in range(1, whole_data.shape[1]):
        col_names[i-1] = "PC"+str(i)
    col_names.insert(0, "date")

    whole_data=pd.DataFrame(data = whole_data.values, columns=col_names)
    return whole_data
           
def fcast_signals(df, thrs):
    
    df1 = np.array(df)
    thrs = np.array(thrs.iloc[:,1:])
    for i in range(1,len(df1[0])):
        for j in range(len(df1)):
            if thrs[0, i-1] < 0:
                if df1[j,i] < thrs[0, i-1]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0
        for j in range(len(df1)):
            if thrs[0, i-1] > 0:
                if df1[j,i] > thrs[0, i-1]:
                    df1[j,i] = 1
                else:
                    df1[j,i] = 0

    df1 = pd.DataFrame(df1)
    col_names = [str for i in range(df1.shape[1]-1)]
    for i in range(1, df1.shape[1]):
        col_names[i-1] = "PC"+str(i)

    
    col_names.insert(0, "date")
        
    df1=pd.DataFrame(data = df1.values, columns=col_names)

    return df1

def logit(prds, df):
    x0 = np.array(df)
    x = x0[:,2:]
    nr = len(x)
    str_num = prds
    y = np.array([[0 for i in range(1)] for j in range(nr)])
    for i in range(len(prds)):
        if len(str_num[i].split('-')) == 1:
            y[int(str_num[i])-2, :] = 1
        elif len(str_num[i].split('-')) == 2:
            f = str_num[i].split('-')
            f1 = int(f[0])
            f2 = int(f[1])
            y[f1-2:f2-2, :] = 1
    ones_arr = np.array([[0.0 for col in range(1)]for row in range(len(x))])
    for i in range(nr):
        ones_arr[i,0] = 1
    x = np.concatenate((ones_arr, x), axis = 1)
    
    b = np.matmul(np.matmul((LA.inv(np.matmul(x.transpose(),x).astype(float))), np.transpose(x)), y).astype(float)
    t = len(x)
    k = len(x[0])
    
    tol = 0.000001
    maxit = 100
    crit = 1.0
    i = np.ones((t,1))
    tmp1 = np.zeros((t,k)).astype(float)
    tmp2 = np.zeros((t,k)).astype(float)
    itr = 1
  
    while itr < maxit and crit > tol:
        tmp = (i+np.exp(-np.matmul(x,b).astype(float)))
        pdf = np.exp(-np.matmul(x,b).astype(float))/np.multiply(tmp,tmp)
        cdf = i/(i+np.exp(-np.matmul(x,b).astype(float)))
        tmp = np.nonzero(cdf <=0)[0]
        n1 = len(tmp)
        if n1 != 0:
               cdf[tmp] = 0.00001
        
        tmp = np.nonzero(cdf >= 1)[0]
        n1 = len(tmp)
        if n1 != 0:
               cdf[tmp] = 0.99999
        term1 = np.multiply(y,np.divide(pdf,cdf))
        term2 = np.multiply(np.subtract(i,y), np.divide(pdf, np.subtract(i,cdf)))
        for kk in range(k):
               A = np.multiply(term1, np.array([x[:,kk]]).astype(float).transpose())
               B = np.multiply(term2, np.array([x[:,kk]]).astype(float).transpose())
               for tt in range(t):
                   tmp1[tt,kk] = A[tt,0]
                   tmp2[tt,kk] = B[tt,0]


        g = tmp1-tmp2
        gs = np.transpose(np.sum((g), axis = 0))
        delta = np.divide(np.exp(np.matmul(x,b).astype(float)), np.add(i, np.exp(np.matmul(x,b).astype(float))))
        H = np.zeros((k,k))
        for ii in range(t):
            xp = np.array([np.transpose(x[ii,:])]).transpose().astype(float)
            H = H - delta[ii,0]*(1-delta[ii,0])*(np.matmul(xp,np.array([x[ii,:]]).astype(float)))
        db = -np.array([np.matmul(LA.inv(H),gs)]).transpose()
        s = 2
        term1 = 0
        term2 = 1
        while term2 > term1:
            s = s/2
            term1 = lo_like(b+s*db,y,x)
            term2 = lo_like(b+s*db/2,y,x)

        bn = b + s*db
        crit = abs(max(max(db)))
        b = bn
        itr = itr + 1


    delta = np.divide(np.exp(np.matmul(x,b).astype(float)), np.add(i, np.exp(np.matmul(x,b).astype(float))))
    H = np.zeros((k,k))
    for i in range(t):
        xp = np.array([np.transpose(x[i,:])]).transpose().astype(float)
        H = H - delta[i,0]*(1-delta[i,0])*(np.matmul(xp,np.array([x[i,:]]).astype(float)))
    covb = -LA.inv(H)
    stdb = np.array([np.sqrt(np.diag(covb))]).transpose()
    tstat = b/stdb
    i = np.ones((t,1))
    prfit = np.ones((t,1))/np.add(i, np.exp(-np.matmul(x,b).astype(float)))
    cols = [str for i in range(4)]
    cols[0] = " "
    cols[1] = "Coef."
    cols[2] = "Stdev"
    cols[3] = "t-Stat."
    nms = df.columns[2:].to_list()
    nms.insert(0,'Const.')
    nms = np.array([nms]).transpose()
    b = np.round(b,3)
    stdb = np.round(stdb, 3)
    tstat = np.round(tstat, 3)
    
    tabs = np.concatenate((nms, b), axis = 1)
    tabs = np.concatenate((tabs,stdb), axis = 1)
    tabs = np.concatenate((tabs, tstat), axis = 1)
    table = pd.DataFrame(data = tabs, columns = cols)
    #ds = np.concatenate((y,x), axis = 1)
    x1 = np.array([x0[:,1]]).transpose()
    prfit = np.concatenate((x1, prfit), axis = 1)
    col_names = []
    col_names.insert(0, "date")
    col_names.insert(1, "X")
    prfit = pd.DataFrame(data = prfit, columns = col_names)
    
    
    return b, prfit, table

def lo_like(b,y,x):
    i = np.ones((len(y),1))
    cdf = i/(i+np.exp(-np.matmul(x,b).astype(float)))
    tmp = np.nonzero(cdf <=0)[0]
    n1 = len(tmp)
    if n1 != 0:
        cdf[tmp] = 0.00001
        
    tmp = np.nonzero(cdf >= 1)[0]
    n1 = len(tmp)
    if n1 != 0:
        cdf[tmp] = 0.99999

    like = np.multiply(y, np.log(cdf)) + np.multiply((i-y), np.log(i-cdf))
    result = np.sum(like)
    return result

def probit(prds, df):
    x0 = np.array(df)
    x = x0[:,2:]
    nr = len(x)
    str_num = prds
    y = np.array([[0 for i in range(1)] for j in range(nr)])
    for i in range(len(prds)):
        if len(str_num[i].split('-')) == 1:
            y[int(str_num[i])-2, :] = 1
        elif len(str_num[i].split('-')) == 2:
            f = str_num[i].split('-')
            f1 = int(f[0])
            f2 = int(f[1])
            y[f1-2:f2-2, :] = 1
    ones_arr = np.array([[0.0 for col in range(1)]for row in range(len(x))])
    for i in range(nr):
        ones_arr[i,0] = 1
    x = np.concatenate((ones_arr, x), axis = 1)
    
    b = np.matmul(np.matmul((LA.inv(np.matmul(x.transpose(),x).astype(float))), np.transpose(x)), y).astype(float)
    
    t = len(x)
    k = len(x[0])
    tol = 0.000001
    maxit = 100
    crit = 1.0
    i = np.ones((t,1))
    tmp1 = np.zeros((t,k)).astype(float)
    tmp2 = np.zeros((t,k)).astype(float)
    itr = 1
    while itr < maxit and crit > tol:
        pdf = norm.pdf(np.matmul(x,b).astype(float))
        cdf = norm.cdf(np.matmul(x,b).astype(float))
        tmp = np.nonzero(cdf <=0)[0]
        n1 = len(tmp)
        if n1 != 0:
               cdf[tmp] = 0.00001*np.ones(len(tmp))
        
        tmp = np.nonzero(cdf >= 1)[0]
        n1 = len(tmp)
        if n1 != 0:
               cdf[tmp] = 0.99999*np.ones(len(tmp))
        term1 = np.multiply(y,np.divide(pdf,cdf))
        term2 = np.multiply(np.subtract(i,y), np.divide(pdf, np.subtract(i,cdf)))
        
        for kk in range(k):
            A = np.multiply(term1, np.array([x[:,kk]]).astype(float).transpose())
            B = np.multiply(term2, np.array([x[:,kk]]).astype(float).transpose())
            for tt in range(t):
                tmp1[tt,kk] = A[tt,0]
                tmp2[tt,kk] = B[tt,0]
        g = tmp1-tmp2
        gs = np.transpose(np.sum((g), axis = 0))
        q =2*y-i
        xxb = np.matmul(x,b)
        pdf = norm.pdf(np.multiply(q,xxb).astype(float))
        cdf = norm.cdf(np.multiply(q,xxb).astype(float))
        lam = np.divide(np.multiply(q,pdf), cdf)
        
        
        H = np.zeros((k,k))
        for ii in range(t):
            xb = np.matmul(np.array([np.transpose(x[ii,:])]).astype(float), b)
            xp = np.array([np.transpose(x[ii,:])]).transpose().astype(float)
            H = H - lam[ii,0]*(lam[ii,0] +xb)*(np.matmul(xp,np.array([x[ii,:]]).astype(float))) 
        db = -np.array([np.matmul(LA.inv(H),gs)]).transpose()
        s = 2
        term1 = 0
        term2 = 1
        while term2 > term1:
            s = s/2
            term1 = pr_like(b+s*db,y,x)
            term2 = pr_like(b+s*db/2,y,x)

        bn = b + s*db
        crit = abs(max(max(db)))
        b = bn
        itr = itr + 1
    q = 2*y - i
    xxb = np.matmul(x,b)
    pdf = norm.pdf(np.multiply(q,xxb).astype(float))
    cdf = norm.cdf(np.multiply(q,xxb).astype(float))
    lam = np.divide(np.multiply(q,pdf), cdf)
    H = np.zeros((k,k))
    for i in range(t):
        xb = np.matmul(np.array([np.transpose(x[i,:])]).astype(float), b)
        xp = np.array([np.transpose(x[i,:])]).transpose().astype(float)
        H = H - lam[i,0]*(lam[i,0] +xb)*(np.matmul(xp,np.array([x[i,:]]).astype(float))) 
    covb = -LA.inv(H)
    stdb = np.array([np.sqrt(np.diag(covb))]).transpose()
    tstat = b/stdb
    i = np.ones((t,1))
    prfit = norm.cdf(np.matmul(x,b).astype(float))
    cols = [str for i in range(4)]
    cols[0] = " "
    cols[1] = "Coef."
    cols[2] = "Stdev"
    cols[3] = "t-Stat."
    nms = df.columns[2:].to_list()
    nms.insert(0,'Const.')
    nms = np.array([nms]).transpose()
    b = np.round(b,3)
    stdb = np.round(stdb, 3)
    tstat = np.round(tstat, 3)
    
    tabs = np.concatenate((nms, b), axis = 1)
    tabs = np.concatenate((tabs,stdb), axis = 1)
    tabs = np.concatenate((tabs, tstat), axis = 1)
    table = pd.DataFrame(data = tabs, columns = cols)

    x1 = np.array([x0[:,1]]).transpose()
    prfit = np.concatenate((x1, prfit), axis = 1)
    col_names = []
    col_names.insert(0, "date")
    col_names.insert(1, "X")
    prfit = pd.DataFrame(data = prfit, columns = col_names)
    
    return b, prfit, table

def pr_like(b,y,x):
    i = np.ones((len(y),1))
    cdf = norm.cdf(np.matmul(x,b).astype(float))
    
    tmp = np.nonzero(cdf <=0)[0]
    n1 = len(tmp)
    if n1 != 0:
        cdf[tmp] = 0.00001*np.ones(len(tmp))
        
    tmp = np.nonzero(cdf >= 1)[0]
    n1 = len(tmp)
    if n1 != 0:
        cdf[tmp] = 0.99999*np.ones(len(tmp))

    out = np.multiply(y, np.log(cdf)) + np.multiply((i-y), np.log(i-cdf))
    result = np.sum(out)
    return result

def optimal_StN3(df, prds, bounds, model, num_figs):
    x = np.array(df)
    nc = df.shape[1]
    f = np.linspace(bounds[0], bounds[1], 2000, endpoint=True)
    arr = np.array([0.0 for i in range(len(f))])
    arr2 = np.array([0.0 for i in range(len(f))])
    results = np.array([[0.0 for i in range(len(f))] for j in range(2)])
    results2 = np.array([[0.0 for i in range(1)] for j in range(2)])
    for j in range(len(f)):
        res = signal_to_noise_ratio2(df.iloc[:,[0,1]], prds, f[j])
        results[0, j] = res[0]
        results[1, j] = res[1]
    results = results[:,(results[1,:] > 0.01)]
    f1 = np.amin(results[1,:])
    min_index = np.argmin(results[1,:],axis=0)
    f2 = results[0, min_index]
    results2[0, 0] = f2
    results2[1, 0] = f1
    thrs = np.array([[0.0 for col in range(1)]for row in range(len(x))])
    for i in range(len(x)):
        thrs[i,0] = results2[0,0]
    x = np.concatenate((x, thrs), axis = 1)
    if model == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x[:,0], x[:,1],linewidth=2 ,color="black", label='Вероятности по Logit модели' )
        ax.plot(x[:,0], x[:,2],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.margins(0)        
        ax.set_yticks(np.arange(0, 1, 0.99999))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        years = mdates.YearLocator(base = 1)
        ax.xaxis.set_major_locator(years)
        plt.xticks(rotation=45)
        ax.set_ylim(ymin=0, ymax = 1)
        ax.set_title("", fontdict={'size':12})
        ax.get_xaxis().tick_bottom()
        plt.xticks(fontsize=6)
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["right"].set_alpha(.0)
        ax.legend();
        ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
        ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
        ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)
        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+num_figs + 2)+'.png')
    if model == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x[:,0], x[:,1],linewidth=2 ,color="black", label='Вероятности по Probit модели' )
        ax.plot(x[:,0], x[:,2],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.margins(0)
        ax.set_yticks(np.arange(0, 1, 0.99999))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        years = mdates.YearLocator(base = 1)
        ax.xaxis.set_major_locator(years)
        plt.xticks(rotation=45)
        ax.set_ylim(ymin=0, ymax = 1)
        ax.set_title("", fontdict={'size':12})
        ax.get_xaxis().tick_bottom()
        plt.xticks(fontsize=6)
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["right"].set_alpha(.0)
        ax.legend();
        ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
        ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
        ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5) 
        plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure' + str(5+num_figs + 3)+'.png')
    plt.show(block = False)

    return results2

        
def compind2(x, y, z, num_figs):
    x = np.array(x)
    coef = x[1,1:]
    coef = 1/coef
    y1 = np.array(y)
    y2 = y1[:,1:]
    z = np.array(z)
    thrs = float(z[0,1])
    thrs2 = np.array([[0.0 for col in range(1)]for row in range(len(y2))])
    for i in range(len(y2)):
        thrs2[i,0] = thrs
    compind = (np.matmul(y2,coef))/coef.sum()
    compind = (np.array([compind])).transpose()
    compind = np.concatenate((y1[:,0:1], compind), axis = 1)
    compind = np.concatenate((compind, thrs2), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(compind[:,0], compind[:,1],linewidth=2 ,color="black", label='Композитный индикатор' )
    ax.plot(compind[:,0], compind[:,2],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_yticks(np.arange(0, 1, 0.99999))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    years = mdates.YearLocator(base = 1)
    ax.xaxis.set_major_locator(years)
    plt.xticks(rotation=45)
    plt.margins(0.00)
    ax.set_ylim(ymin=0, ymax = 1)
    ax.set_title("Сигнальный подход", fontdict={'size':12})
    ax.get_xaxis().tick_bottom()
    plt.xticks(fontsize=6)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    ax.legend();
    ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)

    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure'+ str(5+num_figs + 3+num_figs+1)+'.png')
    plt.show(block = False)
   
    col_names = [str for i in range(2)]
    
    col_names.insert(0, "date")
    col_names[1] = "Композитный индикатор"
    col_names[2] = "Пороговый уровень"
    compind=pd.DataFrame(data = compind, columns=col_names)
    

    return compind
def fcast_logit(df, beta, thrs, num_figs):
    x0 = np.array(df)
    x = x0[:,1:]
    t = len(x)
    i = np.ones((t,1))
    x = np.concatenate((i,x), axis = 1)
    prfit = np.ones((t,1))/np.add(i, np.exp(-np.matmul(x,beta).astype(float)))
    dat = np.array([x0[:,0]]).transpose()
    x = np.concatenate((dat,prfit),axis = 1)
    threshold = np.array([[0.0 for col in range(1)]for row in range(len(x))])
    for i in range(len(x)):
        threshold[i,0] = thrs
    x = np.concatenate((x,threshold), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[:,0], x[:,1],linewidth=2 ,color="black", label='Вероятности по Logit модели' )
    ax.plot(x[:,0], x[:,2],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_yticks(np.arange(0, 1, 0.99999))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    years = mdates.YearLocator(base = 1)
    ax.xaxis.set_major_locator(years)
    plt.xticks(rotation=45)
    plt.margins(0.00)
    ax.set_ylim(ymin=0, ymax = 1)
    ax.set_title("Logit модель", fontdict={'size':12})
    ax.get_xaxis().tick_bottom()
    plt.xticks(fontsize=6)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    ax.legend();
    ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)

    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure'+ str(5+num_figs + 3+num_figs+2)+'.png')
    

    plt.show(block = False)
    return prfit

def fcast_probit(df, beta, thrs, num_figs):
    x0 = np.array(df)
    x = x0[:,1:]
    t = len(x)
    i = np.ones((t,1))
    x = np.concatenate((i,x), axis = 1)
    prfit = norm.cdf(np.matmul(x,beta).astype(float))
    dat = np.array([x0[:,0]]).transpose()
    x = np.concatenate((dat,prfit),axis = 1)
    threshold = np.array([[0.0 for col in range(1)]for row in range(len(x))])
    for i in range(len(x)):
        threshold[i,0] = thrs
    x = np.concatenate((x,threshold), axis = 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[:,0], x[:,1],linewidth=2 ,color="black", label='Вероятности по Probit модели' )
    ax.plot(x[:,0], x[:,2],linewidth=2 ,color="red", linestyle='dashed'  , label='Пороговый уровень')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_yticks(np.arange(0, 1, 0.99999))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    years = mdates.YearLocator(base = 1)
    ax.xaxis.set_major_locator(years)
    plt.xticks(rotation=45)
    plt.margins(0.00)
    ax.set_ylim(ymin=0, ymax = 1)
    ax.set_title("Logit модель", fontdict={'size':12})
    ax.get_xaxis().tick_bottom()
    plt.xticks(fontsize=6)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    ax.legend();
    ax.axvspan(*mdates.datestr2num(['2008-12-01', '2009-02-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2011-03-01', '2011-10-01']), color='gray', alpha=0.5)
    ax.axvspan(*mdates.datestr2num(['2014-12-01', '2015-02-01']), color='gray', alpha=0.5)

    plt.savefig(r'C:\Users\Karen\Desktop\AlertPy\Figure'+ str(5+num_figs + 3+num_figs+3)+'.png')
    plt.show(block = False)
    return prfit

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000
    # display.precision = 2  # set as needed
    
def make_report(table, num, table2, table3, table4):
    with open(r'C:\Users\Karen\Desktop\AlertPy\Report.tex','w',encoding="utf-8") as file:
        file.write('\\documentclass{article}\n')
        file.write('\\usepackage{pdflscape}\n')
        file.write('\\usepackage{graphicx}\n')
        file.write('\\usepackage{pgffor}\n')
        file.write('\\usepackage{tikz,catchfile}\n')
        file.write('\\usepackage[font=large]{caption}\n')
        file.write('\\usepackage{booktabs}\n')
        file.write('\\usepackage{longtable}\n')
        file.write('\\usepackage{bigstrut}\n')
        #file.write('\\usepackage[utf8]{inputenc}\n')
        file.write('\\usepackage[russian]{babel}\n')
        file.write('\\usepackage{lscape}\n')
        file.write('\\usepackage{float}\n')
        file.write('\\usepackage{xfp}\n')
        
        #file.write('\\usepackage{lipsum}\n')
        #file.write('\\usepackage{subcaption}\n')
        file.write('\\def\\fillandplacepagenumber{\\par\pagestyle{empty}\\vbox to 0pt{\\vss}\\vfill\\vbox to 0pt{\\baselineskip0pt\\hbox to\\linewidth{\\hss}\\baselineskip\\footskip\\hbox to\\linewidth{\\hfil\\thepage\\hfil}\\vss}}\n')
        file.write("\\title{\LARGE{ОТЧЕТ} \\\\О результатах расчетов \\\\ по системе раннего предупреждения \\\\ для Республики Беларусь}\n")
        file.write('\\begin{filecontents*}{\jobname.dat}\n')
        file.write(str(num+5))
        file.write('\\end{filecontents*}\n')
        file.write('\\begin{filecontents*}{\jobname1.dat}\n')
        file.write(str(num+5+1))
        file.write('\\end{filecontents*}\n')
        
        file.write('\\begin{document}\n')
        
        file.write('\\begin{titlepage}\n')
        file.write("\\maketitle\n")
        file.write("\\thispagestyle{empty}\n")
        file.write('\\end{titlepage}\n')
        
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure1.png}\n")
        file.write("\\caption{Распределение исходных показателей по критериям отбора}\n")
        file.write('\\end{figure}\n')
                     
       
        file.write('\\begin{landscape}\n')
        file.write(table.to_latex(index = False, caption = 'Отобранные потенциальные предикторы', longtable=True))
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')             
        
        file.write('\\begin{landscape}\n')    
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[height=10cm, width=18cm]{Figure2.png}\n")
        file.write("\\caption{Коэффициенты корреляций между отобранными предикторами}\n")
        file.write('\\end{figure}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')

        file.write('\\begin{landscape}\n')    
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[height=10cm, width=18cm]{Figure3.png}\n")
        file.write("\\caption{Коэффициенты корреляций между отобранными предикторами}\n")
        file.write('\\end{figure}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')
        
        file.write('\\begin{landscape}\n')    
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[height=10cm, width=18cm]{Figure4.png}\n")
        file.write("\\caption{Кластеризация отобранных предикторов на основе коэффициентов корреляций}\n")
        file.write('\\end{figure}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')

        
        file.write('\\begin{landscape}\n')    
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[height=10cm, width=20cm]{Figure5.png}\n")
        file.write("\\caption{Удельные веса исходных данных на главные компонент (ГК)}\n")
        file.write('\\end{figure}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')

            
        file.write('\\CatchFileDef\\upperlimit{\\jobname.dat}{}\n')
        file.write('\\begin{landscape}\n')
        file.write('\\foreach \\x in {6,...,\\upperlimit} \n')
        file.write('{\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n")
        file.write("\\includegraphics[height=10cm, width=18cm]{Figure\\x.png}\n")
        file.write("\\caption{Динамика главных компонент}\n")
        file.write('\\end{figure}\n')
        
        file.write('}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')
        
        
        
        file.write(table2.to_latex(index = False, caption = 'Пороговый уровень по сигнальному подходу'))
        
        file.write('\\CatchFileDef\\upperlimit{\\jobname1.dat}{}\n')
        file.write('\\foreach \\x in {\\fpeval{5+\\upperlimit-6+1},...,\\upperlimit} \n')
        file.write('{\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\x.png}\n")
        file.write("\\caption{Динамика композитного индикатора и пороговый уровень}\n")
        file.write('\\end{figure}\n')
        file.write('}\n')
        file.write(table3.to_latex(index = False, caption = 'Оцененная Logit модель'))
        file.write('\\foreach \\x in {\\fpeval{5+\\upperlimit-6+2},...,\\fpeval{\\upperlimit+1}} \n')
        file.write('{\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\x.png}\n")
        file.write("\\caption{Динамика вероятностей и пороговый уровень}\n")
        file.write('\\end{figure}\n')
        file.write('}\n')
        file.write(table4.to_latex(index = False, caption = 'Оцененная Probit модель'))
        file.write('\\foreach \\x in {\\fpeval{5+\\upperlimit-6+3},...,\\fpeval{\\upperlimit+2}} \n')
        file.write('{\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\x.png}\n")
        file.write("\\caption{Динамика вероятностей и пороговый уровень}\n")
        file.write('\\end{figure}\n')
        file.write('}\n')
        
        file.write('\\begin{landscape}\n')
        file.write('\\foreach \\x in {\\fpeval{5+\\upperlimit-6+4},...,\\fpeval{\\upperlimit+2+\\upperlimit-6}} \n')
        file.write('{\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[height=10cm, width=18cm]{Figure\\x.png}\n")
        file.write("\\caption{Бутстрап прогнозы главных компонент}\n")
        file.write('\\end{figure}\n')
        file.write('}\n')
        file.write("\\fillandplacepagenumber\n")
        file.write('\\end{landscape}\n')

        
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\fpeval{\\upperlimit+3+\\upperlimit-6}.png}\n")
        file.write("\\caption{Прогноз по сигнальному методу}\n")
        file.write('\\end{figure}\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\fpeval{\\upperlimit+4+\\upperlimit-6}.png}\n")
        file.write("\\caption{Прогноз по лигит модели}\n")
        file.write('\\end{figure}\n')
        file.write('\\begin{figure}[h!]\n')
        file.write("\\centering\n")
        file.write("\\setlength{\\unitlength}{\\textwidth}\n") 
        file.write("\\includegraphics[scale=1]{Figure\\fpeval{\\upperlimit+5+\\upperlimit-6}.png}\n")
        file.write("\\caption{Прогноз по пробит модели}\n")
        file.write('\\end{figure}\n')
        file.write('\\end{document}\n')

        

    x = subprocess.call('pdflatex Report.tex')
    if x != 0:
        print('Exit-code not 0, check result!')
    else:
        os.system('start Report.pdf')


    
