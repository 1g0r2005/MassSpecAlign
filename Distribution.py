
# -*- coding: cp1251 -*-
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
from scipy import stats
from functools import partial
start = time.time()


FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
folderHDF = 'HDF\\Data\\'
cash = 'Cash\\indexes.txt'
delta = 2
vis_num = 1000
ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425, 387.152579670685]
def indexator(dataset, n):
    '''
    save ids for specs
    dataset - hdf dataset
    n - number of spec
    '''
    indexdata = dataset[0]
    index = np.where(indexdata == n)
    return index

def vis_dataset(*data):
    '''
    create graph by data in np array
    data - data (numpy)
    '''
    for singledata in data:
            x,y = singledata
            plt.scatter(x,y)
            plt.title('Aligned specs')
            plt.xlabel('m/z')
            plt.ylabel('intensity')

def border (ref,delta=2):
    '''
    borders for visualisation
    return np array of segments
    '''
    borders = []
    start = ref[0]
    end = ref[0]

    for i in range(1,len(ref)):
        if ref[i]-end<=2*delta:
            end = ref[i]
        else:
            borders.append([start-delta,end+delta])
            start = ref[i]
            end = ref[i]
    borders.append([start-delta,end+delta])
    return np.array(borders)

def kdeBandwidth(obj,fac=1./5):
    '''multiplied scott rule'''
    return np.power(obj.n,-1./(obj.d+4))*fac

def kde(rawdata,visual = True):
    condition = (180<=rawdata) & (rawdata<=200)
    data = rawdata[np.where(condition)]
    kde1 = stats.gaussian_kde(data,bw_method=partial(kdeBandwidth,fac=0.5))
    kde2 = stats.gaussian_kde(data,bw_method=partial(kdeBandwidth,fac=0.1))
    kde3 = stats.gaussian_kde(data,bw_method=partial(kdeBandwidth,fac=0.05))
    #delta = (data.max()-data.min())*0.25
    #x_eval = np.linspace(ref[0]-delta,ref[-1]+delta)
    x_eval = np.linspace(180,200)
    if visual:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data,np.zeros(data.shape), 'b+', ms=10)
        ax.plot(x_eval,kde1(x_eval),'red',label='KDE1 for ref peaks')
        ax.plot(x_eval,kde2(x_eval),'green',label='KDE2 for ref peaks')
        ax.plot(x_eval,kde3(x_eval),'yellow',label='KDE3 for ref peaks')


with h5py.File(folderHDF+FileName[1],'r') as f:
    features = f['roi2_e033/00/features']
    data_for_kde = []
    for number in range(vis_num):
        print(number)
        index = indexator(features,number)
        data = np.vstack((features[1][index],features[2][index]))
        for reg in border(ref,delta=delta):
            st,en = reg[0],reg[1]
            condition = (st<=data[0,:]) & (data[0,:]<=en)
            data_filtered = data[:,np.where(condition)]
            data_for_kde.append(data_filtered[0])
            #vis_dataset(data_filtered)
    #for line in ref:
        #plt.axvline(line,color='black',linestyle=':',lw=1)

    kde(np.hstack(data_for_kde)[0])
    end = time.time()
    print("The time of execution of above program is :",(end-start) * 10**3, "ms")
    plt.show()

