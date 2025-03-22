import numpy as np
import h5py
from rpy2.robjects import R
from sympy.physics import minkowski_tensor

import distr_visual as vis
from tqdm import tqdm
import matplotlib.pyplot as plt

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
folderHDF = 'HDF\\Data\\'
cash = 'Cash\\cash.hdf5'
ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425,
       387.152579670685]

def indexes(dataset, ID: int):
    '''
    return colums for spec with special ID
    dataset - hdf vis_dataset
    ID - ID of spec
    '''
    indexdata = dataset[0]
    return np.where(indexdata.astype('int') == ID)


def roi(dots, delta=2):
    '''
    return regions of interest around list of dots
    dost - array of dots in roi
    delta - the width of the region around a single point
    '''
    borders = []
    start = dots[0]
    end = dots[0]

    for i in range(1, len(dots)):
        if dots[i] - end <= 2 * delta:
            end = dots[i]
        else:
            borders.append([start - delta, end + delta])
            start = dots[i]
            end = dots[i]
            borders.append([start - delta, end + delta])
    return np.array(borders)


def dataFilter(rawdata, borders):
    minMZ, maxMZ = borders
    condition = (minMZ <= rawdata[0, :]) & (rawdata[0, :] <= maxMZ)
    dataFiltered = rawdata[:, np.where(condition)]
    return dataFiltered


def main():
    with h5py.File(folderHDF + FileName[0], 'r') as f_raw:
        with h5py.File(folderHDF + FileName[1], 'r') as f_aln:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            features_raw = f_raw['roi2_e033/00/features']
            features_aln = f_aln['roi2_e033/00/features']

            setnum = int(max(features_raw[0])) + 1
            setnum = int(input("setnum for tests: "))
            maxpos = 10
            maxneg = -10
            for ID in tqdm(range(setnum)):
                index_raw= indexes(features_raw, ID)
                index_aln = indexes(features_aln, ID)
                data_raw = np.array(features_raw[1][index_raw])
                data_aln = np.array(features_aln[1][index_aln])
                if data_raw.size!=data_aln.size:
                    print(f'cannot be compared! ID:{ID}')
                    print(data_raw.size)
                    print('-----------')
                    print(data_aln.size)
                #print(data_raw)
                #print(data_aln)
                minsize = min(len(data_aln), len(data_raw))
                dist_array = np.sort(data_aln[:minsize]) - np.sort(data_raw[:minsize])
                vis.histogramm(dist_array, np.arange(maxneg, maxpos, 0.1),alpha=0.5, subplot=ax1)
                vis.histogramm(dist_array, np.arange(maxneg, maxpos, 1),alpha=0.5, subplot=ax2)
                '''
                maxtemp,mintemp = dist_array.max(),dist_array.min()
                if maxpos<maxtemp:
                    maxpos=maxtemp
                    print(maxpos,maxneg)
                if maxneg>mintemp:
                    maxneg=mintemp
                    print(maxpos, maxneg)
                '''

            colw = 0.1
            #shiftmz = np.hstack(shiftmz)
            #print(len(shiftmz))
            #vis.histogramm(shiftmz, np.arange(maxneg, maxpos, 0.1), subplot=ax1)
            #vis.histogramm(shiftmz, np.arange(maxneg, maxpos, 1), subplot=ax2)

            #vis.visual_scatter((shiftmz,np.full(len(shiftmz), 0)),subplot=ax2)

    vis.show()




if __name__ == '__main__':
    main()
