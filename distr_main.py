import numpy as np
import h5py
from rpy2.robjects import R
import distr_visual as vis
import distr_dens as dens
from tqdm import tqdm
import matplotlib.pyplot as plt

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
folderHDF = 'HDF\\Data\\'
cash = 'Cash\\cash.hdf5'
ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425, 387.152579670685]

'''
def cash_call(datasetname,file):
    def actualDecorator(f):
        def wrapper(*args,**kwargs):
            #(func,args,kwargs,value)
            name = f.__name__()
            unnamedargs_current= args
            namedargs_current = kwargs

            dataset = file[datasetname]

            condition = np.all(dataset[:3] == np.array((name,unnamedargs_current,namedargs_current)))
            if

        return wrapper
    return actualDecorator
'''

def indexes(dataset,ID:int):
    '''
    return colums for spec with special ID
    dataset - hdf vis_dataset
    ID - ID of spec
    '''
    indexdata = dataset[0]
    return np.where(indexdata == ID)

def roi(dots,delta = 2):
    '''
    return regions of interest around list of dots
    dost - array of dots in roi
    delta - the width of the region around a single point
    '''
    borders = []
    start = dots[0]
    end = dots[0]

    for i in range(1,len(dots)):
        if dots[i]-end <= 2*delta:
            end = dots[i]
        else:
            borders.append([start-delta,end+delta])
            start = dots[i]
            end = dots[i]
            borders.append([start-delta,end+delta])
    return np.array(borders)

def dataFilter(rawdata,borders):
    minMZ,maxMZ = borders
    condition = (minMZ<=rawdata[0,:]) & (rawdata[0,:]<=maxMZ)
    dataFiltered = rawdata[:,np.where(condition)]
    return dataFiltered

def main():
    with h5py.File(folderHDF+FileName[1],'r') as f:
        features = f['roi2_e033/00/features']
        setnum = int(max(features[0]))+1
        regions = roi(ref,delta=2)
        setnum = 1000
        print(regions)
        roiForAll = regions[int(input('Enter number of ROI:'))-1]
        print(roiForAll)
        xDataSummary = []

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        for ID in tqdm(range(setnum)):
            index = indexes(features,ID)
            data = np.vstack((features[1][index],features[2][index]))
            # data - one spec
            dataFiltered = dataFilter(data,roiForAll)
            vis.visual_scatter(dataFiltered,refs=ref,subplot=ax1)
            # only m/z for kde (maybe i should use intensity or int. area as weights)
            xDataSummary.append(dataFiltered[0])
            

        #dens.kdePython(np.hstack(xDataSummary)[0],region=roiForAll,show=True)
        densROI = dens.kdePython(np.hstack(xDataSummary)[0],region=roiForAll,show=True,bw='ISJ',color='g',subplot=ax2)
        #densROI = dens.kdeR(np.hstack(xDataSummary)[0],region=roiForAll,show=True,bw='ucv',color='g',subplot=ax2)
        locMax,locMin = dens.findpeaks(densROI)
        
        vis.visual_scatter(locMax,subplot=ax2)
        vis.visual_scatter(locMin,subplot=ax2)
        
        vis.visual_additional(limmin=roiForAll[0],limmax=roiForAll[1],subplot=ax1)
        vis.visual_additional(limmin=roiForAll[0],limmax=roiForAll[1],subplot=ax2)
        vis.show()

if __name__ == '__main__':
    main()
