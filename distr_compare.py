import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import distr_dens
import distr_dens as dens
import distr_visual as vis

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
folderHDF = 'HDF\\Data\\'
cash = 'Cash\\cash.hdf5'
ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425,
       387.152579670685]


def indexes(dataset, ds_id: int):
    """
    return colums for spec with special ID
    dataset - hdf vis_dataset
    ID - ID of spec
    """
    indexdata = dataset[0]
    return np.where(indexdata == ds_id)

def roi(dots, delta=2):
    """
    return regions of interest around list of dots
    dost - array of dots in roi
    delta - the width of the region around a single point
    """
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
    """
    filter dataset by defined borders (to specify which subset to discover)
    """
    minMZ, maxMZ = borders
    condition = (minMZ <= rawdata[0, :]) & (rawdata[0, :] <= maxMZ)
    dataFiltered = rawdata[:, np.where(condition)]
    return dataFiltered

def func_process(destination,roiForAll,sbplt,result_dict):
    with h5py.File(destination,'r') as f:
        features = f['roi2_e033/00/features']
        setnum = int(max(features[0])) + 1

        setnum = 100 # Ограничение для отладки

        xData = []
        yData = []
        for ID in tqdm(range(setnum)):
            index = indexes(features, ID)
            data = np.vstack((features[1][index], features[2][index]))
            # data - one spec
            dataFiltered = dataFilter(data, roiForAll)
            # only m/z for kde (maybe i should use intensity or int. area as weights)
            #print(dataFiltered)
            yData.append(dataFiltered[1])
            xData.append(dataFiltered[0])


        xDataSummary = np.hstack(xData)[0]
        yDataSummary = np.hstack(yData)[0]
        DataSummary = np.vstack((xDataSummary,yDataSummary))
        densROI = dens.kdePython(xDataSummary, region=roiForAll, show=False, bw='ISJ', color='g',
                                 subplot=sbplt)
        locMax, locMin = dens.findpeaks(densROI)

        out = {"DS":DataSummary,'DENS':densROI,"MAX":locMax}
        result_dict.update(out)

def main():
    manager = mp.Manager()
    processes = []
    result_total = [manager.dict() for _ in range(len(FileName))]
    regions = roi(ref,delta=2)

    print(regions)
    roiForAll = regions[int(input('Enter number of ROI:')) - 1]
    print(roiForAll)

    fig, axes = plt.subplots(2, 1, sharex=True)

    for number in range(len(FileName)):

        p = mp.Process(target=func_process, args=(folderHDF+FileName[number],roiForAll,axes[number],result_total[number]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print('Completed\n')

    for i in range(len(FileName)):
        visRawData = result_total[i]['DS']
        visDens = result_total[i]['DENS']
        print(visRawData)
        print(roiForAll)

        distr_dens.cluster(visRawData[0].reshape(-1,1),show=True,subplot=axes[i])
        #vis.visual_scatter(visRawData, refs=[], subplot=axes[i])
        vis.kdeVis(data=visDens, rawdata=visRawData, kdeType='ISJ', show_input=True, color='k', subplot=axes[i])
        vis.visual_additional(limmin=roiForAll[0], limmax=roiForAll[1], subplot=axes[i])


    vis.show()


if __name__ == '__main__':
    main()
