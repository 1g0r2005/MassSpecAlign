import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
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

def func_process(destination,roiForAll,queue):
    with h5py.File(destination,'r') as f:
        features = f['roi2_e033/00/features']
        setnum = int(max(features[0])) + 1

        setnum = 1000 # Ограничение для отладки

        xDataSummary = []

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        for ID in tqdm(range(setnum)):
            index = indexes(features, ID)
            data = np.vstack((features[1][index], features[2][index]))
            # data - one spec
            dataFiltered = dataFilter(data, roiForAll)
            #vis.visual_scatter(dataFiltered, refs=ref, subplot=ax1)
            # only m/z for kde (maybe i should use intensity or int. area as weights)
            xDataSummary.append(dataFiltered[0])

        densROI = dens.kdePython(np.hstack(xDataSummary)[0], region=roiForAll, show=True, bw='ISJ', color='g',
                                 subplot=ax1)
        locMax, locMin = dens.findpeaks(densROI)

        out = (dataFiltered,densROI,locMax)
        queue.put(out)

def main():
    resultQueue = mp.Queue()
    processes = []

    regions = roi(ref,delta=2)

    print(regions)
    roiForAll = regions[int(input('Enter number of ROI:')) - 1]
    print(roiForAll)

    for fileHDF in FileName:
        p = mp.Process(target=func_process, args=(folderHDF+fileHDF,roiForAll,resultQueue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = []
    while not resultQueue.empty():
        results.append(resultQueue.get())

    print('Completed\n')
    print(f'Results: {results}')


if __name__ == '__main__':
    main()
