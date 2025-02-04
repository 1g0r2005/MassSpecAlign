"""module for density estimation"""
import rpy2
from rpy2 import robjects
import rpy2.robjects.packages  as pkgs
import rpy2.robjects.numpy2ri
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import distr_visual
import distr_visual as vis
import matplotlib.pyplot as plt
from KDEpy import FFTKDE


def initr():
    global base
    global utils
    global stats
    base = pkgs.importr("base")
    utils = pkgs.importr("utils")
    stats = pkgs.importr("stats")
    rpy2.robjects.numpy2ri.activate()

def kdeR(rawdata,region,show=True,bw='nrd0',adjust=1,color='k',subplot=plt): #should try SJ
    """
    more complex algorithm for multimodal data using R language packages
    """
    initr()
    rawdata = np.array(rawdata,dtype="float64")
    dens = stats.density(rawdata,bw=bw,adjust=adjust)
    #base.plot(dens,main = "KDE")
    data = np.vstack((np.array(dens.rx2('x')),np.array(dens.rx2('y'))))
    if show:
        vis.kdeVis(rawdata=rawdata,data=data,kdeType=f'R;{bw}',show_input=True,color=color,subplot=plt)
    return data

def kdePython(rawdata,region,show=True,bw='ISJ',color='k',subplot=plt):
    """
    Build kernel density estimation for vis_dataset
    data - np.array vis_dataset
    region - borders for kdeBandwidth
    show - draw matplotlib graph if True
    """
    initr()
    kde = FFTKDE(kernel='biweight', bw=bw)
    x_eval = np.linspace(region[0],region[1])
    kde_values = kde.fit(rawdata).evaluate(x_eval)

    data = np.vstack((x_eval,kde_values))
    if show:
        vis.kdeVis(rawdata=rawdata,data=data,kdeType=f'Python;{bw}',show_input=True,color=color,subplot=subplot)
    return data

def findpeaks(dens):
    """
    find local maxima points and clusters
    """
    ex_code = """
        local_extrema <- function(points){

        x <- points[1,]
        y <- points[2,]
        print(x)
        print(y)
        diffs <- diff(y)
        sign_changes <- diff(sign(diffs))
        minima <- which(sign_changes == 2 | (sign_changes == 1 & sign(diffs[-1]) > 0))+1
        maxima <- which(sign_changes == -2 | (sign_changes == -1 & sign(diffs[-1]) < 0))+1
        
        
        return(list(minima = rbind(x[minima],y[minima]), maxima = rbind(x[maxima],y[maxima])))
        }
    """

    robjects.r(ex_code)
    rlocalextrema = robjects.globalenv['local_extrema']
    localminima,localmaxima = rlocalextrema(dens)
    return localmaxima,localminima

def opt_epsilon(distances):
    dy = np.gradient(distances)
    d2y = np.gradient(dy)

    curv = np.abs(d2y)/(1+dy**2)**(3/2)

    index = np.argmax(curv)
    return distances[index]

def k_dist(data):
    print('----------------')
    x = np.sort(np.unique(np.ravel(data)))
    print('x ', x)

    left,right = np.full(x.shape,np.inf),np.full(x.shape,np.inf)
    for i in range(x.shape[0]):
        if i != 0:
            left[i] = abs(x[i] - x[i - 1])
        if i != x.shape[0] - 1:
            right[i] = abs(x[i] - x[i + 1])

    distances = np.sort(np.minimum(left,right))
    print('dist ',distances)
    print('----------------')
    return distances

def cluster(data,show=False,subplot=plt):
    """
    DBSCAN clustering
    """
    epsilon = opt_epsilon(k_dist(data))
    labels = DBSCAN(eps=epsilon, min_samples=2).fit_predict(data)
    uniq_labels = set(labels)-{-1}
    n_clusters_ = len(uniq_labels)

    print(n_clusters_)
    if show:
        distr_visual.dbscanVis(data,labels,uniq_labels,subplot=subplot)
    return labels