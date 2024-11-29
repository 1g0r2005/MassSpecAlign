"""
module for data visualisation
"""
import matplotlib.pyplot as plt
import numpy as np

def visual_additional(limmin=0,limmax=0, subplot = plt):
    print(type(subplot),subplot)
    nameplot = input('Enter name for plot:')
    xlabel,ylabel = [axis.strip() for axis in input('Enter name for each axis (x:,y:):').split(',')]
    subplot.set_title(nameplot)
    subplot.set_xlabel(xlabel)
    subplot.set_ylabel(ylabel)
    if limmin!=0 and limmax!=0:
        d = (limmax-limmin)//4
        subplot.set_xlim(limmin-d,limmax+d)

def visual_scatter(*data,refs = [] ,subplot = plt):
    '''create scatter plot from data'''
    x,y = data[0]
    subplot.scatter(x,y, s=6)
    if refs!=[]:
        for refline in refs:
            subplot.axvline(refline,color='black',linestyle=':',lw=1)

def kdeVis(rawdata,data,kdeType,show_input,color='k',subplot = plt):
    '''kde visualisation'''
    if show_input:
        subplot.plot(rawdata,np.zeros(rawdata.shape),'b+',ms=10)
    #print(data)
    subplot.plot(data[0],data[1],color,label=f'KDE({kdeType})')

def show():
    plt.show()
