
# coding: utf-8

# In[10]:


import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
sys.path.append('/project/bioinformatics/DLLab/Alex/Projects/utilities/')
import network_functions as nf
import re
import math
#from machine_learning_stats import metrics_from_confusion_matrix
#import tensorflow_utilities
import seaborn as sns
from sklearn.neighbors import KernelDensity
import scipy.stats
from scipy import stats
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
import seaborn as sns
import pickle as pkl
import numpy as np
from BioMarkerIdentification import fLoadModels
from IMPAC_DenseNetwork import read_config_file

def fCountHidden(config=None, iModel=None):
    if config == None:
        config = nf.read_config_file('IniFiles/Dense_{0:02d}.ini'.format(iModel))
    intHidden = [xi for xi in config.keys() if 'dense' in xi.lower()].__len__()
    return intHidden

def fDropout(config=None, iModel=None):
    if config == None:
        config = nf.read_config_file('IniFiles/Dense_{0:02d}.ini'.format(iModel))
    flDropout = float(config['layer/Dropout0']['rate'])
    return (flDropout)

def fBottomLayerSizeLog2(config=None, iModel=None):
    if config == None:
        config = nf.read_config_file('IniFiles/Dense_{0:02d}.ini'.format(iModel))
    intLayerSize = int(config['layer/input']['units'])

    return (math.log(intLayerSize, 2.0))

def fRegularization(config=None, iModel=None):
    if config == None:
        config = nf.read_config_file('IniFiles/Dense_{0:02d}.ini'.format(iModel))
    flRegularization = float(config['layer/input']['regularizer'])
    return (flRegularization)

def plt2DKDEs(dfModelMetrics, lComparison, flPercent):
    topModels = dfModelMetrics.head(int(dfModelMetrics.shape[0] * (flPercent / 100.)))
    bottomModels = dfModelMetrics.tail(int(dfModelMetrics.shape[0] * (flPercent / 100.)))

    Zdata = [np.array(topModels[lComparison]), np.array(bottomModels[lComparison])]
    xmin = np.min([Zi[:, 0].min() for Zi in Zdata]) * .8
    xmax = np.max([Zi[:, 0].max() for Zi in Zdata]) * 1.2
    ymin = np.min([Zi[:, 1].min() for Zi in Zdata]) * .8
    ymax = np.max([Zi[:, 1].max() for Zi in Zdata]) * 1.2
    X, Y = np.mgrid[xmin:xmax:100j,
           ymin:ymax:100j]
    kdes = [KernelDensity(kernel='gaussian', bandwidth=1).fit(Zi) for Zi in Zdata]
    Z = [np.exp(kde.score_samples(np.dstack([X.flatten(), Y.flatten()])[0]).reshape(*X.shape)) for kde in kdes]

    fPlot2DDist(X, Y, Z, ['Orange', 'Blue'], ['Oranges', 'Blues'], ' vs '.join(lComparison), lComparison[0],
                lComparison[1])

def fPlot2DDist(X, Y, Zs, colors, cmaps, title, xlabel, ylabel):
    fig = plt.figure(figsize=(10,6))
    #plt.suptitle(title)
    ax = fig.add_subplot(1,1,1,projection='3d')

#     ax1 = fig.add_subplot(2,2,1,projection='3d')
#     ax2 = fig.add_subplot(2,2,2,projection='3d')
#     ax3 = fig.add_subplot(2,2,3,projection='3d')
#     ax4 = fig.add_subplot(2,2,4,projection='3d')
#     for ax in [ax1,ax2,ax3,ax4]:
#         ax.view_init(30, 30)
#         ax.set_ylabel(ylabel)
#         ax.set_xlabel(xlabel)
    ax.view_init(30, 30)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    flMin = np.amin(Zs)
    flMax = np.amax(Zs)
    flMid = (flMin + flMax) / 2
    ax.set_zticks([flMin, flMid, flMax])#, ['Low', 'Med', 'High'])
    ax.set_zticklabels(['Low', 'Med', 'High'])

    fig.tight_layout()

    for Zi, Ci, CMi in zip(Zs, colors, cmaps):
        ax.contour3D(X, Y, Zi, 30, cmap=CMi)
        cset = ax.contour(X, Y, Zi, zdir='x', offset=X.min(), cmap=CMi)
        cset = ax.contour(X, Y, Zi, zdir='y', offset=Y.min(), cmap=CMi)
        ax.set_zticks([flMin, flMid, flMax])#, ['Low', 'Med', 'High'])
        ax.set_zticklabels(['Low', 'Med', 'High'])

    ax.set_zticks([flMin, flMid, flMax])#, ['Low', 'Med', 'High'])
    ax.set_zticklabels(['Low', 'Med', 'High'])
    ax.set_zlabel('Density')
    ax.view_init(elev=30, azim=30)
    plt.show()

def fFetchList(sType, sModality='combined', sAtlas='basc122'):
    if sType=='LSTM' or sType=='Dense':
        # initialize dataframe
        pdPerformance=pd.DataFrame(index=range(50), columns=['Performance','hidden_layers','log_2_bottom_layer_width',
                                                             'Dropout','regularization'])
        # load results and parameters for each model
        for i in range(50):
            sIniLoc='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/' \
                    f'Parallelization/IniFiles/{sType}_{i:02}.ini'
            lsPerf=[]

            # Take average of performance across cross validation folds
            for iCV in range(1,4):
                sPerfLoc='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/' \
                         f'Parallelization/TrainedModels/ISBIRerun/{sType}/'\
                         f'{sType}_{i:02}{sModality}{sAtlas}ROCScoreCrossVal{iCV}.p'
                lsPerf.append(pkl.load(open(sPerfLoc, 'rb')))
            flPerf=np.mean(lsPerf)

            # put in pd dataframe
            pdPerformance.loc[i, 'Performance']=flPerf

            config = read_config_file(sIniLoc)
            pdPerformance.loc[i,'log_2_bottom_layer_width']=math.log(int(config['layer/input']['units']),2)
            pdPerformance.loc[i,'hidden_layers']=len([x for x in config if x.__contains__(f'layer/{sType}')])
            pdPerformance.loc[i,'Dropout']=float(config['layer/Dropout0']['rate'])
            pdPerformance.loc[i,'regularization']=float(config['layer/input']['regularizer'])
            pdPerformance=pdPerformance.sort_values(by=['Performance'], ascending=False)

        return pdPerformance, ['hidden_layers','log_2_bottom_layer_width',
                                                             'Dropout','regularization']

    else:
        cModel=fLoadModels('LinRidge', 'combined', 'basc122')[0]
        pdPerformance=pd.DataFrame(index=range(50), columns=['Performance','max_iter_x_10000', 'alpha'])
        for i in range(50):
            pdPerformance.loc[i,'Performance']=cModel.grid_scores_[i].mean_validation_score
            pdPerformance.loc[i,'max_iter_x_10000']=cModel.grid_scores_[i].parameters['max_iter']/10000
            pdPerformance.loc[i,'log_10(alpha)']=math.log(cModel.grid_scores_[i].parameters['alpha'],10)
        pdPerformance=pdPerformance.sort_values(by=['Performance'], ascending=False)

        return pdPerformance, ['max_iter_x_10000', 'log_10(alpha)']


if __name__=='__main__':
    pdPerformance, lToTest = fFetchList('LinRidge', sModality='combined', sAtlas='basc122')
    plt.style.use('seaborn')
    for lComparison in itertools.combinations(lToTest, 2):
        plt2DKDEs(pdPerformance, list(lComparison), 15)
        plt.savefig(
            '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/KDEs/{}vs{}.png'\
            .format(lComparison[0].replace("\'", "").replace(" ",""), lComparison[1].replace("\'", "").replace(" ","")))
