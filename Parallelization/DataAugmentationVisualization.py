#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 18 Nov, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from textwrap import wrap
import pickle as pkl
import itertools
import glob
import numpy as np
import keras


def fSmooth(aArray):
    """
    Applies smoothing filter to timeseries
    :param aArray: timeseries
    :return: aSmoothed timeseries
    """
    import scipy as sp

    aSmoothed = sp.ndimage.gaussian_filter(aArray, 5)

    return aSmoothed

def fPlotTraining(dSingleResult, sTitle, sSaveLoc):
    """
    Plot the training data of the experiment
    TODO poorly tested
    :param dSingleResult:
    :param sTitle:
    :param sSaveLoc:
    :return:
    """
    iMinLen = min([len(np.array(dSingleResult[0].history['val_acc'])),
                   len(np.array(dSingleResult[1].history['val_acc'])),
                   len(np.array(dSingleResult[2].history['val_acc']))])
    aValAcc = (np.array(dSingleResult[0].history['val_acc'])[:iMinLen] + np.array(dSingleResult[1].history['val_acc'])[
                                                                         :iMinLen] + np.array(
        dSingleResult[2].history['val_acc'])[:iMinLen]) / 3.0
    aAcc = (np.array(dSingleResult[0].history['acc'][:iMinLen]) + np.array(dSingleResult[1].history['acc'])[
                                                                  :iMinLen] + np.array(dSingleResult[2].history['acc'])[
                                                                              :iMinLen]) / 3.0

    if True:  # np.max(aValAcc)>0.75 and np.max(fSmooth(aValAcc))>0.75:
        plt.plot(range(len(aValAcc)), fSmooth(aValAcc))
        plt.plot(range(len(aAcc)), fSmooth(aAcc))

        print(f'Max Validation Acc: {np.max(aValAcc)}')
        print(f'Max Smoothed Validation Acc: {np.max(fSmooth(aValAcc))}')

        plt.title(f'{sTitle} Acc \n Max Validation Acc: {np.max(aValAcc)} \n Max Smoothed Validation Acc: {np.max(fSmooth(aValAcc))}')
        plt.tight_layout()
        # plt.savefig(f'{sSaveLoc}/{sTitle}.png')
        plt.show()

def fLoadROC(sKey, sTrValTe):
    """
    Loads the ROC results for the experiment
    :param sKey: key of dResults, ie. directory for a given trial
    :param sTrValTe: flag for which data to fetch
    :return: mean Tr/Val/Te, std Tr/Val/Te across trials
    """
    #set path to data
    sRoot = os.path.join(
        '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Augmented/', sKey)

    # Set path to either Tr, Val, or Te depending on flag given
    if sTrValTe == 'Tr':
        lsFiles = [f'Dense_08Dense_TrROCScore_CV{iCV}.p' for iCV in range(3)]
    elif sTrValTe == 'Val':
        lsFiles = [f'Dense_08Dense_ROCScore_CV{iCV}.p' for iCV in range(3)]
    elif sTrValTe == 'Te':
        lsFiles = [f'Dense_08Dense_ROCScore_Full.p']

    # Fill array with results
    aData = np.zeros((len(lsFiles)))
    for i, sFile in enumerate(lsFiles):
        try:
            aData[i] = pkl.load(open(os.path.join(sRoot, sFile), 'rb'))
        except:
            aData[i] = 0

    # use only nonzero values for mean and std
    aData = aData[np.nonzero(aData)]

    # return mean and std of data
    return np.mean(aData), np.std(aData)

def fNoise(sKey):
    """
    Formats noise flag correctly
    :param sKey:
    :return: Correct flag
    """
    if 'Both' in sKey:
        return 'Both'
    elif 'Finaltimeseries' in sKey:
        return 'Timeseries'
    elif 'Components' in sKey:
        return 'Components'
    else:
        return 'None'

if __name__ == '__main__':
    # set the Paths
    sRoot = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Augmented'
    sSaveLoc = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Images' \
              '/DL_Aug_Comparison_Windowed'

    # Fetch the trial names
    lsAugMethods = glob.glob(f'{sRoot}/*Windowsize*')

    # Load the files for each trial
    dResults={}
    for sAugMethod in lsAugMethods:
        dResults[sAugMethod.split('/')[-1]] = {}
        for iCV in range(3):
            sFile = f'{sAugMethod}/Dense_08Dense_History_CV{iCV}.p'
            print(sFile)
            try:
                dResults[sAugMethod.split('/')[-1]][iCV] = pkl.load(open(sFile, 'rb'))
            except:
                dResults[sAugMethod.split('/')[-1]][iCV] = 0

    # Initialize pandas Dataframe
    pdWindowedResults = pd.DataFrame(columns=['Augmentation Method', 'Noise in timeseries', 'Noise in components',
                                              'Number of additional samples', 'Train AUC ROC', 'Train std. error',
                                              'Validation AUC ROC', 'Validation std. error', 'Test AUC ROC',
                                              'Generalization Error', 'Noise', 'Generalization Error Te',
                                              'Window Size'])
    # Place results in the pd dataframe
    iCol=0
    for sKey in dResults.keys():
        if 'Windowsize' in sKey:
            print(sKey)
            pdWindowedResults = pdWindowedResults.append(pd.Series(name=iCol))
            pdWindowedResults.loc[iCol]['Augmentation Method'] = sKey[0:3]
            pdWindowedResults.loc[iCol]['Noise in timeseries'] = 'None' if not (('Both' in sKey) or ('Finaltimeseries' in sKey)) else 'Present'
            pdWindowedResults.loc[iCol]['Noise in components'] = 'None' if not (('Both' in sKey) or ('Components' in sKey)) else 'Present'
            pdWindowedResults.loc[iCol]['Number of additional samples'] = int(sKey.split('_')[-3].strip('xAugmented'))*2
            pdWindowedResults.loc[iCol]['Train AUC ROC'], pdWindowedResults.iloc[iCol]['Train std. error'] = fLoadROC(sKey,'Tr')
            pdWindowedResults.loc[iCol]['Validation AUC ROC'], pdWindowedResults.iloc[iCol]['Validation std. error'] = fLoadROC(sKey, 'Val')
            pdWindowedResults.loc[iCol]['Test AUC ROC'], _ = fLoadROC(sKey, 'Te')
            pdWindowedResults.loc[iCol]['Generalization Error'] = pdWindowedResults.iloc[iCol]['Train AUC ROC'] - \
                                                                  pdWindowedResults.iloc[iCol]['Validation AUC ROC']
            pdWindowedResults.loc[iCol]['Noise'] = fNoise(sKey)
            pdWindowedResults.loc[iCol]['Generalization Error Te'] = pdWindowedResults.iloc[iCol]['Validation AUC ROC'] - \
                                                                     pdWindowedResults.iloc[iCol]['Test AUC ROC']
            pdWindowedResults.loc[iCol]['Window Size'] = int(sKey.split('_')[-2][-1])
            iCol += 1

    # save dataframe
    pkl.dump(pdWindowedResults, open(
        '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/DataAugmentation'
        '/pdWindowedResults_3Trials.p','wb'))
