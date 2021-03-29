#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file runs the top IMPAC models on the external ABIDE dataset
    
Originally developed for use in the IMPAC project
Created by Cooper Mellema on 24 Feb, 2020
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import copy
import pandas as pd
import keras as k
import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedShuffleSplit
from BioMarkerIdentification import fLoadModels


def fLoadRetestData(sTag, sNormed=''):
    """
    Loads retest data
    :param sTag: string, one of: ABIDE1, ABIDE2_Site1, or ABIDE2_Site2
    :return: dict of the results
    """
    sRoot='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/RetestData/'

    # load target values first (ASD or not)
    dResults = {'Target': pd.read_csv(f'{sRoot}{sTag}_Targets.csv', index_col=0, header=None)}

    # Load per-atlas features
    for sAtlas in ['BASC064', 'BASC122', 'BASC197']:
        dResults.update({sAtlas: pd.read_csv(f'{sRoot}{sTag}_{sAtlas}_AllFeatures{sNormed}.csv', index_col=0)})

    return dResults

def fCalculatePerformance(dModels, dData, aTarget):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    for sModel in [sKey for sKey in dModels.keys() if not'Target' in sKey]:
        cModel=dModels[sModel]

        # make a compiled predictor for the Neural network
        cPredictor = k.models.Model(inputs=cModel.inputs, outputs=cModel.outputs)
        cPredictor._make_predict_function()

        # predict the results
        aPredicted = cPredictor.predict(np.expand_dims(np.expand_dims(dData[sModel].values, axis=1), axis=3))

        # compare to actual:
        flRawScore = skm.roc_auc_score(aTarget, aPredicted)
        dResults.update({sModel: flRawScore})

    # return results per model
    return dResults

def fTuneModel(dModels, dData, aTarget):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    for sModel in [sKey for sKey in dModels.keys() if not'Target' in sKey]:
        cModel=copy.deepcopy(dModels[sModel])

        # Perform a cross-validation experiment
        cSplitter=StratifiedShuffleSplit(n_splits=10, random_state=0)

        # train on 1/3, validate on 2/3
        i=0
        dResults[sModel]={}
        for idxTest, idxTrain in cSplitter.split(dData[sModel].values, aTarget):
            aXTrain = np.expand_dims(np.expand_dims(dData[sModel].values[idxTrain], axis=1), axis=3)
            aXTest = np.expand_dims(np.expand_dims(dData[sModel].values[idxTest], axis=1), axis=3)
            aYTrain = aTarget[idxTrain]
            aYTest = aTarget[idxTest]

            # identical fit parameters to original data
            kerStopping = k.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.01, patience=20,
                                                            restore_best_weights=True)
            cModel.fit(x=aXTrain, y=aYTrain, validation_data=(aXTest, aYTest),
                                     epochs=500, callbacks=[kerStopping], batch_size=128)

            # compare to actual:
            flRawScore = skm.roc_auc_score(aYTest, cModel.predict(aXTest))

            print(f'{sModel}: Val. Perf: {flRawScore}')
            dResults[sModel][i] = flRawScore
            i+=1

    # return results per model
    return dResults

def fTuneModelDiffData(dModels, dTrainData, dTestData, aYTrain, aYTest):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    for sModel in [sKey for sKey in dModels.keys() if not'Target' in sKey]:
        cModel=copy.deepcopy(dModels[sModel])

        # Perform a cross-validation experiment
        #cSplitter=StratifiedShuffleSplit(n_splits=10, random_state=0)

        dResults[sModel]={}
        aXTrain = np.expand_dims(np.expand_dims(dTrainData[sModel].values, axis=1), axis=3)
        aXTest = np.expand_dims(np.expand_dims(dTestData[sModel].values, axis=1), axis=3)

        # identical fit parameters to original data
        kerStopping = k.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.01, patience=20,
                                                        restore_best_weights=True)
        cModel.fit(x=aXTrain, y=aYTrain, validation_data=(aXTest, aYTest),
                                 epochs=500, callbacks=[kerStopping], batch_size=128)

        # compare to actual:
        flRawScore = skm.roc_auc_score(aYTest, cModel.predict(aXTest))

        print(f'{sModel}: Val. Perf: {flRawScore}')
        dResults[sModel] = flRawScore

    # return results per model
    return dResults

if '__main__'==__name__:
    # load up the data
    dABIDE2Site1 = fLoadRetestData('ABIDE2_Site1')
    dABIDE2Site2 = fLoadRetestData('ABIDE2_Site1')
    dABIDE2Data = {
        'BASC064': dABIDE2Site1['BASC064'].append(dABIDE2Site2['BASC064']),
        'BASC122': dABIDE2Site1['BASC122'].append(dABIDE2Site2['BASC122']),
        'BASC197': dABIDE2Site1['BASC197'].append(dABIDE2Site2['BASC197']),
        'Target': dABIDE2Site1['Target'].append(dABIDE2Site2['Target'])
    }
    lsData =[fLoadRetestData('ABIDE1', sNormed='HistNormed'), fLoadRetestData('ABIDE2', sNormed='HistNormed'),
             fLoadRetestData('ABIDE1'), dABIDE2Data]

    # load up the models
    dModels = {
        # # 1st best models
        'Model1': fLoadModels('Dense', 'combined', 'basc064', 43)[0],
        'Model2': fLoadModels('Dense', 'combined', 'basc122', 39)[0],
        'Model3': fLoadModels('Dense', 'combined', 'basc197', 15)[0],
        # # 2nd best models
        'Model4': fLoadModels('Dense', 'combined', 'basc064', 21)[0],
        'Model5': fLoadModels('Dense', 'combined', 'basc122', 7)[0],
        'Model6': fLoadModels('Dense', 'combined', 'basc197', 18)[0],
        # 3rd best models
        'Model7': fLoadModels('Dense', 'combined', 'basc064', 2)[0],
        'Model8': fLoadModels('Dense', 'combined', 'basc122', 2)[0],
        'Model9': fLoadModels('Dense', 'combined', 'basc197', 25)[0],
        # 4th best models
        'Model10': fLoadModels('Dense', 'combined', 'basc064', 31)[0],
        'Model11': fLoadModels('Dense', 'combined', 'basc122', 22)[0],
        'Model12': fLoadModels('Dense', 'combined', 'basc197', 6)[0],
        # 5th best models
        'Model13': fLoadModels('Dense', 'combined', 'basc064', 22)[0],
        'Model14': fLoadModels('Dense', 'combined', 'basc122', 15)[0],
        'Model15': fLoadModels('Dense', 'combined', 'basc197', 2)[0]
    }

    dAllModelResults={}
    i=0
    # Loop through each dataset
    #for dXData in lsData:
    # Reformat the tuning data to the right form
    dXData1=lsData[0]
    dFormattedXData1 = {
        'Model1': dXData1['BASC064'].drop([x for x in dXData1['BASC064'].columns if ('Age' in x)], axis=1),
        'Model2': dXData1['BASC122'].drop([x for x in dXData1['BASC122'].columns if ('Age' in x)], axis=1),
        'Model3': dXData1['BASC197'].drop([x for x in dXData1['BASC197'].columns if ('Age' in x)], axis=1),
        'Model4': dXData1['BASC064'].drop([x for x in dXData1['BASC064'].columns if ('Age' in x)], axis=1),
        'Model5': dXData1['BASC122'].drop([x for x in dXData1['BASC122'].columns if ('Age' in x)], axis=1),
        'Model6': dXData1['BASC197'].drop([x for x in dXData1['BASC197'].columns if ('Age' in x)], axis=1),
        'Model7': dXData1['BASC064'].drop([x for x in dXData1['BASC064'].columns if ('Age' in x)], axis=1),
        'Model8': dXData1['BASC122'].drop([x for x in dXData1['BASC122'].columns if ('Age' in x)], axis=1),
        'Model9': dXData1['BASC197'].drop([x for x in dXData1['BASC197'].columns if ('Age' in x)], axis=1),
        'Model10': dXData1['BASC064'].drop([x for x in dXData1['BASC064'].columns if ('Age' in x)], axis=1),
        'Model11': dXData1['BASC122'].drop([x for x in dXData1['BASC122'].columns if ('Age' in x)], axis=1),
        'Model12': dXData1['BASC197'].drop([x for x in dXData1['BASC197'].columns if ('Age' in x)], axis=1),
        'Model13': dXData1['BASC064'].drop([x for x in dXData1['BASC064'].columns if ('Age' in x)], axis=1),
        'Model14': dXData1['BASC122'].drop([x for x in dXData1['BASC122'].columns if ('Age' in x)], axis=1),
        'Model15': dXData1['BASC197'].drop([x for x in dXData1['BASC197'].columns if ('Age' in x)], axis=1),
    }

    # Reformat the tuning data to the right form
    dXData2=lsData[1]
    dFormattedXData2 = {
        'Model1': dXData2['BASC064'].drop([x for x in dXData2['BASC064'].columns if ('Age' in x)], axis=1),
        'Model2': dXData2['BASC122'].drop([x for x in dXData2['BASC122'].columns if ('Age' in x)], axis=1),
        'Model3': dXData2['BASC197'].drop([x for x in dXData2['BASC197'].columns if ('Age' in x)], axis=1),
        'Model4': dXData2['BASC064'].drop([x for x in dXData2['BASC064'].columns if ('Age' in x)], axis=1),
        'Model5': dXData2['BASC122'].drop([x for x in dXData2['BASC122'].columns if ('Age' in x)], axis=1),
        'Model6': dXData2['BASC197'].drop([x for x in dXData2['BASC197'].columns if ('Age' in x)], axis=1),
        'Model7': dXData2['BASC064'].drop([x for x in dXData2['BASC064'].columns if ('Age' in x)], axis=1),
        'Model8': dXData2['BASC122'].drop([x for x in dXData2['BASC122'].columns if ('Age' in x)], axis=1),
        'Model9': dXData2['BASC197'].drop([x for x in dXData2['BASC197'].columns if ('Age' in x)], axis=1),
        'Model10': dXData2['BASC064'].drop([x for x in dXData2['BASC064'].columns if ('Age' in x)], axis=1),
        'Model11': dXData2['BASC122'].drop([x for x in dXData2['BASC122'].columns if ('Age' in x)], axis=1),
        'Model12': dXData2['BASC197'].drop([x for x in dXData2['BASC197'].columns if ('Age' in x)], axis=1),
        'Model13': dXData2['BASC064'].drop([x for x in dXData2['BASC064'].columns if ('Age' in x)], axis=1),
        'Model14': dXData2['BASC122'].drop([x for x in dXData2['BASC122'].columns if ('Age' in x)], axis=1),
        'Model15': dXData2['BASC197'].drop([x for x in dXData2['BASC197'].columns if ('Age' in x)], axis=1),
    }

    # Reformat the tuning data to the right form
    dXData3=lsData[2]
    dFormattedXData3 = {
        'Model1': dXData3['BASC064'].drop([x for x in dXData3['BASC064'].columns if ('Age' in x)], axis=1),
        'Model2': dXData3['BASC122'].drop([x for x in dXData3['BASC122'].columns if ('Age' in x)], axis=1),
        'Model3': dXData3['BASC197'].drop([x for x in dXData3['BASC197'].columns if ('Age' in x)], axis=1),
        'Model4': dXData3['BASC064'].drop([x for x in dXData3['BASC064'].columns if ('Age' in x)], axis=1),
        'Model5': dXData3['BASC122'].drop([x for x in dXData3['BASC122'].columns if ('Age' in x)], axis=1),
        'Model6': dXData3['BASC197'].drop([x for x in dXData3['BASC197'].columns if ('Age' in x)], axis=1),
        'Model7': dXData3['BASC064'].drop([x for x in dXData3['BASC064'].columns if ('Age' in x)], axis=1),
        'Model8': dXData3['BASC122'].drop([x for x in dXData3['BASC122'].columns if ('Age' in x)], axis=1),
        'Model9': dXData3['BASC197'].drop([x for x in dXData3['BASC197'].columns if ('Age' in x)], axis=1),
        'Model10': dXData3['BASC064'].drop([x for x in dXData3['BASC064'].columns if ('Age' in x)], axis=1),
        'Model11': dXData3['BASC122'].drop([x for x in dXData3['BASC122'].columns if ('Age' in x)], axis=1),
        'Model12': dXData3['BASC197'].drop([x for x in dXData3['BASC197'].columns if ('Age' in x)], axis=1),
        'Model13': dXData3['BASC064'].drop([x for x in dXData3['BASC064'].columns if ('Age' in x)], axis=1),
        'Model14': dXData3['BASC122'].drop([x for x in dXData3['BASC122'].columns if ('Age' in x)], axis=1),
        'Model15': dXData3['BASC197'].drop([x for x in dXData3['BASC197'].columns if ('Age' in x)], axis=1),
    }

    # Reformat the tuning data to the right form
    dXData4=lsData[3]
    dFormattedXData4 = {
        'Model1': dXData4['BASC064'].drop([x for x in dXData4['BASC064'].columns if ('Age' in x)], axis=1),
        'Model2': dXData4['BASC122'].drop([x for x in dXData4['BASC122'].columns if ('Age' in x)], axis=1),
        'Model3': dXData4['BASC197'].drop([x for x in dXData4['BASC197'].columns if ('Age' in x)], axis=1),
        'Model4': dXData4['BASC064'].drop([x for x in dXData4['BASC064'].columns if ('Age' in x)], axis=1),
        'Model5': dXData4['BASC122'].drop([x for x in dXData4['BASC122'].columns if ('Age' in x)], axis=1),
        'Model6': dXData4['BASC197'].drop([x for x in dXData4['BASC197'].columns if ('Age' in x)], axis=1),
        'Model7': dXData4['BASC064'].drop([x for x in dXData4['BASC064'].columns if ('Age' in x)], axis=1),
        'Model8': dXData4['BASC122'].drop([x for x in dXData4['BASC122'].columns if ('Age' in x)], axis=1),
        'Model9': dXData4['BASC197'].drop([x for x in dXData4['BASC197'].columns if ('Age' in x)], axis=1),
        'Model10': dXData4['BASC064'].drop([x for x in dXData4['BASC064'].columns if ('Age' in x)], axis=1),
        'Model11': dXData4['BASC122'].drop([x for x in dXData4['BASC122'].columns if ('Age' in x)], axis=1),
        'Model12': dXData4['BASC197'].drop([x for x in dXData4['BASC197'].columns if ('Age' in x)], axis=1),
        'Model13': dXData4['BASC064'].drop([x for x in dXData4['BASC064'].columns if ('Age' in x)], axis=1),
        'Model14': dXData4['BASC122'].drop([x for x in dXData4['BASC122'].columns if ('Age' in x)], axis=1),
        'Model15': dXData4['BASC197'].drop([x for x in dXData4['BASC197'].columns if ('Age' in x)], axis=1),
    }

    #dResults = fCalculatePerformance(dModels, dFormattedXData, dXData['Target'].values)
    dAllModelResults['TrainABIDE1Norm, TestABIDE2Norm'] = fTuneModelDiffData(dModels, dFormattedXData1,
                                            dFormattedXData2, dXData1['Target'].values, dXData2['Target'].values)
    dAllModelResults['TrainABIDE2Norm, TestABIDE1Norm'] = fTuneModelDiffData(dModels, dFormattedXData2,
                                            dFormattedXData1, dXData2['Target'].values, dXData1['Target'].values)
    dAllModelResults['TrainABIDE1, TestABIDE2'] = fTuneModelDiffData(dModels, dFormattedXData3, dFormattedXData4,
                                             dXData3['Target'].values, dXData4['Target'].values)
    dAllModelResults['TrainABIDE2, TestABIDE1'] = fTuneModelDiffData(dModels, dFormattedXData4, dFormattedXData3,
                                             dXData4['Target'].values, dXData3['Target'].values)
    #i+=1

        # for sKey in dResults.keys():
        #     print(f'{sKey}: {dResults[sKey]} AUROC')
