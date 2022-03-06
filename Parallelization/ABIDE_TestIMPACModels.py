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

import pandas as pd
import os
import copy
import keras as k
import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedShuffleSplit
from BioMarkerIdentification import fLoadModels


def fLoadABIDE_Data(lsPaths=['/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC64_FormattedData.csv',
    '/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC122_FormattedData.csv',
    '/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC197_FormattedData.csv']):
    """loads preprocessed ABIDE data

    Args:
        lsPaths (list, optional): list of paths to processed data. Defaults to:
        ['/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC64_FormattedData.csv',
        '/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC122_FormattedData.csv',
        '/archive/bioinformatics/DLLab/CooperMellema/src/IMPAC_processing/BASC197_FormattedData.csv'].

    Returns:
        dict: dictionary of data
    """
    dData={}
    for sPath in lsPaths:
        sKey = f"BASC{int(sPath.split('BASC')[1].split('_')[0]):03}"
        dData.update({sKey: pd.read_csv(sPath, index_col=0)})
        dData[sKey]=dData[sKey].sort_values(by=['ABIDE', 'subject', 'session', 'run']).drop_duplicates('subject')
    return dData

def fTuneModel(cModel_orig, sModel, dDat, aTarget, iAbide):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    cModel=copy.deepcopy(cModel_orig)

    # Perform a cross-validation experiment
    cSplitter=StratifiedShuffleSplit(n_splits=10, random_state=0)

    # train on 1/3, validate on 2/3
    i=0
    dResults={}
    for idxTest, idxTrain in cSplitter.split(dDat[sModel].values, aTarget):
        sSavePath=f'/archive/bioinformatics/DLLab/CooperMellema/results/Autism/ABIDE_Tuned/ABIDE{int(iAbide)}/{sModel}_CV{i}'
        if not os.path.isfile(sSavePath):
            aXTrain = np.expand_dims(np.expand_dims(dDat[sModel].values[idxTrain], axis=1), axis=3)
            aXTest = np.expand_dims(np.expand_dims(dDat[sModel].values[idxTest], axis=1), axis=3)
            aYTrain = aTarget[idxTrain]
            aYTest = aTarget[idxTest]

            # identical fit parameters to original data
            kerStopping = k.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.01, patience=20,
                                                            restore_best_weights=True)
            cModel.fit(x=aXTrain, y=aYTrain, validation_data=(aXTest, aYTest),
                                        epochs=500, callbacks=[kerStopping], batch_size=128)
            cModel.save(sSavePath)
            cModel.save(f'{sSavePath}.h5')
        else:
            import tensorflow as tf
            cModel=tf.keras.models.load_model(sSavePath)

        # compare to actual:
        flRawScore = skm.roc_auc_score(aYTest, cModel.predict(aXTest))

        print(f'{sModel}: Val. Perf: {flRawScore}')
        dResults[f'{sModel}_CV{i}'] = flRawScore
        i+=1

    # return results per model
    return dResults

def fCalculatePerformance(dModels, dData, bTune=False, aTarget=None, idxAbide=None, iAbide=None):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    for sModel in [sKey for sKey in dModels.keys() if not'Target' in sKey]:
        # predict the results
        if bTune:
            dData_copy = copy.deepcopy(dData)
            dData_copy[sModel] = dData_copy[sModel].loc[idxAbide]
            dResults.update(fTuneModel(dModels[sModel], sModel, dData_copy, aTarget[idxAbide], iAbide))
        else:
            aPredicted = dModels[sModel].predict_proba(np.expand_dims(np.expand_dims(dData[sModel].values, axis=1), axis=3))
            dResults.update({sModel: np.squeeze(aPredicted)})

    # return results per model
    return dResults

def fMakeTable(pdResults, bLowMotion=False, iABIDE=1):
    if bLowMotion:
        pdResults=pdResults[~pdResults['HighMotion']]
    pdResults=pdResults[pdResults['ABIDE']==iABIDE]
    s=''
    lsAtlas=['&BASC (64 ROIs)&\n', '&BASC (122 ROIs)&\n', '&BASC (197 ROIs)&\n']
    for i in range(3):
        ls=[]
        s+=lsAtlas[i]
        for j in range(1,15,3):
            flScore=skm.roc_auc_score(pdResults['ASD'], pdResults[f'Model{j+i}'])*100
            ls.append(flScore)
            s+=f"{(flScore):.1f}&\n"
        s+=f'{np.array(ls).mean():.1f}$\pm${np.array(ls).std():.1f}\\\\\n'
    with open(f'ABIDE{iABIDE}{"_LowMotion" if bLowMotion else ""}.txt', 'w') as f:
        f.write(s)

if '__main__'==__name__:
    bTune=True
    # load up the data
    dABIDE = fLoadABIDE_Data()
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

    # Loop through each dataset
    # Reformat the data to the right form
    dData = {
        'Model1': dABIDE['BASC064'].iloc[:,11:].drop([x for x in dABIDE['BASC064'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model2': dABIDE['BASC122'].iloc[:,11:].drop([x for x in dABIDE['BASC122'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model3': dABIDE['BASC197'].iloc[:,11:].drop([x for x in dABIDE['BASC197'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model4': dABIDE['BASC064'].iloc[:,11:].drop([x for x in dABIDE['BASC064'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model5': dABIDE['BASC122'].iloc[:,11:].drop([x for x in dABIDE['BASC122'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model6': dABIDE['BASC197'].iloc[:,11:].drop([x for x in dABIDE['BASC197'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model7': dABIDE['BASC064'].iloc[:,11:].drop([x for x in dABIDE['BASC064'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model8': dABIDE['BASC122'].iloc[:,11:].drop([x for x in dABIDE['BASC122'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model9': dABIDE['BASC197'].iloc[:,11:].drop([x for x in dABIDE['BASC197'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model10': dABIDE['BASC064'].iloc[:,11:].drop([x for x in dABIDE['BASC064'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model11': dABIDE['BASC122'].iloc[:,11:].drop([x for x in dABIDE['BASC122'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model12': dABIDE['BASC197'].iloc[:,11:].drop([x for x in dABIDE['BASC197'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model13': dABIDE['BASC064'].iloc[:,11:].drop([x for x in dABIDE['BASC064'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model14': dABIDE['BASC122'].iloc[:,11:].drop([x for x in dABIDE['BASC122'].iloc[:,11:].columns if ('Age' in x)], axis=1),
        'Model15': dABIDE['BASC197'].iloc[:,11:].drop([x for x in dABIDE['BASC197'].iloc[:,11:].columns if ('Age' in x)], axis=1)
    }

    if bTune:
        dResults={}
        for iAbide in [1,2]:
            dResults[f'ABIDE{iAbide}'] = fCalculatePerformance(dModels, dData, bTune=True, aTarget=np.squeeze(np.array(dABIDE['BASC122']['ASD'].values)), idxAbide = (dABIDE['BASC122']['ABIDE']==iAbide).values, iAbide=iAbide)
        pdTemp = pd.DataFrame.from_dict(dResults)
        pdTemp['CV'] = [int(x.split('CV')[1]) for x in pdTemp.index]
        pdTemp=pdTemp.reset_index()
        pdTemp=pdTemp.rename({'index':'Model'}, axis=1)
        pdTemp['Model']=[int(x.split('Model')[1].split('_')[0]) for x in pdTemp['Model']]
        pdTemp.to_csv('/archive/bioinformatics/DLLab/CooperMellema/results/Autism/ABIDE_Tuned/all.csv')
    else:
        dResults = fCalculatePerformance(dModels, dData)
        pdResults=pd.DataFrame.from_dict(dResults)
        pdResults['ASD']=dABIDE['BASC122']['ASD'].values
        pdResults['ABIDE']=dABIDE['BASC122']['ABIDE'].values
        pdResults['HighMotion']=dABIDE['BASC122']['HighMotion'].values

