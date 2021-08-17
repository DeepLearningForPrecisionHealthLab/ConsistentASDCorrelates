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
import keras as k
import numpy as np
import sklearn.metrics as skm
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
        dData.update({f"BASC{int(sPath.split('BASC')[1].split('_')[0]):03}":
            pd.read_csv(sPath, index_col=0)})
    return dData

def fCalculatePerformance(dModels, dData):
    """
    Calculates
    :param dModels: the data frame containing the model objects and targets
    :return: dictionary of AUROC scores per model
    """
    dResults={}
    for sModel in [sKey for sKey in dModels.keys() if not'Target' in sKey]:
        # predict the results
        aPredicted = dModels[sModel].predict_proba(np.expand_dims(np.expand_dims(dData[sModel].values, axis=1), axis=3))
        dResults.update({sModel: np.squeeze(aPredicted)})

    # return results per model
    return dResults

if '__main__'==__name__:
    # load up the data
    dABIDE = fLoadABIDE_Data()
    raise NotImplementedError
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

    dResults = fCalculatePerformance(dModels, dData)
    pdResults=pd.concat([dABIDE['BASC122']['ASD'], pd.DataFrame.from_dict(dResults)], axis=1)
