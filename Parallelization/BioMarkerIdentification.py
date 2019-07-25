#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file Generically has a class to run permutation feature importance testing

This file, if run alone, will perform the PFI testing on the top 5 TSE models in the IMPAC study

Originally developed for use in the ASD comparison project
Created by Cooper Mellema on 11 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

import os
import sys
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
import pickle as pkl
import dill
import sklearn.metrics as skm
import seaborn as sb
import pandas as pd
import numpy as np
import keras as k
import multiprocessing
import time

from joblib import Parallel, delayed
from joblib import parallel_backend
from joblib import wrap_non_picklable_objects
from IMPAC_DenseNetwork import read_config_file
from IMPAC_DenseNetwork import format_dict
from IMPAC_DenseNetwork import metrics_from_string
from IMPAC_DenseNetwork import get_optimizer_from_config
from IMPAC_DenseNetwork import add_layer_from_config
from IMPAC_LSTM import fLoadLSTM
from IMPAC_DenseNetwork import network_from_ini_2

def fPermuteFeature(iPermutationNum, sFeature, pdData):
    """
    randomly permutes a feature in the provided data
    :param iPermutaionNum: the number of this permutation (of a set)
    :param sFeature: the name of the feature to be permuted
    :param pdData: pandas dataframe of the features
    :return pdData: dataframe with the features permuted
    """
    # permute using a the repetition number as the random seed
    np.random.seed(iPermutationNum)
    pdData2 = pdData.copy()
    aData2 = np.random.permutation(pdData[sFeature].values)
    pdData2[sFeature] = aData2
    return pdData2

def fPerformTest(iPermutationNum, sFeature, pdData, sModelType, cModel, aActual, cOutputter=None):
    """
    Performs the predictions in the test
    :param iPermutationNum: number of the permutation
    :param sFeature: the name of the feature to be permuted
    :param pdData: dataframe with the features to be permuted
    :param sModelType: string, model type (Dense, LSTM, etc)
    :param cModel: the model objects
    :param aActual: array Ytrue
    :param cOutputter: class, outputter class for Neural networks (for speedup)
    :return:
    """
    pdPermutedData = fPermuteFeature(iPermutationNum, sFeature, pdData)
    aXPrimeData = fExpandDataframe(sModelType, pdPermutedData)
    if sModelType == 'Dense' or sModelType == 'LSTM':
        aPredicted = cOutputter([aXPrimeData])[0]
    elif hasattr(cModel, 'predict_proba'):
        aPredicted = cModel.predict_proba(aXPrimeData)[:, 0]
    else:
        aPredicted = cModel.predict(aXPrimeData)
    return skm.roc_auc_score(aActual, aPredicted)

def fTestModelNPermutations(nPermutations, cModel, sFeature, aActual, pdData, nParallel=1, cOutputter=None):
    """
    tests a feature in a model for nPermutations permutations
    :param nPermutaions: the number of Permutations to run
    :param cModel: the model objects
    :param sKey: the key for the DF of models
    :param sFeature: the name of the feature to be permuted
    :param aActual: array of true values
    :param pdData: dataframe with the features to be permuted
    :return: flFeatureImportance feature importance (float) as measured by base performance
    minus performance with the feature randomly permuted
    """
    # initialize variables and create baseline score
    sModelType=fRetreiveModelType(cModel)
    aXData=fExpandDataframe(sModelType, pdData)

    if sModelType=='Dense' or sModelType=='LSTM':
        aPredicted=cOutputter.predict(aXData)
    elif not hasattr(cModel, 'predict_proba'):
        aPredicted = cModel.predict(aXData)
    else:
        aPredicted = cModel.predict_proba(aXData)[:, 0]

    #Generate baseline score
    flRawScore = skm.roc_auc_score(aActual, aPredicted)

    print(f'   Feature {sFeature} being permuted.....')

    # for each permutation, randomly permute the original features, then predict with the
    # permuted features as inputs
    lsPermutedFeatues=[]
    for iPermutationNum in range(nPermutations):
        lsPermutedFeatues.append(fExpandDataframe(sModelType, fPermuteFeature(iPermutationNum, sFeature, pdData)))

    # make the predictions in parallel
    tic=time.clock()
    if sModelType=='Dense' or sModelType=='LSTM':
        lsPermutedPredictions=Parallel(n_jobs=nParallel)(delayed(cOutputter.predict)(x)for x in lsPermutedFeatues)
    else:
        lsPermutedPredictions=Parallel(n_jobs=nParallel)(delayed(cModel.predict)(x)for x in lsPermutedFeatues)
    toc=time.clock()
    print(f'For n_jobs={nParallel}: Time={toc-tic}')

    # put ROC AUC scores in list
    lsPermutedScores=[]
    for aPrediction in lsPermutedPredictions:
        lsPermutedScores.append(skm.roc_auc_score(aActual, aPrediction))

    # calculate feature importance (float) as measured by base performance
    # minus performance with the feature randomly permuted
    flFeatureImportance = flRawScore - np.mean(lsPermutedScores)

    return flFeatureImportance

def fPermuteAllFeatures(nPermutations, cModel, aActual, pdData):
    """
    tests each feature provided to a model by randomly permuting that feature nPermutations times
    :param nPermutaions: the number of Permutations to run
    :param dModel: the data frame containing the model objects
    :param sKey: the key for the DF of models
    :param aActual: array of true values
    :param pdData: dataframe with the features to be permuted
    :return: dFeaturePerformances: dictionary with feature importances (floats) with the
    feature name as the key. Feature importance is measured as baseline (unpermuted) performance
    minus performance with the feature randomly permuted
    """
    dFeaturePerformances = {}

    sModelType=fRetreiveModelType(cModel)
    print(f'Permuting {sModelType} model now')

    # make a compiled predictor for the Neural network
    if sModelType=='Dense' or sModelType=='LSTM':
        cPredictor=k.models.Model(inputs=cModel.inputs, outputs=cModel.outputs)
        cPredictor._make_predict_function()
    else:
        cPredictor=None

    # loop through each feature, running N permutations on each feature each time
    for sFeature in pdData.columns:
        flFeaturePerformance = fTestModelNPermutations(nPermutations, cModel, sFeature, aActual, pdData,
                                                       cOutputter=cPredictor)
        dFeaturePerformances.update({sFeature: flFeaturePerformance})

    return dFeaturePerformances

def fPermuteAllModels(nPermutations, aActual, dData, dModels):
    """
    does permutation tests to each of the models in dModels to determine feature importance
    :param nPermutaions: the number of Permutations to run
    :param aActual: array of true values
    :param pdData: dataframe with the features to be permuted
    :param dModels: dictionary containing the trained model objects, each with a .predict function
    :return: dFeaturePerformances: dictionary with feature importances (floats) with the
    feature name as the key. Feature importance is measured as baseline (unpermuted) performance
    minus performance with the feature randomly permuted
    """
    dFeatureImportanceByModel = {}

    # for each model in the dictionary, calculate the importance per feature via permutation
    for sModel in list(dModels.keys()):
        dFeatureImportance = fPermuteAllFeatures(nPermutations, dModels[sModel], aActual, dData[sModel])
        sDir=f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/AtlasResolutionComparison{nPermutations}Permutations'
        if not os.path.isdir(sDir):
            os.mkdir(sDir)
        pkl.dump(dFeatureImportance, open(os.path.join(sDir,f'{sModel}Importances.p'), 'wb'))
        dFeatureImportanceByModel.update({f'{sModel}': dFeatureImportance})

    return dFeatureImportanceByModel

def fPlotFeaturesByImportance(dFeatureImportanceByModel, sModel, sSavePath=None):
    """
    Plots features sorted by importance
    :param dFeatureImportanceByModel: dictionary of feature importances,
        {sModelType: dFeatureImportance}
    :param sFeature: string, Model name
    :param sSavePath: string, save location for fig
    :return: plot
    """
    # organize to pd dataframe, then sort data
    dFeatureImportance = dFeatureImportanceByModel[sModel]
    pdFeatureImportance = pd.DataFrame.from_dict(dFeatureImportance)
    pdFeatureImportance.sort_values(by='col1')

    # make bar plot
    pdFeatureImportance.plot(kind='bar')

    # save if location is provided
    if sSavePath is not None:
        plt.savefig(sSavePath)

def fLoadData(sNNType, iModelNum, sInputName, sSubInputName):
    # initialize paths
    sIni = f'{sNNType}_{iModelNum:02d}'
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels'

    # Path to data for TSE training
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestDataWithConfounds.p'

    # Load the data which was used to train the model
    # required for generating correct input size
    [dXData, dXTest, aYData, aYTest] = pkl.load(open(sDataPath, 'rb'))

    if not sSubInputName==None:
        aXData = dXData[sInputName][sSubInputName]
        aXTest = dXTest[sInputName][sSubInputName]
    else:
        aXData = dXData[sInputName]
        aXTest = dXTest[sInputName]

    # The required dimensions for the dense network is size
    # N x H x W x C, where N is the number of samples, C is
    # the number of channels in each sample, and, H and W are the
    # spatial dimensions for each sample.
    aXData = np.expand_dims(aXData, axis=1)
    aXData = np.expand_dims(aXData, axis=3)

    aXTest = np.expand_dims(aXTest, axis=1)
    aXTest = np.expand_dims(aXTest, axis=3)

    aXData = np.float32(aXData)
    aXTest = np.float32(aXTest)
    aYData = np.float32(aYData)
    aYTest = np.float32(aYTest)

    if sInputName=='connectivity':
        flCorrectConnectivity=36
    else:
        flCorrectConnectivity=0

    # initialize the shape of the input layer
    iDataShape=aXData[0,:].shape[1]-1-flCorrectConnectivity
    aDataShape=[1,iDataShape,1]

    return aDataShape

def fLoadNNModel(sNNType, sInputName, sSubInputName, iModelNum):
    #Doesnt load BrainNet Data

    #Initialize paths
    sIni = f'{sNNType}_{str(iModelNum).zfill(2)}'
    sIniPath = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/{sIni}.ini'
    # correct format for anatomical data alone
    if sSubInputName==None:
        sSelector=''
    else:
        sSelector=sSubInputName
    sWeightsPath = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun' \
        f'/{sNNType}/{sNNType}_{iModelNum:02d}_{sInputName}{sSelector}weights.h5'

    # Load data shape
    aDataShape = fLoadData(sNNType, iModelNum, sInputName, sSubInputName)

    # create the model architecture, then load the weights
    if sNNType=='LSTM':
        kmModel=fLoadLSTM(sInputName, str(iModelNum).zfill(2), sSubInputName=sSubInputName)
    else:
        kmModel = network_from_ini_2(sIniPath, aInputShape=aDataShape)
        kmModel.load_weights(sWeightsPath)

    # Return model and data
    return kmModel

def fLoadPDData():
    sData='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AllDataWithConfounds.p'
    #Dictionary that containes the whole dataset (train and test) in pd dataframe
    [dXData, aYData] = pkl.load(open(sData, 'rb'))
    return dXData, aYData

def fLoadShallowModel(sModelType, sModality, sAtlas):
    # reformat the anatomical selection procedure
    if sModality=='anatomy':
        sModality='anatomical'
        sAtlas='only'
    sShallowFile=f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/SortedShallowModel{sModality}{sAtlas}DF.p'
    if not os.path.isfile(sShallowFile):
        # load a pd Dataframe of all the models
        sShallowModel='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/ShallowModelsDict.p'
        sShallowModelDF='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/ShallowModelsDF.p'

        # only load dataframes once per session (takes more memory, but is much faster)
        if not (('dShallowModels' in globals()) and ('pdShallowModels' in globals())):
            global dShallowModels
            global pdShallowModels

            dShallowModels = pkl.load(open(sShallowModel, 'rb'))
            pdShallowModels = pkl.load(open(sShallowModelDF, 'rb'))

        #reorganize to match order of NN models
        dShallow={}
        dShallow2={}
        for sKey in dShallowModels.keys():
            if sKey.__contains__(sModality):
                dShallow.update({sKey: dShallowModels[sKey]})
                if sKey.__contains__(sAtlas):
                    dShallow2.update({sKey: dShallowModels[sKey]})
        pdShallow=pdShallowModels[pdShallowModels.index.str.match(sModality+'_')]
        pkl.dump(pdShallow, open(sShallowFile, 'wb'))
    else:
        pdShallow=pkl.load(open(sShallowFile, 'rb'))

    # select out the model
    cModel=pdShallow.loc[f'{sModality}_{sAtlas}', f'{sModelType}']

    return cModel

def fLoadModels(sModelType, sModality, sAtlas=None, iModelNum=None):
    if not iModelNum==None:
        cModel=fLoadNNModel(sModelType, sModality, sAtlas, iModelNum)
        lsSpecs=[sModelType, sModality, sAtlas, iModelNum]
    else:
        cModel=fLoadShallowModel(sModelType, sModality, sAtlas)
        lsSpecs=[sModelType, sModality, sAtlas]

    return cModel, lsSpecs

def fRetreiveModelType(cModel):
    """
    Retreives the kind of model being used
    """
    if not hasattr(cModel, 'layers'):
        return 'sklearn'
    elif 'LSTM' in f"{cModel.layers}":
        return 'LSTM'
    else:
        return 'Dense'

def fExpandDataframe(sModel, pdDataframe):
    """
    Reshapes the data for use in the different models
    :param sModel: 'LSTM', 'Dense', or 'sklearn'
    :param pdDataframe: pandas dataframe of the data
    :return: array of the data
    """
    if sModel=='Dense':
        aData=np.expand_dims(np.expand_dims(pdDataframe.values, axis=1),axis=3)
    elif sModel=='LSTM':
        aData=np.expand_dims(pdDataframe.values, axis=1)
    else:
        aData=pdDataframe.values

    return aData

if '__main__'==__name__:

    dXData, aYData = fLoadPDData()

    # NOTE, this uses the full model, not the output predictions
    # load up the models and the data
    dModels = {
               'Model1': fLoadModels('Dense', 'connectivity', 'basc064', 46)[0],
               'Model2': fLoadModels('Dense', 'connectivity', 'basc122', 2)[0],
               'Model3': fLoadModels('Dense', 'connectivity', 'basc197', 32)[0],
               'Model4': fLoadModels('LinRidge', 'connectivity', 'basc064')[0],
               'Model5': fLoadModels('LinRidge', 'connectivity', 'basc122')[0],
               'Model6': fLoadModels('LinRidge', 'connectivity', 'basc197')[0],
               'Model7': fLoadModels('Dense', 'combined', 'basc064', 43)[0],
               'Model8': fLoadModels('Dense', 'combined', 'basc122', 39)[0],
               'Model9': fLoadModels('Dense', 'combined', 'basc197', 15)[0],
               'Model10': fLoadModels('LinRidge', 'combined', 'basc064')[0],
               'Model11': fLoadModels('LinRidge', 'combined', 'basc122')[0],
               'Model12': fLoadModels('LinRidge', 'combined', 'basc197')[0],
               'Model13': fLoadModels('Dense', 'anatomy', iModelNum=44)[0],
               'Model14': fLoadModels('LinRidge', 'anatomy')[0]
    }

    # Reformat the data to the right form
    dFormattedXData = {
        'Model1': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model2': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model3': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model4': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model5': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model6': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1),
        'Model7': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('Age'))], axis=1),
        'Model8': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('Age'))], axis=1),
        'Model9': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('Age'))], axis=1),
        'Model10': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('Age'))], axis=1),
        'Model11': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('Age'))], axis=1),
        'Model12': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('Age'))], axis=1),
        'Model13': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('ROI')or
                                                                                    x.__contains__('Age'))], axis=1),
        'Model14': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('ROI')or
                                                                                    x.__contains__('Age'))], axis=1)
    }

    # Permute the models
    nPermutations = 1
    dFeatureImportanceByModel = fPermuteAllModels(nPermutations, aYData, dFormattedXData, dModels)

    # Save it
    sSaveDir=f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/'\
        f'AtlasResolutionComparison{nPermutations}Permutations'

    if not os.path.isdir(sSaveDir): os.mkdir(sSaveDir)

    pkl.dump(dFeatureImportanceByModel, open(os.path.join(
        sSaveDir,f'AllTSEFeatureImportances{nPermutations}Permutations.p'),'wb'))
