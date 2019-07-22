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
import sklearn.metrics as skm
import seaborn as sb
import pandas as pd
import numpy as np
import keras as k

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

def fTestModelNPermutations(nPermutations, dModel, sKey, sFeature, aActual, pdData):
    """
    tests a feature in a model for nPermutations permutations
    :param nPermutaions: the number of Permutations to run
    :param dModel: the data frame containing the model objects
    :param sKey: the key for the DF of models
    :param sFeature: the name of the feature to be permuted
    :param aActual: array of true values
    :param pdData: dataframe with the features to be permuted
    :return: flFeatureImportance feature importance (float) as measured by base performance
    minus performance with the feature randomly permuted
    """
    # initialize variables and create baseline score
    aXData = pdData.values
    aXData = np.expand_dims(aXData, axis=1)
    aXData = np.expand_dims(aXData, axis=3)
    aPredicted = dModel.loc[sKey, 'Model'].predict(aXData)
    try:
        aPredicted = dModel.loc[sKey,'Model'].predict_proba(pdData.values)[:,0]
    except:
        aPredicted = dModel.loc[sKey,'Model'].predict(pdData.values)
    flRawScore = skm.roc_auc_score(aActual, aPredicted)
    lsPermutedScores = []

    # for each permutation, randomly permute the original features, then predict with the
    # permuted features as inputs
    for iPermutationNum in range(nPermutations):
        pdPermutedData = fPermuteFeature(iPermutationNum, sFeature, pdData)
        aXPrimeData = pdPermutedData.values
        aXPrimeData = np.expand_dims(aXPrimeData, axis=1)
        aXPrimeData = np.expand_dims(aXPrimeData, axis=3)
        try:
            aPredicted = dModel.loc[sKey, 'Model'].predict_proba(aXPrimeData)[:,0]
        except:
            aPredicted = dModel.loc[sKey, 'Model'].predict(aXPrimeData)
        lsPermutedScores.append(skm.roc_auc_score(aActual, aPredicted))

    # calculate feature importance (float) as measured by base performance
    # minus performance with the feature randomly permuted
    flFeatureImportance = flRawScore - np.mean(lsPermutedScores)

    return flFeatureImportance

def fPermuteAllFeatures(nPermutations, pdModel, sKey, aActual, pdData):
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

    # for each feature, permute nPermutations times and return the value to a dictionary
    for sFeature in pdData.columns:
        flFeaturePerformance = fTestModelNPermutations(nPermutations, pdModel, sKey, sFeature, aActual, pdData)
        dFeaturePerformances.update({sFeature: flFeaturePerformance})

    return dFeaturePerformances

def fPermuteAllModels(nPermutations, aActual, pdData, pdModels):
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
    for sModel in list(pdModels.index):
        dFeatureImportance = fPermuteAllFeatures(nPermutations, pdModels, sModel, aActual, pdData)
        dFeatureImportanceByModel.update({'{}'.format(pdModels.loc[sModel, 'Model']): dFeatureImportance})

    return dFeatureImportanceByModel

def fPlotFeaturesByImportance(dFeatureImportanceByModel, sFeature, sSavePath=None):
    """
    Plots features sorted by importance
    :param dFeatureImportanceByModel: dictionary of feature importances,
        {sModelType: dFeatureImportance}
    :param sFeature: string, feature name
    :param sSavePath: string, save location for fig
    :return: plot
    """
    # organize to pd dataframe, then sort data
    dFeatureImportance = dFeatureImportanceByModel[sFeature]
    pdFeatureImportance = pd.DataFrame.from_dict(dFeatureImportance)
    pdFeatureImportance.sort_values(by='col1')

    # make bar plot
    pdFeatureImportance.plot(kind='bar')

    # save if location is provided
    if sSavePath is not none:
        plt.savefig(sSavePath)

def fLoadData(sNNType, iModelNum, sInputName, sSubInputName):
    # initialize paths
    sIni = f'{sNNType}_{iModelNum}'
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels'

    # Path to data for TSE training
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestDataWithConfounds.p'

    # Load the data which was used to train the model
    # required for generating correct input size
    [dXData, dXTest, aYData, aYTest] = pkl.load(open(sDataPath, 'rb'))

    aXData = dXData[sInputName][sSubInputName]
    aXTest = dXTest[sInputName][sSubInputName]

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

    # initialize the shape of the input layer
    iDataShape=aXData[0,:].shape[1]-1
    print(iDataShape)
    aDataShape=[1,iDataShape,1]

    return aDataShape

def fLoadNNModel(sNNType, sInputName, sSubInputName, iModelNum):
    #Doesnt load BrainNet Data

    #Initialize paths
    sIni = f'{sNNType}_{str(iModelNum).zfill(2)}'
    sIniPath = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/{sIni}.ini'
    sWeightsPath = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun/{sNNType}/{sNNType}_{iModelNum}_{sInputName}{sSubInputName}weights.h5'

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

    # load a pd Dataframe of all the models
    sShallowModel='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/ShallowModelsDict.p'
    sShallowModelDF='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/ShallowModelsDF.p'
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

    # select out the model
    cModel=pdShallow.loc[f'{sModality}_{sAtlas}', f'{sModelType}']

    return cModel

def fLoadModels(sModelType, sModality, sAtlas, iModelNum=None):
    if not iModelNum==None:
        cModel=fLoadNNModel(sModelType, sModality, sAtlas, iModelNum)
        lsSpecs=[sModelType, sModality, sAtlas, iModelNum]
    else:
        cModel=fLoadShallowModel(sModelType, sModality, sAtlas)
        lsSpecs=[sModelType, sModality, sAtlas]

    return cModel, lsSpecs


if '__main__'==__name__:

    # load up the models and the data
    dModels = {1: [fLoadModels('Dense', 'combined', 'basc122', 39)],
               2: [fLoadModels('Dense', 'combined', 'basc197', 15)],
               3: [fLoadModels('LSTM', 'combined', 'basc122', 1)],
               4: [fLoadModels('LSTM', 'connectivity', 'basc122', 39)],
               5: [fLoadModels('LinRidge', 'combined', 'basc122')]
    }

    dXData, aYData = fLoadPDData()




    # # In[7]:
    #
    #
    # dShallow={}
    # dShallow2={}
    # for sKey in dShallowModels.keys():
    #     if sKey.__contains__(sInputName):
    #         dShallow.update({sKey: dShallowModels[sKey]})
    #         if sKey.__contains__(sSubInputName):
    #             dShallow2.update({sKey: dShallowModels[sKey]})
    # pdShallow=pdShallowModels[pdShallowModels.index.str.match(sInputName+'_')]
    #
    #
    # # In[8]:
    #
    #
    # pdShallow2=pdShallow.loc[f'{sInputName}_{sSubInputName}']
    # TestData=dXData[sSubInputName]
    # TestData=TestData.drop(['Age'], axis=1)
    #
    #
    # # In[9]:
    #
    #
    # sk.metrics.roc_auc_score(aYData, pdShallow2.iloc[5].predict(TestData.values))
    #
    #
    # # In[ ]:
    #
    #
    # #pdShallow2=pdShallow2.drop(['NaiveBayes'])
    # #pdShallow2
    # pdDeep=pd.DataFrame(index=[f'Dense{iModelNum}'], columns=['Model'])
    # pdDeep.loc[f'Dense{iModelNum}', 'Model']  =kmModel
    # pdDeep
    #
    #
    # # In[11]:
    #
    #
    # dXData[sSubInputName].columns.get_loc('Age')
    #
    #
    # # In[12]:
    #
    #
    # XDat=dXData[sSubInputName].drop('Age', axis=1)
    # XDat=XDat.fillna(0)
    # aYData=np.nan_to_num(aYData)
    # np.isnan(XDat.values).any()
    # np.isnan(aYData).any()
    # XDat
    #
    #
    # # In[13]:
    #
    #
    # # aPermuted = fPermuteFeature(5,'Site01',XDat)
    # # #XDat['Site01'].values
    # # XDat2=XDat.copy()
    # # XDat2['Site01']=aPermuted
    # # XDat2['Site01']==XDat['Site01']
    #
    #
    # # In[14]:
    #
    #
    #
    #
    # # In[15]:
    #
    #
    # aXData = XDat.values
    # aXData = np.expand_dims(aXData, axis=1)
    # aXData = np.expand_dims(aXData, axis=3)
    # aXData.shape
    # pdDeep.loc[f'Dense{iModelNum}', 'Model'].predict(aXData)
    # print('        {}'.format(pdDeep.loc[f'Dense{iModelNum}', 'Model']))
    #
    #
    # # In[ ]:
    #
    #
    # nPermutations=200
    # dFeatureImportanceByModel = fPermuteAllModels(nPermutations, aYData, XDat, pdDeep)
    #
    #
    # # In[ ]:
    #
    #
    #
    #
    #
    # # In[ ]:
    #
    #
    # dFeatureImportanceByModel['<keras.engine.sequential.Sequential object at 0x2aab33ee8160>']
    #
    #
    # # In[ ]:
    #
    #
    # sFeature='<keras.engine.sequential.Sequential object at 0x2aab33ee8160>'
    # dFeatureImportance=dFeatureImportanceByModel[sFeature]
    #
    #
    # # In[ ]:
    #
    #
    # pdFeatureImportance=pd.DataFrame.from_dict(dFeatureImportance, orient='index')
    # pdFeatureImportance=pdFeatureImportance.sort_values(by=[0], ascending=False)
    # pdFeatureImportance.head(20).plot(kind='bar')
    #
    #
    # # In[ ]:
    #
    #
    # pdFeatureImportance.head()
    #
    #
    # # In[ ]:
    #
    #
    # #sPermTest1='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Dense47_2FeaturePermutations.p'
    # sPermTest1='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Dense{iModelNum}_25FeaturePermutations.p'
    # pkl.dump(pdFeatureImportance, open(sPermTest1, 'wb'))

