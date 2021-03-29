#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" this file is intended to fetch the correct

This file fetches the cross validation and test results of dep learning networks
that have been trained on the IMPAC dataset
    
The types of models are:
    Dense feed-forward
    LSTM
    BrainNet Convolutional Neural Network
    
Originally developed for use in the IMPAC Autism project
Created by Cooper Mellema on 03 Jan, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import os
import pandas as pd
import numpy as np
import pickle

############initializations###########################

sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun/'
lsAtlases = ['basc064', 'basc122', 'basc197', 'craddock_scorr_mean', 'harvard_oxford_cort_prob_2mm', 'msdl', 'power_2011']
lsModalities = ['anatomy', 'connectivity', 'combined']
lsNetworks = ['Dense', 'LSTM', 'BrainNetCNN']
dResults={}
for sNetwork in lsNetworks:
    dResults[sNetwork]={}
    for sModality in lsModalities[1:]:
        dResults[sNetwork][sModality]={}

########################################################

def fFetch(sFile):
    """
    This function tries to fetch the specified file, but if it can't,
    it returns 0
    :param sFile: the file location (full path)
    :return: either the pickled file or 0 if unable to load with pickle
    """
    try:
        return pickle.load(open(sFile, 'rb'))
    except:
        try:
            return pickle.load(open(sFile, 'rb'), encoding='bytes')
        except:
            return 0

for sNetworkName in lsNetworks:
    sNetwork = sNetworkName
    sNetworkPath = os.path.join(sDataPath, sNetwork)
    #First, we loop over the possible atlases (listed above
    for sAtlas in lsAtlases:
        # then we loop over the possible modes (i.e. anatomical alone,
        # then connectivity matrices alone, then combined)
        for sModality in lsModalities:
            # initialize dataframes to hold results
            pdNetworkResults = pd.DataFrame(index=range(50), columns=['Avg CV ROC AUC', 'Test ROC AUC'])

            # Loop over each model tested
            for iModel in range(50):

                #format model number correctly for loading
                if iModel <10:
                    sModel = '0' + str(iModel)
                else:
                    sModel = str(iModel)

                sNetwork = sNetworkName + '_' + sModel

                # generate correct file name
                if sModality=='anatomy':
                    sNetwork = sNetwork+sModality+'ROCScore'
                else:
                    if not 'BrainNetCNN' in sNetwork:
                        sNetwork = sNetwork+sModality+sAtlas+'ROCScore'
                    elif sModality=='connectivity':
                        sNetwork = sNetwork + sAtlas + 'ROCScore'

                flNetworkAvg=0

                # fetch the average performance across 3 cross-validations
                for iCV in range(3):
                    iCV=iCV + 1
                    sCV=str(iCV)

                    flNetworkCV=fFetch(os.path.join(sNetworkPath, sNetwork) + 'CrossVal' + sCV + '.p')
                    flNetworkAvg = flNetworkAvg + flNetworkCV

                # Fetch the results of a specified model, modality, and atlas
                flNetworkAvg = (flNetworkAvg / (3.0))
                flNetworkTest = fFetch(os.path.join(sNetworkPath, sNetwork) + 'Test.p')

                pdNetworkResults.iloc[iModel]['Avg CV ROC AUC'] = flNetworkAvg
                pdNetworkResults.iloc[iModel]['Test ROC AUC'] = flNetworkTest

                #save the results in a dictionary
                if sModality=='anatomy':
                    dResults[sNetworkName].update({sModality: pdNetworkResults})
                else:
                    dResults[sNetworkName][sModality].update({sAtlas: pdNetworkResults})
                # Final dictionary is of the form:
                # dict[dense/lstm/BrainNet][anatomy/connectivity/combined][atlas(no key for anatomy)]
                # and each entry in this form is a pandas dataframe with row= model number,
                # col = average CV ROC AUC, test ROC AUC

# reformat everything into a pandas dataframe - initialization here
lsConn = ['connectivity ' + sAtlas for sAtlas in lsAtlases]
lsComb = ['combined ' + sAtlas for sAtlas in lsAtlases]

lsAllTested = ['anatomy'] + lsConn[:] + lsComb[:]

pdFinalResults = pd.DataFrame(index=lsAllTested, columns=lsNetworks)
pdFinalModelNumbers = pd.DataFrame(index=lsAllTested, columns=lsNetworks)

# fill the pandas dataframe with the test results of the models that had
# the highest performance by cross-validation

#Loop over network types
for sNetwork in lsNetworks:
    # Loop over the rows to be filled
    for sIndex in pdFinalResults.index:
        # fetch the top performing (by CV ROC AUC) TEST performance
        # anatomy only, due to anatomy being formatted differently
        if sIndex == 'anatomy':
            flMaxCV = dResults[sNetwork][sIndex]['Avg CV ROC AUC'].max()
            iLoc = dResults[sNetwork][sIndex]['Avg CV ROC AUC'].index[
                dResults[sNetwork][sIndex]['Avg CV ROC AUC'] == flMaxCV].tolist()[0]
            flTestVal = dResults[sNetwork][sIndex]['Test ROC AUC'][iLoc]

        # fetch the top performing (by CV ROC AUC) TEST performance
        # looping through all other modalities
        else:
            sPrefix, sAtlas = sIndex.split()
            flMaxCV = dResults[sNetwork][sPrefix][sAtlas]['Avg CV ROC AUC'].max()
            iLoc = dResults[sNetwork][sPrefix][sAtlas]['Avg CV ROC AUC'].index[
                dResults[sNetwork][sPrefix][sAtlas]['Avg CV ROC AUC'] == flMaxCV].tolist()[0]
            flTestVal = dResults[sNetwork][sPrefix][sAtlas]['Test ROC AUC'][iLoc]

        pdFinalResults.loc[sIndex][sNetwork] = flMaxCV
        pdFinalModelNumbers.loc[sIndex][sNetwork] = iLoc

pickle.dump(pdFinalResults, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                 '/TrainedModels/ISBIRerun/BestROCAUCSummary.p', 'wb'))
pickle.dump(pdFinalModelNumbers, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                 '/TrainedModels/ISBIRerun/BestModelNumbers.p', 'wb'))

# Fetch the top ten best architectures per model
dTopTen = {}
dTopTen['Dense'] = {}
dTopTen['LSTM'] = {}
for sAtlas in lsAtlases:
    dTopTen['Dense'].update({sAtlas: list(dResults['Dense']['combined'][sAtlas].sort_values('Avg CV ROC AUC',
                                                   ascending=False).index[0:5])})
    dTopTen['LSTM'].update({sAtlas: list(dResults['LSTM']['combined'][sAtlas].sort_values('Avg CV ROC AUC',
                                                   ascending=False).index[0:5])})

pickle.dump(dTopTen, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                 '/TrainedModels/ISBIRerun/TopTens.p', 'wb'))
