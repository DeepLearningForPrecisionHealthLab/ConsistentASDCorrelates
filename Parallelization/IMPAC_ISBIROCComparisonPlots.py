#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file Creates Comparative ROC Plots for different trained models in the IMPAC Autism Dataset
    Example:
    
Details
    Plots multiple ROC Plots of the highest performing models in different categories
    
Originally developed for use in the *** project
Created by Cooper Mellema on 04 Jan, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd
import pickle as pkl
import sys
import os

def fFetchPredictedResults(dModels):
    """

    :param dModels:
    :return:
    """
    # Fetch the X and Y Test data:
    sTrainDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
    dXData, dXTest, aYData, aYTest = pkl.load(open(sTrainDataPath, 'rb'))

    dPredicted = {}

    for sKey in dModels.keys():
        sModel = sKey
        sInput = dModels[sKey]

        if ('Dense' in sModel) or sModel=='LSTM' or sModel=='BrainNetCNN':
            sDataPath ='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun'
            if 'Dense' in sModel:
                sModel='Dense'
            sModelPath = os.path.join(sDataPath, sModel)

            pdBestModelNumbers = pkl.load(open(os.path.join(sDataPath, 'BestModelNumbers.p'), 'rb'))

            iModelNum = pdBestModelNumbers.loc[sInput][sModel]

            if iModelNum<10:
                sModelNum = '0' + str(iModelNum)
            else:
                sModelNum = str(iModelNum)

            print(sInput)
            if not sInput=='anatomy':
                sCombination, sAtlas = sInput.split()
            else:
                sCombination=sInput
                sAtlas=''

            if not sModel=='BrainNetCNN':
                sPredicted = os.path.join(sModelPath, sModel + '_' + sModelNum + sCombination + sAtlas +
                                          'PredictedResults.p')
            else:
                sPredicted = os.path.join(sModelPath, sModel + '_' + sModelNum + sAtlas + 'FullModelPredicted.p')

            try:
                aPredicted = pkl.load(open(sPredicted, 'rb'))
            except:
                aPredicted = pkl.load(open(sPredicted, 'rb'), encoding='bytes')

            dPredicted[sModel + ' ' + sInput] = aPredicted


        else:
            sDataPath = "/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Results/3_fold_cv_50_random_" \
                        "hyperparameter_initializations_random_search0924"

            sModelPath = os.path.join(sDataPath, sInput, sModel + '.p')

            skModel = pkl.load(open(sModelPath, 'rb'))

            if not sInput=='anatomy':
                lsInputSplit = sInput.split('_')
                sCombination = lsInputSplit[0]
                if len(lsInputSplit[1:]) > 1:
                    sAtlas = '_'.join(lsInputSplit[1:])
                else:
                    sAtlas = lsInputSplit[1]

                aXTest = dXTest[sCombination][sAtlas]
            else:
                aXTest = dXTest[sInput]

            try:
                aPredicted = np.array(skModel.predict_proba(aXTest))[:,1] #predict vs predict_proba
            except:
                aPredicted = np.array(skModel.predict(aXTest))


            if not len(aPredicted.shape)>1:
                aPredicted = np.expand_dims(aPredicted, axis=1)

            dPredicted[sModel + ' ' + sInput] = aPredicted

    return dPredicted

def fPlotMultiROC(dPredictions, sKey, aTrue, sSavePath):
    """

    :param aPredictions:
    :param aLabels:
    :param sSavePath:
    :return:
    """
    plt.style.use('seaborn-deep')
    plt.figure()

    for sType in dPredictions.keys():
        aPredicted = dPredictions[sType]
        aFPR, aTPR, aThresholds = skm.roc_curve(aTrue, aPredicted)
        flROCAUC = skm.roc_auc_score(aTrue, aPredicted)
        plt.plot(aFPR, aTPR, label=(sType + '; area: {0:.3f}'.format(flROCAUC)))

    # make a 'chance guessing' line
    plt.plot([0,1], [0,1], linestyle='--')

    #set plot variables
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve comparison for ' + sKey)
    plt.legend(loc='lower right')

    #save the figure
    plt.savefig(sSavePath + '.png', format='png')
    plt.show()

if __name__ == '__main__':
    #Make plots for ISBI

    # Compare ROCs for best dense models
    dModelComparison1 = {
        'Dense': 'anatomy',
        'Denser': 'connectivity basc122',
        'Densest': 'combined basc122'
    }

    #Compare ROCs across DL - best DL model across all imputs per modality
    dModelComparison2 = {
        'Dense': 'combined basc122',
        'LSTM': 'connectivity basc122',
        'BrainNetCNN': 'connectivity basc122'
    }

    #Compare ROCs across Nonlinear, Linear, and deep
    dModelComparison3 = {
        'NaiveBayes': 'connectivity_basc122',
        'XGBoost': 'combined_basc064',
        'ExRanTrees': 'combined_basc122',
        'LinRidge': 'combined_power_2011',
        'Dense': 'combined basc122',
        'BrainNetCNN': 'connectivity basc122'
    }

    #create a single, unified dictionary
    dAllComparisons = {
        'Dense Comparison': dModelComparison1,
        'Deep Comparison': dModelComparison2,
        'Broad Comparison': dModelComparison3
    }

    dPredictions={}
    sSaveDir = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun/'

    for sKey in dAllComparisons:
        dPredictions.update({sKey: fFetchPredictedResults(dAllComparisons[sKey])})
    for sKey in dAllComparisons:
        # Fetch the X and Y Test data:
        sTrainDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
        dXData, dXTest, aYData, aYTest = pkl.load(open(sTrainDataPath, 'rb'))

        fPlotMultiROC(dPredictions[sKey], sKey, aYTest, sSavePath=(sSaveDir + sKey))
