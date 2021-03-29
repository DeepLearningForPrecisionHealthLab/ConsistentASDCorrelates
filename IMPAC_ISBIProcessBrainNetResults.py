#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file processes the BrainNetCNN training results for the IMPAC Autism Project.
    
Originally developed for use in the IMPAC Autism Project
Created by Cooper Mellema on 04 Jan, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import numpy as np
import pickle
import sys
import os
import sklearn.metrics as skm


def fSeparateCVTestResults():
    """
    Separates out the test data and validation data for later use
    :return: NA, pickles CV1, 2, 3 validation, as well as test data
    """
    # Initialize Paths
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestDataPy2.pkl'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'\
                '/TrainedModels/ISBIRerun/BrainNetCNN/'

    # only pickle the files if they aren't already done
    if not os.path.isfile(sSavePath+'TrueTargetsTest.p') \
        and not os.path.isfile(sSavePath+'TrueTargetsCV1.p')\
        and not os.path.isfile(sSavePath+'TrueTargetsCV2.p')\
        and not os.path.isfile(sSavePath+'TrueTargetsCV3.p'):
        # Fetch the target data for both the Cross-Validation and test
        dData = pickle.load(open(sDataPath, 'rb'))
        aYData = dData['aYData']
        aYTest = dData['aYtest']

        # Reformat the data again
        aYData = np.float32(aYData)
        aYTest = np.float32(aYTest)

        iSplitSize = int(aYData.shape[0]/3)

        # Split the Data in the same way as in the 3x cross validation
        lsYVal =[[aYData[:iSplitSize,:]], # include only beginning
                 [aYData[iSplitSize:2*iSplitSize,:]], # include only middle
                 [aYData[2*iSplitSize:,:]] # include only end
                 ]

        # reshape the data
        aCV1=lsYVal[0][0]
        aCV1 = np.squeeze(aCV1, axis=1)
        pickle.dump(aCV1, open(sSavePath+'TrueTargetsCV1.p', 'wb'))

        aCV2 = lsYVal[1][0]
        aCV2 = np.squeeze(aCV2, axis=1)
        pickle.dump(aCV2, open(sSavePath + 'TrueTargetsCV2.p', 'wb'))

        aCV3 = lsYVal[2][0]
        aCV3 = np.squeeze(aCV3, axis=1)
        pickle.dump(aCV3, open(sSavePath + 'TrueTargetsCV3.p', 'wb'))

        aYTest = np.squeeze(aYTest, axis=1)
        pickle.dump(aYTest, open(sSavePath + 'TrueTargetsTest.p', 'wb'))

def fGetROCAUCFromResults(sPickledFile):
    """
    Takes the predictions of the BrainNetCNN and then uses them to generate ROC AUC metrics
    :param sPickledFile: The file location of the trained network predictions
    :return: the ROC AUC score in question
    """
    try:
        aResults = pickle.load(open(sPickledFile))
    except ValueError:
        print "Something went wrong in loading the Pickled File. Either the file name is incorrect," \
              "the file is corrupted, or something else went wrong. Try pickle.load(open("+str(sPickledFile)\
              +")) for more information"

    if "CrossVal0Predicted" in sPickledFile:
        iCV=1
    elif "CrossVal1Predicted" in sPickledFile:
        iCV=2
    elif "CrossVal2Predicted" in sPickledFile:
        iCV=3
    elif "FullModelPredicted" in sPickledFile:
        iCV=0
    else:
        print "skipping file " + sPickledFile
        flROCAUCScore = 0
        return flROCAUCScore

    if not iCV==0:
        aTarget = pickle.load(open('/project/bioinformatics/DLLab/Cooper/Code/'
                                   'AutismProject/Parallelization/TrainedModels'
                                   '/ISBIRerun/BrainNetCNN/TrueTargetsCV'+str(iCV)+'.p','rb'))
    else:
        aTarget = pickle.load(open('/project/bioinformatics/DLLab/Cooper/Code/'
                                   'AutismProject/Parallelization/TrainedModels'
                                   '/ISBIRerun/BrainNetCNN/TrueTargetsTest.p','rb'))

    #Don't generate ROC scores for unsuccesfull networks
    print sPickledFile
    #print aResults
    #print aTarget
    if not np.isnan(aResults).any():
        flROCScore = skm.roc_auc_score(aTarget, aResults)
    else:
        flROCScore = 0

    return flROCScore

def fFormatROCName(sFileName):
    if 'CrossVal' in sFileName:
        sCVNum = str(int(sFileName[-12])+1)
        sFileName = sFileName[:-20]
        sFileName = sFileName + 'ROCScoreCrossVal' + sCVNum +'.p'
    elif 'FullModel' in sFileName:
        sFileName = sFileName[:-20]
        sFileName = sFileName + 'ROCScoreTest.p'
    else:
        None
    return sFileName


if __name__ == '__main__':

    # first, generate the targets for the ROC AUC test if they don't exist
    fSeparateCVTestResults()

    # loop through the directory of results
    for sRoot, lsDirs, lsFiles in os.walk('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                          '/TrainedModels/ISBIRerun/BrainNetCNN'):
        for sFile in lsFiles:
            # take the predictions and make the ROC AUC scores
            if "Predicted" in sFile:
                sSaveAs = fFormatROCName(os.path.join(sRoot, sFile))
                #If the score file doesn't exist, make it
                if not os.path.isfile(sSaveAs):
                    flROCAUCScore = fGetROCAUCFromResults(os.path.join(sRoot,sFile))
                    print flROCAUCScore
                    if not flROCAUCScore==0:
                        pickle.dump(flROCAUCScore, open(sSaveAs, 'wb'))
