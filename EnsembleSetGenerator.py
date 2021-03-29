#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 24 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import pandas as pd
import pickle as pkl
from BioMarkerIdentification import fLoadPDData, fLoadModels

def fFetchNthBestShallow(cRandSearch, N=0):
    """
    Loads the Nth best shallow ML model given a random search object
        Note: N starts at 0 as the best model, 1 for the next best model, etc
    :param sPickledSearchLoc: filepath to pickled results file
    :param N: Nth best shallow model
    :return: The model class
    """
    # load the random search class and format it
    pdMLResults = pd.DataFrame(cRandSearch.cv_results_)
    pdMLResults.sort_values(by='rank_test_score', inplace=True)

    # fetch params for Nth best model
    dParamsNthBest = pdMLResults.loc[N, 'params']

    # generate the classifier
    cClassifier= cRandSearch.best_estimator_.set_params(**dParamsNthBest)

    return cClassifier

if '__main__' == __name__:
    # Load the data that will be used
    dXData, aYData = fLoadPDData()

    # List of the models and atlases to be used
    lsModels=['LinRidge', 'LinLasso', 'LinSVM', 'Dense', 'LSTM']
    lsAtlases=[
        'combined basc064',
        'combined basc122',
        'combined basc197',
        'combined craddock_scorr_mean',
        'combined power_2011'
    ]

    # Generate holder variables
    dModels={}
    dData={}

    # Load the list of top ten architectures
    dTopTen=pkl.load(open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                              '/TrainedModels/ISBIRerun/TopTens.p', 'rb'))

    # Loop through models and atlases to populate dict with
    for sModel in lsModels:
        for sAtlas in lsAtlases:
            for iRank in range(10):
                # Fetch the architecture number of the Nth best dense/LSTM
                if sModel=='Dense' or sModel=='LSTM':
                    iModelNum=dTopTen[sModel][sAtlas.split(" ")[1]][iRank]
                else:
                    iModelNum=None

                # Fetch the model
                if sModel=='Dense' or sModel=='LSTM':
                    dModels.update({f'{sModel}_{sAtlas.replace(" ","_")}_Rank{iRank}':
                                    fLoadModels(sModel, 'combined', sAtlas.split(" ")[1], iModelNum=iModelNum)[0]})
                else:
                    dModels.update({f'{sModel}_{sAtlas.replace(" ", "_")}_Rank{iRank}':
                                        fFetchNthBestShallow(fLoadModels(sModel, 'combined',
                                                                         sAtlas.split(" ")[1])[0], iRank)})

                # Fetch the data
                if not sModel=='LSTM':
                    dData.update({f'{sModel}_{sAtlas.replace(" ","_")}_Rank{iRank}':
                                  dXData[sAtlas.split(" ")[1]].drop([x for x in
                                  dXData[sAtlas.split(" ")[1]].columns if (x.__contains__('Age'))], axis=1)})
                else:
                    dData.update({f'{sModel}_{sAtlas.replace(" ", "_")}_Rank{iRank}':
                                  dXData[sAtlas.split(" ")[1]]})

    # Store results
    sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles/EnsembleModelSet.p'
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles/EnsembleDataSet.p'
    pkl.dump(dModels, open(sModelPath, 'wb'))
    pkl.dump(dData, open(sDataPath, 'wb'))
