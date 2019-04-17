#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 08 Feb, 2019
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

# Initialize the path to the data and the indices to loop over
sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/2AtlasErrors'

dModels={
 'anatomical_basc064_basc122': '32',
 'basc064_basc122': '30',
 'anatomical_basc064_basc197': '25',
 'basc064_basc197': '23',
 'anatomical_basc064_craddock_scorr_mean': '02',
 'basc064_craddock_scorr_mean': '32',
 'anatomical_basc064_harvard_oxford_cort_prob_2mm': '02',
 'basc064_harvard_oxford_cort_prob_2mm': '47',
 'anatomical_basc064_msdl': '04',
 'basc064_msdl': '15',
 'anatomical_basc064_power_2011': '15',
 'basc064_power_2011': '18',
 'anatomical_basc122_basc197': '18',
 'basc122_basc197': '39',
 'anatomical_basc122_craddock_scorr_mean': '02',
 'basc122_craddock_scorr_mean': '22',
 'anatomical_basc122_harvard_oxford_cort_prob_2mm': '30',
 'basc122_harvard_oxford_cort_prob_2mm': '28',
 'anatomical_basc122_msdl': '36',
 'basc122_msdl': '33',
 'anatomical_basc122_power_2011': '15',
 'basc122_power_2011': '02',
 'anatomical_basc197_craddock_scorr_mean': '23',
 'basc197_craddock_scorr_mean': '36',
 'anatomical_basc197_harvard_oxford_cort_prob_2mm': '25',
 'basc197_harvard_oxford_cort_prob_2mm': '22',
 'anatomical_basc197_msdl': '15',
 'basc197_msdl': '15',
 'anatomical_basc197_power_2011': '25',
 'basc197_power_2011': '18',
 'anatomical_craddock_scorr_mean_harvard_oxford_cort_prob_2mm': '36',
 'craddock_scorr_mean_harvard_oxford_cort_prob_2mm': '15',
 'anatomical_craddock_scorr_mean_msdl': '02',
 'craddock_scorr_mean_msdl': '18',
 'anatomical_craddock_scorr_mean_power_2011': '02',
 'craddock_scorr_mean_power_2011': '02',
 'anatomical_harvard_oxford_cort_prob_2mm_msdl': '13',
 'harvard_oxford_cort_prob_2mm_msdl': '10',
 'anatomical_harvard_oxford_cort_prob_2mm_power_2011': '02',
 'harvard_oxford_cort_prob_2mm_power_2011': '36',
 'anatomical_msdl_power_2011': '06',
 'msdl_power_2011': '15'
}

lsAtlases=list(dModels.keys())

# Initialize a dataframe to hold results
pdResults=pd.DataFrame(index=range(50), columns=lsAtlases)

for root, dirs, files in os.walk(sDataPath):
    files.sort()
    for file in files:
        if file.endswith('ROCScoreTest.p'):
            flTestPerf=pickle.load(open(os.path.join(sDataPath, file), 'rb'))
            iIteration=int(file[-16:-14])
            for iModel in range(42):
                if file.__contains__('anatomical') and lsAtlases[iModel].__contains__('anatomical'):
                    if file.__contains__(lsAtlases[iModel]):
                        pdResults.iloc[iIteration, iModel] = flTestPerf
                elif (not file.__contains__('anatomical')) and (not lsAtlases[iModel].__contains__('anatomical')):
                    if file.__contains__(lsAtlases[iModel]):
                        pdResults.iloc[iIteration, iModel] = flTestPerf


pdResults

pdResults=pdResults.replace(0.5, np.nan)

pdMeans=pdResults.mean()
pdStd=pdResults.std()

lsModelTags = ['basc064', 'basc122', 'basc197', 'craddock', 'harvard', 'msdl', 'power']
pdMeanPerf = pd.DataFrame(index=lsModelTags, columns=lsModelTags)
pdStdDev = pd.DataFrame(index=lsModelTags, columns=lsModelTags)


for iRow in range(len(lsModelTags)):
    for iCol in range(len(lsModelTags)):

        if iCol>iRow:
            sRow=lsModelTags[iRow]
            sCol=lsModelTags[iCol]
            for sKey in dModels.keys():
                if sKey.__contains__(sRow) and sKey.__contains__(sCol) and (not sKey.__contains__('anatomical')):
                    sKeyToUse=sKey
            flMean=pdMeans[sKeyToUse]
            flStd=pdStd[sKeyToUse]
            pdMeanPerf.iloc[iRow, iCol] = flMean
            pdStdDev.iloc[iRow, iCol] = flStd

        elif iRow>iCol:
            sRow=lsModelTags[iRow]
            sCol=lsModelTags[iCol]
            for sKey in dModels.keys():
                if sKey.__contains__(sRow) and sKey.__contains__(sCol) and sKey.__contains__('anatomical'):
                    sKeyToUse=sKey


            flMean=pdMeans[sKeyToUse]
            flStd=pdStd[sKeyToUse]
            pdMeanPerf.iloc[iRow, iCol] = flMean
            pdStdDev.iloc[iRow, iCol] = flStd

        else:
            pdMeanPerf.iloc[iRow, iCol]=np.nan
            pdStdDev.iloc[iRow, iCol] = np.nan
