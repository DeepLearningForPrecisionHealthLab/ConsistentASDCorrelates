#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file generates multiple FC matrices for use
in the IMPAC autism analysis

Metrics used:

    
Originally developed for use in the *** project
Created by Cooper Mellema on 08 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(42)

# Set up paths
sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
sProjectIdentification = "AutismProject"
sImagesPath = os.path.join(sProjectRootDirectory, sProjectIdentification, "Images")
sAutismStartingKit = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master"
sDataPath = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master/data"
sAutismCode = "/project/bioinformatics/DLLab/Cooper/Libraries/paris-saclay-cds-ramp-workflow-v0.2.0-41-g31d4037/paris-saclay-cds-ramp-workflow-31d4037"
sys.path.append(sAutismStartingKit)
sys.path.append(sAutismCode)
sys.path.append(sDataPath)

# import last few modules that were in the path added above (sAutismCode)
from problem import get_train_data, get_cv
from download_data import fetch_fmri_time_series

# This function was included in the IMPAC autism starting set, I altered it to grab the data from the correct folder
def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(os.path.join('/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master',
                                              subject_filename), header=None).values
                     for subject_filename in fmri_filenames])

# This class was included in the IMPAC autism starting set
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, **ConnectArgOptions):
        """
        make a transformer which will load the time series and compute the
        connectome matrix
        """
        ConnectArgs = {
            'ConnectivityMetric': 'tangent',
            'ConnectivityVector': True
        }
        ConnectArgs.update(ConnectArgOptions)
        ConnectivityMetric = ConnectArgs['ConnectivityMetric']
        ConnectivityVector = ConnectArgs['ConnectivityVector']

        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind=ConnectivityMetric, vectorize=ConnectivityVector))# Vectorize = false for connectivity matrix

    def fit(self, X_df, y, sAtlasName):
        # get only the time series for the atlas with name=sAtlasName
        fmri_filenames = X_df['fmri_' + sAtlasName]
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df, sAtlasName):
        fmri_filenames = X_df['fmri_' + sAtlasName]
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        if len(X_connectome.shape)==3:
            X_connectome = X_connectome.reshape((X_connectome.shape[0], X_connectome.shape[1]*X_connectome.shape[2]))
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]

        return X_connectome

def fFetchFMRIData(fMRIDataNames, sFile,
                   sDataPath='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics',
                             **ConnectArgOptions):
    """
    if a pickled set of the Connectivity data is available, it loads it,
    otherwise, it calculates the connectivity for every atlas, saves them all
    in a single dictionary, and then pickles it
    :param fMRIDataNames:
    :param sFile: string, name of file to be stored
    :param sDataPath: string, path to save data
    :return: dict of connectivity per atlas
    """
    if os.path.isfile(os.path.join(sDataPath, sFile + '.p')):

        dTotalConnectivity = pd.read_pickle(os.path.join(sDataPath, sFile + '.p'))
        return dTotalConnectivity

    else:
        def fGetConnectivityData(fMRIDataNames, fMRILabels, sAtlasName):

            PreExtractedFMRI=FeatureExtractor(**ConnectArgOptions)
            PreExtractedFMRI.fit(fMRIDataNames, fMRILabels, sAtlasName)
            Connectivity=PreExtractedFMRI.transform(fMRIDataNames, sAtlasName)
            return Connectivity

        # get the connectivity for each atlas
        dTotalConnectivity = {}
        for sAtlas in fMRIAtlases:
            print('Computing Connectivity for ' + sAtlas)
            pdConnectivity = fGetConnectivityData(fMRIData, fMRILabels, sAtlas)
            pdConnectivity.columns = [sAtlas + "_" + str(col) for col in pdConnectivity.columns]
            dTotalConnectivity[sAtlas] = pdConnectivity

        pickle.dump(dTotalConnectivity, open(sDataPath + '/' + sFile + '.p', 'wb'))
        return dTotalConnectivity

if "__main__"==__name__:

    # Set location
    # sRoot = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject'
    # lsAllDataWithConfounds = pickle.load(open(os.path.join(sRoot, 'AllDataWithConfounds.p'), 'rb'))
    # lsAtlases=list(lsAllDataWithConfounds[0].keys())

    # Set connectivity parameters
    dCorr={'ConnectivityMetric': 'correlation'}
    dPartialCorr={'ConnectivityMetric': 'partial correlation'}
    dCov={'ConnectivityMetric': 'covariance'}
    dPrecision={'ConnectivityMetric': 'precision'}

    # make into list to iterate over
    lsConnOptions=[
        dCorr,
        dPartialCorr,
        dCov,
        dPrecision
    ]

    ###############  fMRI  ###############
    # Load the Raw data
    RawData, RawLabels = get_train_data(path="/project/bioinformatics/DLLab/Cooper/Jupyter_notebooks/autism-master/autism-master")
    fMRIData = RawData
    fMRILabels = RawLabels

    # Here we select only the columns of fMRI_select=1 (1=fMRI present, 0=fMRI absent)
    fMRIData = RawData[[col for col in RawData.columns if col.startswith('fmri')]]
    GoodStudiesIndex = np.where(RawData.fmri_select != 0)
    fMRIData = fMRIData.iloc[GoodStudiesIndex[0]]
    fMRILabels = fMRILabels[GoodStudiesIndex[0]]

    # Discard unnecessary QA columns
    fMRIData = fMRIData.drop('fmri_select', axis=1)
    fMRIData = fMRIData.drop('fmri_motions', axis=1)

    # Select out the names of the atlases used
    fMRIAtlases = list(fMRIData.columns)
    for i in range(len(fMRIAtlases)):
        fMRIAtlases[i] = fMRIAtlases[i][5:]

    dDataByConnType={}
    # Iterate over each connectivity option, calculating the connectivity each time
    for dConnOption in lsConnOptions:
        sTag=dConnOption[list(dConnOption.keys())[0]].title().replace(" ", "")
        dTotalConnectivity = fFetchFMRIData(fMRIData, f'{sTag.capitalize()}ConnectivityData', **dConnOption,
                                            ConnectivityVector=True)
        dDataByConnType.update({sTag: dTotalConnectivity})