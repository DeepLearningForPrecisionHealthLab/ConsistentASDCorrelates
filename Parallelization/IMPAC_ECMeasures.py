#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file contains functions to compute measures of causality between fMRI time
    signals in order to determine effective causality between signals

This file has methods for the following:
    -Granger Causality
    -Mutual information
    -Structured equation modelling
    -Dynamic Causal Modelling
    
Includes use of the following toolboxes:
    Tapas toolbox
    IDTxL toolbox
    OpenMX toolbox

As the IMPAC project comes with a partial pipeline, We use the base FeatureExtractor
class form their project
    
Originally developed for use in the IMPAC ASD identification project
Created by Cooper Mellema on 06 Dec, 2018
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
from sklearn.base import BaseEstimator as skBaseEstimator
from sklearn.base import TransformerMixin as skTransformerMixin
from sklearn.preprocessing import FunctionTransformer as skFunctionTransformer
from nilearn.connectome import ConnectivityMeasure as nilConnectivityMeasure
from sklearn.pipeline import make_pipeline as skPipeline

# Add additional required paths
sAutismStartingKit = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master"
sDataPath = "/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
sAutismCode = "/project/bioinformatics/DLLab/Cooper/Libraries/paris-saclay-cds-ramp-workflow" \
              "-v0.2.0-41-g31d4037/paris-saclay-cds-ramp-workflow-31d4037"
sys.path.append(sAutismStartingKit)
sys.path.append(sAutismCode)
sys.path.append(sDataPath)

# import last few modules that were in the path added above (sAutismCode)
from problem import get_train_data, get_cv
from download_data import fetch_fmri_time_series

# The native feature extractor included in the IMPAC starting kit
# As of yet, not supported, but working on trying to integrate my methods
# into this pipeline
class cFeatureExtractor(skBaseEstimator, skTransformerMixin):
    """
    Extracts the FMRI timeseries
    This class was included in the IMPAC autism starting set
    """

    def __init__(self, **ConnectArgOptions):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        ConnectArgs = {
            'ConnectivityMetric': 'tangent',
            'ConnectivityVector': True
        }
        ConnectArgs.update(ConnectArgOptions)
        ConnectivityMetric = ConnectArgs['ConnectivityMetric']
        ConnectivityVector = ConnectArgs['ConnectivityVector']

        self.transformer_fmri = skPipeline(
            skFunctionTransformer(func=_load_fmri, validate=False),
            nilConnectivityMeasure(kind=ConnectivityMetric, vectorize=ConnectivityVector))

    def fit(self, X_df, y, sAtlasName):
        # get only the time series for the atlas with name=sAtlasName
        fmri_filenames = X_df['fmri_' + sAtlasName]
        a=self.transformer_fmri.fit(fmri_filenames, y)
        return self, a

    def transform(self, X_df, sAtlasName):
        fmri_filenames = X_df['fmri_' + sAtlasName]
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        if len(X_connectome.shape) == 3:
            X_connectome = X_connectome.reshape((X_connectome.shape[0], X_connectome.shape[1] * X_connectome.shape[2]))
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]

        return X_connectome

def fFetchRawFMRIData(bQualityAssurance=True):
    """ Fetches the raw FMRI data
    :param bQualityAssurance:
    :return: Raw FMRI data in a pandas dataframe and the corresponding labels (ASD vs HC)
    """
    pdRawData, aRawLabels = get_train_data(
        path="/project/bioinformatics/DLLab/Cooper/Jupyter_notebooks/autism-master/autism-master")
    if bQualityAssurance:

        # select the fMRI and sMRI Data Separately
        pdFMRIData = pdRawData[[col for col in pdRawData.columns if col.startswith('fmri')]]
        pdSMRIData = pdRawData[[col for col in pdRawData.columns if col.startswith('anat')]]

        # Drop the columns where the quality assurance metric denotes an unsatisfactory study
        #
        # For sMRI:
        # Reselect only the trials that were acceptable (2) or good (1) while throwing out the
        # (0) no MPRAGE available and (3) MPRAGE quality is unsatisfactory studies (subject's worth
        # of data)
        #
        # For fMRI:
        # Reselect only the trials that have available fMRI data (1) while throwing out the (0),
        # no fMRI data available

        aFMRIGoodStudiesIndex = np.where(pdRawData.fmri_select != 0)
        aSMRIGoodStudiesIndex = np.where(pdRawData.anatomy_select != (0 or 3))
        aCombinedGoodStudiesIndex = np.intersect1d(aFMRIGoodStudiesIndex, aSMRIGoodStudiesIndex)

        pdSMRIData = pdSMRIData.iloc[aCombinedGoodStudiesIndex]
        pdFMRIData = pdFMRIData.iloc[aCombinedGoodStudiesIndex]
        pdFMRIData = pdFMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
        pdFMRIData = pdFMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well

        return pdFMRIData, aRawLabels

    else:
        pdFMRIData = pdRawData[[col for col in pdRawData.columns if col.startswith('fmri')]]
        return pdFMRIData, aRawLabels

def fFetchFMRIFileNames(pdFMRIData):
    """ Fetches the names of the FMRI atlases and returns them

    :param pdFMRIData: the pandas dataframe of the data
    :return: names of the fmri atlases
    """
    # Select out the names of the atlases used
    lsFMRIAtlases = list(pdFMRIData.columns)
    for i in range(len(lsFMRIAtlases)):
        lsFMRIAtlases[i] = lsFMRIAtlases[i][5:]

    return lsFMRIAtlases

def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas.
    This function was included in the IMPAC autism starting set, I altered it to grab the data from the correct folder"""
    return np.array([pd.read_csv(os.path.join('/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master',
                    subject_filename), header=None).values for subject_filename in fmri_filenames])

def fFetchFMRIData(sFileName, lsFMRIAtlases, pdFMRIData, pdFMRILabels, **ConnectArgOptions):
    """Imports the timeseries per atlas
    if a pickled set of the Connectivity data is available, it loads it,
    otherwise, it calculates the connectivity for every atlas, saves them all
    in a single dictionary, and then pickles it
    :param sFileName: The file name of the time series (existing or what will be saved)
    :param aFMRIAtlases: Array of atlas names
    :param ConnectArgOptions: Options to pass to cFeatureExtractor
    :return: np array of the timeseries
    """
    if os.path.isfile(os.path.join(sDataPath, sFileName)+'.p'):

        dTimeSeriesByAtlas = pd.read_pickle(os.path.join(sDataPath, sFileName + '.p'))
        return dTimeSeriesByAtlas
    else:
        dTimeSeriesByAtlas = {}
        for dirs, root, files in os.walk('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                         '/TimeseriesFilenames'):
            for file in files:
                print('Fetching Time Series for ' + file[5:])
                pdFiles=pkl.load(open(os.path.join(dirs, file), 'rb'))
                dTimeSeriesByAtlas[file[5:-2]] = _load_fmri(pdFiles)
                pkl.dump(dTimeSeriesByAtlas, open(os.path.join(sDataPath, sFileName)+'.p', 'wb'))
        return dTimeSeriesByAtlas


if __name__ == '__main__':
    """ when run, this segment will process the data into a dictionary
    with each key being an atlas, and each element being an np array
    of the timeseries of each ROI in that atlas
    """
    pdFMRIData, aFMRILabels = fFetchRawFMRIData()
    lsFMRIAtlases = fFetchFMRIFileNames(pdFMRIData)
    dTimeSeriesByAtlas = fFetchFMRIData('ROITimeseries', lsFMRIAtlases, pdFMRIData, aFMRILabels)

    # The dictionary is formatted as follows:
    # dTimeSeriesByAtlas{Atlas Name: [pt number(0 to 915)] [timepoint, roi number]}
    # Timepoints, labelled above, is the signal per timestep. I.E, the value at
    # dTimeSeriesByAtlas['basc_064'][0][0,0] is the value of the signal for patient
    # zero at time t=0, ROI zero, using the basc atlas with 64 parcellations.
    # dTimeSeriesByAtlas['basc_122'][1][10,12] would be the value of the signal for patient
    # 1 at time t=10, ROI 12, using the basc atlas with 122 parcellations.
    # The number of time-points per patient is NOT the same
    # The data has NOT been split into train/test yet
