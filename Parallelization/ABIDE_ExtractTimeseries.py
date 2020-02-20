#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 14 Feb, 2020
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import nilearn as nil
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
import os
import glob
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# import provided programs from starting kit
sAutismStartingKit = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master"
sDataPath = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master/data"
sAutismCode = "/project/bioinformatics/DLLab/Cooper/Libraries/paris-saclay-cds-ramp-workflow-v0.2.0-41-g31d4037/paris-saclay-cds-ramp-workflow-31d4037"
sys.path.append(sAutismStartingKit)
sys.path.append(sAutismCode)
sys.path.append(sDataPath)

from problem import get_train_data, get_cv
from download_data import fetch_fmri_time_series


def fExtractTimeseriesMNIAligned(sNII_File, sAtlasPath):
    """
    Extracts a timeseries from a 4D nii file
    :param sNII_File: path to 4D nii file, in MNI space
    :param sAtlasPath: path to atlas
    :return: array (timeseries)
    """
    from nilearn.input_data import NiftiLabelsMasker
    cMasker = NiftiLabelsMasker(labels_img=sAtlasPath, standardize=True)
    return cMasker.fit_transform(sNII_File)

def fAtlasSetup():
    """
    Gets a dict of atlases and names used in the IMPAC project
    :return: dAtlases{sAtlas: sPathToAtlas}
    """
    dAtlases = {
        'BASC064': nil.datasets.fetch_atlas_basc_multiscale_2015()['scale064'],
        'BASC122': nil.datasets.fetch_atlas_basc_multiscale_2015()['scale122'],
        'BASC197': nil.datasets.fetch_atlas_basc_multiscale_2015()['scale197'],
    }
    return dAtlases

def fProcessAllSubs(sDir, dAtlases, sAtlas):
    """
    Extract the timeseries of all of the subjects in the provided directory with the provided atlas
    :param sDir: (str) Directory to process
    :param sAtlas: (str) path to the atlas where
    :return: n/a, saves the file in sDir/Timeseries/{sAtlas}/subjectx_timeseriesy_....npy
    """
    # get all associated motion corrected files in the root dir
    lsFiles = glob.glob(f'{sDir}/sub-*/ses-*/fmri/motion/*task-rest_run*EPInormMNI*smoothedintensitynormed_bold'
                        f'.nii.gz')
    lsFiles = lsFiles + glob.glob(f'{sDir}/sub-*/fmri/motion/*task-rest_run*EPInormMNI*smoothedintensitynormed'
                                  f'_bold.nii.gz')

    # loop through all the processed files
    for sFile in lsFiles:
        # extract the timeseries
        aData = fExtractTimeseriesMNIAligned(sFile, dAtlases[sAtlas])

        # set the new filename
        sSub = sFile.split('fmri')[0].split('/')[-2]
        sScanName = sFile.split('/')[-1].split('_bold.nii')[0]
        sTimeDir = f'{sDir}/Timeseries/{sAtlas}'
        if not os.path.isdir(sTimeDir):
            os.makedirs(sTimeDir)
        sFile2 = f'{sTimeDir}/{sSub}_{sAtlas}_Timeseries_{sScanName}'
        print(sFile2)

        # save the extracted data
        try:
            np.save(sFile2, aData)
        except:
            print(f'Could not save file for {sFile2}')

# This function was included in the IMPAC autism starting set, I altered it to grab the data from the correct folder
def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(os.path.join('/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master',
                                              subject_filename), header=None).values
                     for subject_filename in fmri_filenames])

# This class was included in the IMPAC autism starting set
class FeatureExtractor(BaseEstimator, TransformerMixin):
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

        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind=ConnectivityMetric, vectorize=ConnectivityVector))# Vectorize =
            # false for connectivity matrix

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

def fLoadBASCTSE_Transformers():
    """
    From original IMPAC data, load the data and fit the TSE models
    """
    # import last data and labels
    RawData, RawLabels = get_train_data(path="/project/bioinformatics/DLLab/Cooper/Jupyter_notebooks/autism-master/autism-master")

    #Set labels for later
    fMRIData = RawData
    fMRILabels = RawLabels

    # Here we select only the columns of fMRI_select=1 (1=fMRI present, 0=fMRI absent)
    fMRIData = RawData[[col for col in RawData.columns if col.startswith('fmri')]]
    GoodStudiesIndex = np.where(RawData.fmri_select != 0)
    fMRIData = fMRIData.iloc[GoodStudiesIndex[0]]
    fMRILabels = fMRILabels[GoodStudiesIndex[0]]

    fMRIData = fMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
    fMRIData = fMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well

    # set paths for feature extractor
    sys.path.append('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/')
    # Select out the names of the atlases used
    fMRIAtlases = list(fMRIData.columns)
    for i in range(len(fMRIAtlases)):
        fMRIAtlases[i] = fMRIAtlases[i][5:]

    # build feature extractor for each atlas
    dTSEs = {}
    for sAtlasName in fMRIAtlases:
        if 'basc' in sAtlasName:
            cTSE = ConnectivityMeasure(kind='tangent', vectorize=True)
            dTSEs[sAtlasName] = cTSE.fit([np.loadtxt(f'/project/bioinformatics/DLLab/STUDIES/Autism_IMPAC/autism-master{sFile[1:]}',
                                         delimiter=',') for sFile in fMRIData[f'fmri_{sAtlasName}'].values], fMRILabels)

    # return dict of fitted TSE models
    return dTSEs

if __name__ == '__main__':
    # set locations of where data is held
    lsRoots = [
        '/project/bioinformatics/DLLab/STUDIES/ABIDE1/RawDataBIDS/NYU/Derivatives/DLLabPipeline',
        '/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/GroupBySite/ABIDEII-NYU_1/Derivatives/DLLabPipeline',
        '/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/GroupBySite/ABIDEII-NYU_2/Derivatives/DLLabPipeline'
    ]

    dAtlases = fAtlasSetup()

    # For each location, loop through and find all appropriate scans
    bExtractTimeseries=False
    if bExtractTimeseries:
        for sRoot in lsRoots:
            for sAtlas in dAtlases.keys():
                fProcessAllSubs(sRoot, dAtlases, sAtlas)

    # Fit TSE model on original data
    dTSEs = fLoadBASCTSE_Transformers()
    cScaler = StandardScaler()

    for sRoot in lsRoots:
        for sAtlas in dTSEs.keys():
            # get list of files
            lsFiles = glob.glob(f'{sRoot}/Timeseries/{sAtlas.upper()}/*.npy')
            for sFile in lsFiles:
                lsData = [cScaler.fit_transform(np.load(sFile))]
                aResults = dTSEs[sAtlas].transform(lsData)
                np.save(f'{sRoot}/IMPAC_TSE/{sAtlas.upper()}/{sFile.split("/")[-1].replace("Timeseries","TSE")}',
                        aResults)

    # Then fit TSE model for all appropriate scans
    for sRoot in lsRoots:
        for sAtlas in dAtlases.keys():
            fProcessAllSubs(sRoot, dAtlases, sAtlas)
