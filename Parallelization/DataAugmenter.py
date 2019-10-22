#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 09 Sep, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import os
import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nilearn.connectome as nic
import seaborn as sns
import pickle as pkl
from collections import OrderedDict as od
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from IMPAC_DenseNetwork import fRunDenseNetOnInput_v2
np.random.seed(42)


class cDataAugmenter:
    def __init__(self, dXTimeseriesData, sDecomp='PCA', dYData=None, dXAdditionalData=None):
        """
        :param dXTimeseriesData: (ordered dict) {Subject: np.array(nTimepoints x nROIs) or (1 x nConnMetric)}
        :param sDecomp: PCA, ICA, or PLS-DA
        :param dYData:(ordered dict) {Subject: 1 or 0 (class label)}
        :param dXAdditionalData:
            NOTE: This should be pre-normalized
        """
        # set data
        self.dXData = dXTimeseriesData
        if dXAdditionalData is not None:
            self.dXAdditionalData = dXAdditionalData

        # initialize decomposition method
        self.sDecomp = sDecomp
        if sDecomp == 'PCA':
            cDimReducer = PCA()
        elif sDecomp == 'ICA':
            cDimReducer = FastICA()
        elif sDecomp == 'PLS-DA':
            cDimReducer = PLSRegression(n_components=2)#dXTimeseriesData[list(dXTimeseriesData.keys())[0]].shape[1])

        # Fit dimension reducer
        self.fFitDimReducer(cDimReducer, dXTimeseriesData, dYData)
        self.fPopulateLatentSpace(dXTimeseriesData)

    def fFitDimReducer (self, cDimReducer, dXTimeseriesData, dYData=None):
        """
        Fits transformer to latent space with data
        :param cDimReducer: the dimensionality reduction object
        :param dXTimeseriesData: ordered dictionary of timeseries data
        :param dYData: ordered dictionary of targets for timeseries data
        :return: NA populates:
            self.cSubspace (fitted dimensionality reduction object)
        """
        self.dXData = dXTimeseriesData
        if self.sDecomp=='PCA' or self.sDecomp=='ICA':
            self.cSubspace = cDimReducer.fit(np.concatenate([dXTimeseriesData[sKey] for sKey in dXTimeseriesData.keys()]))
        elif self.sDecomp=='PLS-DA':
            aX = np.concatenate([dXTimeseriesData[sKey] for sKey in dXTimeseriesData.keys()])
            dY=od()
            dY.update({sKey:
                           np.concatenate((np.ones((dXTimeseriesData[sKey].shape[0],1)),
                                          np.zeros((dXTimeseriesData[sKey].shape[0],1))), axis=1) if dYData[sKey]==1
                           else
                           np.concatenate((np.zeros((dXTimeseriesData[sKey].shape[0], 1)),
                                          np.ones((dXTimeseriesData[sKey].shape[0], 1))), axis=1) for sKey in dYData.keys()
                       })
            #dY.update({sKey: (np.array([[1,0]]) if dYData[sKey]==1 else np.array([[0,1]])) for sKey in dYData.keys()})
            aY = np.concatenate([dY[sKey] for sKey in dY.keys()], axis=0)
            self.cSubspace = cDimReducer.fit(aX, aY)
            self.cSubspace.inverse_transform=self._fInverseTransform

    def _fInverseTransform(self, aReduced):
        aProjectedToRealSpace = np.dot(aReduced, self.cSubspace.components)

        return aProjectedToRealSpace

    def fPopulateLatentSpace(self, dXTimeseriesData):
        """
        Populates latent space with data
        :param dXTimeseriesData: ordered dictionary of timeseries data
        :return: NA populates:
            self.dXData_LatentSpace
            self.aXData_LatentSpace
            self.aXData
        """
        # dXData in latent space
        self.dXData_LatentSpace = od()
        self.dXData_LatentSpace.update({sKey: self.cSubspace.transform(dXTimeseriesData[sKey]) for sKey in dXTimeseriesData.keys()})

        self.aXData = self.fMakeArrayFromDict(dXTimeseriesData)
        self.aXData_LatentSpace = self.fMakeArrayFromDict(self.dXData_LatentSpace)

    def fMakeArrayFromDict(self, dXData):
        """
        Converts dict of timeseries to array to be further processed
        :param dXData: dict of timeseries data (in timesereis space or latent space)
        :return: aXData: array of the above
        """
        # make array to draw from, cropping all timeseries to shortest timeseries
        iMinTime = min([dXData[sKey].shape[0] for sKey in dXData.keys()])
        iFeatures = self.dXData_LatentSpace[list(self.dXData_LatentSpace.keys())[0]].shape[1]
        iSamples = len(self.dXData_LatentSpace.keys())

        # populate array from dict
        aXData = np.zeros((iMinTime, iFeatures, iSamples))
        for iSub, sKey in enumerate(dXData.keys()):
            aXData[:, :, iSub] = self.dXData_LatentSpace[sKey][:iMinTime, :]

        return aXData

    def fGenerator(self, sNoiseTo='finaltimeseries'):
        """
        Generator function that generates new random combination of other timeseries
        :return: a New timeseries using a random combination of the other timeseries
        """
        # ensure reproducability by setting seed
        iGeneration = 0
        np.random.seed(iGeneration)
        while True:
            if sNoiseTo == 'finaltimeseries':
                # pull a random timeseries
                aData = self.fRandomPull()
                # add noise if option is set
                aData = self.fAddNoise(aData)
            elif sNoiseTo == 'components':
                aData = self.fRandomPull(bNoise=True)
            elif sNoiseTo == 'both':
                aData = self.fRandomPull(bNoise=True)
                aData = self.fAddNoise(aData)
            else:
                aData = self.fRandomPull()

            yield aData # nRois x nTimepoints
            iGeneration += 1

    def fAddNoise(self, aArray):
        """
        Adds gaussian noise with SNR of 0.35
        :param aArray: array to add noise to
        :return: aArray with noise added
        """
        # set mean SNR to 0.35, from:
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0077089
        iNoiseSize = np.mean(np.abs(aArray))*0.35
        aNoise = np.random.normal(0, iNoiseSize, aArray.shape)
        return aArray+aNoise

    def fRandomPull(self, sMethod='gaussian', bNoise=False):
        """
        Pulls random combination of PC timeseries
        :param sMethod= gaussian or avg_N
        :param
        :return: a New timeseries using a random combination of the other timeseries
        """
        # select random weightings

        if sMethod=='gaussian':
            # random gaussian combo
            aRand = np.random.randn(len(self.dXData_LatentSpace.keys()))
            aRand = aRand/np.mean(aRand)
            aNew = self.aXData_LatentSpace*np.expand_dims(np.expand_dims(aRand, 0), 0)
            aNew = np.mean(aNew, axis=2)
            # if adding noise to combination directly,
            if bNoise:
                aNew = self.fAddNoise(aNew)

        elif 'avg' in sMethod:
            # random average of N
            iAvg=int(sMethod.split('_')[1])
            aRand = np.random.choice(list(range(self.aXData_LatentSpace.shape[2])), iAvg, replace=False)
            aNew = np.concatenate([self.aXData_LatentSpace[:, :, aRand[iN]] for iN in range(iAvg)], axis=2)
            aNew = np.mean(aNew, axis=2)

        # project to real space again
        return self.cSubspace.inverse_transform(aNew)

def fTriVect(aDat):
    """"
    Flattens an array of conn matrices into upper triangular vectors
    :param aDat: array of data of form nSub x nROI x nROI
    :return: aNewDat: array of data of form nSub x nFeatures
    """
    #if just a single conn matrix is passed,
    if len(aDat.shape) == 2:
        aDat = np.expand_dims(aDat, axis=0)

    aNewDat = np.zeros((aDat.shape[0], int(aDat.shape[1] * (aDat.shape[2] + 1) / 2)))

    for i in range(aDat.shape[0]):
        aNewDat[i, :] = aDat[i, :, :][np.triu_indices(aDat.shape[1])]
    return aNewDat

def fSoftmax(aDecisionFunction):
    """
    Performs softmax classification given a decision function
    :param aDecisionFunction: array of the outputs of the decision function
    :return: aProbabilites
    """
    aProbabilities = np.exp(aDecisionFunction) / np.sum(np.exp(aDecisionFunction))
    return aProbabilities

def fLoadSubjects(lsSubjects):
    """
    Loads a dict of timeseries from a list of subjects
    :param lsSubjects: list of subjects (numerical)
    :return: dSubjects (dict) dict of {subject: timeseries}
    """
    dSubjects=od()
    for iSubject in lsSubjects:
        sSubjectPath = f'{sRoot}/{iSubject}/run_1/{iSubject}_task-Rest_confounds.csv'
        try:
            pdControl = pd.read_csv(sSubjectPath, header=None, index_col=None)
            cScaler = StandardScaler()
            dSubjects.update({f'{iSubject}': cScaler.fit_transform(pdControl.values)})
        except:
            pass
    return dSubjects

def fPlotTSE(dSubjects, lsSubjects, sSaveLoc = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                            '/DataAugmentation/Images'):
    """
    Plots the tangent connectivity maps for a set of subjects in lsSubjects
    :param dSubjects: dict of TSE maps per subject
    :param lsSubjects: list of subjects to plot
    :param sSaveLoc: path to save location
    :return: saves in save location
    """
    # Set up mask for plotting
    aMask = np.zeros((dSubjects[lsSubjects[0]].shape[1], dSubjects[lsSubjects[0]].shape[1]))
    aMask[np.triu_indices_from(aMask, k=1)] = True
    if not os.path.isdir(sSaveLoc):
        os.mkdir(sSaveLoc)

    for i in range(len(lsSubjects)):
        sns.heatmap(dSubjects[lsSubjects[i]], mask=aMask, cmap='YlGnBu_r',
                    vmax=0.5, vmin=-0.5, square=True, cbar_kws={"shrink": .5})
        plt.title(f'Tangent Connectivity for Subject {lsSubjects[i]}')
        plt.xlabel('ROI')
        plt.ylabel('ROI')
        plt.savefig(f'{sSaveLoc}/TangentConnectivity_Subject{lsSubjects[i]}')
        plt.show()


if __name__ == '__main__':
    """
    Runs a brief example augmentation with ASD and Control subjects
    """
    #######
    # Set parameters for augmentation
    lsSamples = [500, 1000, 5000, 10000, 25000, 50000]#[500, 1000, 5000, 10000, 25000, 50000]
    lsDecomp = ['PCA', 'ICA']#['PCA', 'ICA','PLS-DA']
    lsNoiseTo = ['None', 'finaltimeseries','components','both']#['None', 'finaltimeseries','components','both']
    iSplits = 3
    lsCombos = list(itertools.product(*[lsSamples, lsDecomp, lsNoiseTo]))
    #lsCombos.insert(0, (0, 'None', 'None')) #(a control to insert)
    iMaxEpochs=300
    os.nice(5)
    pdAugmentationResults = pd.DataFrame(
        index=range(len(lsCombos)),
        columns=[
            'Augmentation Method',
            'Noise in timeseries',
            'Noise in components',
            'Number of additional samples',
            'Which',
            'Validation Train Acc',
            'Validation Train std. error',
            'Validation Acc',
            'Validation Acc std. error',
            'Validation AUC ROC',
            'Validation AUC ROC std. error',
            'Final Train Acc',
            'Final Test AUC ROC',
            'Generalization Error (Val-Te ROC AUC)'
    ])
    #######
    for iAugExperiment, (iSamples, sDecomp, sNoiseTo) in enumerate(lsCombos):
        sSavePath=f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Augmented/' \
            f'{sDecomp}Decomposition_{sNoiseTo.title()}Noise_{iSamples}xAugmented'
        pdAugmentationResults.loc[iAugExperiment]['Augmentation Method']= sDecomp
        pdAugmentationResults.loc[iAugExperiment]['Noise in timeseries'] = 1 if (sNoiseTo=='both' or
                                                                             sNoiseTo=='finaltimeseries') else 0
        pdAugmentationResults.loc[iAugExperiment]['Noise in components'] = 1 if (sNoiseTo=='both' or
                                                                             sNoiseTo=='components') else 0
        pdAugmentationResults.loc[iAugExperiment]['Number of additional samples'] = 2*iSamples
        pdAugmentationResults.loc[iAugExperiment]['Which'] = 'Augmented' if sDecomp!='None' else 'Standard'

        # load data
        sRoot = '/project/bioinformatics/DLLab/STUDIES/Autism_IMPAC/autism-master/data/fmri/msdl'
        pdSubjects = pd.read_csv('/project/bioinformatics/DLLab/STUDIES/Autism_IMPAC/autism-master/data'
                                           '/participants.csv')
        lsASD = list(pdSubjects[pdSubjects['asd'] == 1]['subject_id'].values)
        lsControls = list(pdSubjects[pdSubjects['asd'] == 0]['subject_id'].values)

        # Tr Te split
        lsASDTe = lsASD[int(len(lsASD)*0.8):]
        lsASD = lsASD[:int(len(lsASD)*0.8)]
        lsControlsTe = lsControls[int(len(lsControls)*0.8):]
        lsControls = lsControls[:int(len(lsControls)*0.8)]

        # Load control subjects
        dControls = fLoadSubjects(lsControls)
        dControlsTe = fLoadSubjects(lsControlsTe)

        # Load ASD Subjects
        dASD = fLoadSubjects(lsASD)
        dASDTe = fLoadSubjects(lsASDTe)

        # make dicts of all data
        dAll = od(**dControls, **dASD)
        dAllTe = od(**dControlsTe, **dASDTe)
        aYAll = np.concatenate((np.zeros((len(dControls.keys()))), np.ones((len(dASD.keys())))), axis=0)
        aYAllTe = np.concatenate((np.zeros((len(dControlsTe.keys()))), np.ones((len(dASDTe.keys())))), axis=0)

        # Calculate TSE Metric for training data WITHIN Tr set
        cAllConn = nic.ConnectivityMeasure(kind='tangent')
        cAllConn.fit([dAll[sKey] for sKey in dAll.keys()])

        # Generate TSE measures for Original and test data
        print('Calculating TSE')
        cScaler = StandardScaler()
        aAllTSE = fTriVect(cAllConn.transform([cScaler.fit_transform(dAll[sKey]) for sKey in dAll.keys()]))
        aAllTSETe = fTriVect(cAllConn.transform([cScaler.fit_transform(dAllTe[sKey]) for sKey in dAllTe.keys()]))

        if sDecomp !='None':
            # make augmenter object
            print('Performing data decomposition')
            if sDecomp == 'PLS-DA':
                dYData = od()
                dYData.update({sKey: (0 if sKey in dControls.keys() else 1) for sKey in dAll})
                cIMPAC_Augmenter = cDataAugmenter(dAll, sDecomp=sDecomp, dYData=dYData)
                cIMPAC_Control_Augmenter = cIMPAC_Augmenter
                cIMPAC_ASD_Augmenter = copy.deepcopy(cIMPAC_Augmenter)
                cIMPAC_Control_Augmenter.fPopulateLatentSpace(dControls)
                cIMPAC_Control_Augmenter.fPopulateLatentSpace(dASD)
            else:
                cIMPAC_Control_Augmenter = cDataAugmenter(dControls, sDecomp=sDecomp)
                cIMPAC_ASD_Augmenter = cDataAugmenter(dASD, sDecomp=sDecomp)

            # make generator
            print('Producing generator object')
            fControlGenerator = cIMPAC_Control_Augmenter.fGenerator(sNoiseTo=sNoiseTo)
            fASDGenerator = cIMPAC_ASD_Augmenter.fGenerator(sNoiseTo=sNoiseTo)

            # generate iSamples new datasets
            dNewControls = od()
            dNewASD = od()
            print('Generating new samples')
            for iSample in range(iSamples):
                dNewControls.update({f'Control_{iSample}': next(fControlGenerator)})
                dNewASD.update({f'ASD_{iSample}': next(fASDGenerator)})

            # make dict of all new Training data (X and Y)
            dAllAug = od(**dNewControls, **dNewASD)
            aYAllAug = np.concatenate((np.zeros((len(dNewControls.keys()))), np.ones((len(dNewASD.keys())))), axis=0)

            # Generate TSE measures for augmented data
            print('Calculating TSE for augmented original data')
            cScaler = StandardScaler()
            aAllAugTSE = fTriVect(cAllConn.transform([cScaler.fit_transform(dAllAug[sKey]) for sKey in dAllAug.keys()]))
        else:
            dAllAug = od()
            aYAllAug = []
            aAllAugTSE = []

        # Split data:
        cSplit = StratifiedShuffleSplit(n_splits=iSplits)
        aX = np.concatenate((np.array(list(dControls.keys())), np.array(list(dASD.keys()))))
        aY = np.concatenate((np.zeros(len(list(dControls.keys()))), np.ones(len(list(dASD.keys())))))
        cSplit.get_n_splits(aX, aY)

        # Set up way to iterate through data in splits
        lsAugIterator = []
        lsOrigIterator = []
        iRows = (aX.shape[0] + 2*iSamples)*iSplits
        # This will need to be changed when structural data is added...
        iColumns = int(dControls[list(dControls.keys())[0]].shape[1] * (dControls[list(dControls.keys())[0]].shape[1]+1)/2)
        aAugX = np.zeros((iRows, iColumns))
        aAugY = np.zeros((iRows))
        iStartingIndex = 0

        # Loop through splits
        iSplit = 1
        for lsTrainIndex, lsValIndex in cSplit.split(aX, aY):
            aXTr, aXVal = aX[lsTrainIndex], aX[lsValIndex]
            aYTr, aYVal = aY[lsTrainIndex], aY[lsValIndex]

            # Create necessary dicts for augmenter object
            dAllTr = od()
            dAllTr.update({x: dAll[x] for x in aXTr})
            dAllVal = od()
            dAllVal.update({x: dAll[x] for x in aXVal})
            dControlTr = od()
            dControlTr.update({x: dAllTr[x] for x in dAllTr if x in dControls.keys()})
            dASDTr = od()
            dASDTr.update({x: dAllTr[x] for x in dAllTr if x in dASD.keys()})


            # Calculate TSE Metric for training data WITHIN Tr set
            cConn = nic.ConnectivityMeasure(kind='tangent')
            cConn.fit([dAllTr[sKey] for sKey in dAllTr.keys()])

            cScaler = StandardScaler()
            aOrigTSETr = fTriVect(cConn.transform([cScaler.fit_transform(dAllTr[sKey]) for sKey in dAllTr.keys()]))
            aOrigTSEVal = fTriVect(cConn.transform([cScaler.fit_transform(dAllVal[sKey]) for sKey in dAllVal.keys()]))

            if sDecomp !='None':
                # make augmenter object
                print('Performing data decomposition')
                if sDecomp == 'PLS-DA':
                    dYData = od()
                    dYData.update({sKey: (0 if sKey in dControls.keys() else 1) for sKey in dAll})
                    cIMPAC_Augmenter = cDataAugmenter(dAll, sDecomp=sDecomp, dYData=dYData)
                    cIMPAC_Control_Augmenter = cIMPAC_Augmenter
                    cIMPAC_ASD_Augmenter = copy.deepcopy(cIMPAC_Augmenter)
                    cIMPAC_Control_Augmenter.fPopulateLatentSpace(dControls)
                    cIMPAC_Control_Augmenter.fPopulateLatentSpace(dASD)
                else:
                    cIMPAC_Control_Augmenter = cDataAugmenter(dControls, sDecomp=sDecomp)
                    cIMPAC_ASD_Augmenter = cDataAugmenter(dASD, sDecomp=sDecomp)

                # make generator
                print('Producing generator object')
                fControlGenerator = cIMPAC_Control_Augmenter.fGenerator(sNoiseTo=sNoiseTo)
                fASDGenerator = cIMPAC_ASD_Augmenter.fGenerator(sNoiseTo=sNoiseTo)

                # generate iSamples new datasets
                dNewControls = od()
                dNewASD = od()
                print('Generating new samples')
                for iSample in range(iSamples):
                    dNewControls.update({f'Control_{iSample}': next(fControlGenerator)})
                    dNewASD.update({f'ASD_{iSample}': next(fASDGenerator)})

                # make dict of all new Training data (X and Y)
                dAllTrAug = od(**dNewControls, **dNewASD)
                aYTrAug = np.concatenate((np.zeros((len(dNewControls.keys()))), np.ones((len(dNewASD.keys())))), axis=0)

                # Generate TSE measures for Original and augmented data
                print(f'Calculating TSE for split {iSplit}')
                aAugTSETr = fTriVect(cConn.transform([cScaler.fit_transform(dAllTrAug[sKey]) for sKey in dAllTrAug.keys()]))

                # make massive array to iterate over
                # Set training data and indices of training data
                aAugTr = np.concatenate((aOrigTSETr, aAugTSETr), axis=0)
                aTrIndices = slice(iStartingIndex, (iStartingIndex + aOrigTSETr.shape[0]))
                aTrAugIndices = slice(iStartingIndex, (iStartingIndex + aAugTr.shape[0]))
                aAugX[aTrAugIndices, :] = aAugTr
                aAugY[aTrAugIndices] = np.concatenate((aYTr, aYTrAug), axis=0)

                # set validation data and indices of validation data
                iStartingIndex += aAugTr.shape[0]
                aValIndices = slice(iStartingIndex, (iStartingIndex + aOrigTSEVal.shape[0]))
                aValAugIndices = aValIndices
                aAugX[aValAugIndices, :] = aOrigTSEVal
                aAugY[aValAugIndices] = aYVal
                iStartingIndex += aOrigTSEVal.shape[0]

            else:
                # make massive array to iterate over
                # Set training data and indices of training data
                aAugTr = aOrigTSETr
                aTrIndices = slice(iStartingIndex, (iStartingIndex + aOrigTSETr.shape[0]))
                aTrAugIndices = slice(iStartingIndex, (iStartingIndex + aOrigTSETr.shape[0]))
                aAugX[aTrAugIndices, :] = aAugTr
                aAugY[aTrAugIndices] = aYTr

                # set validation data and indices of validation data
                iStartingIndex += aOrigTSETr.shape[0]
                aValIndices = slice(iStartingIndex, (iStartingIndex + aOrigTSEVal.shape[0]))
                aValAugIndices = aValIndices
                aAugX[aValAugIndices, :] = aOrigTSEVal
                aAugY[aValAugIndices] = aYVal
                iStartingIndex += aOrigTSEVal.shape[0]

            # make list for iterator object
            lsOrigIterator.append((aTrIndices, aValIndices))
            lsAugIterator.append((aTrAugIndices, aValAugIndices))
            iSplit += 1

        # Create iterator object for augmented data
        cAugmentedIterator = iter(lsAugIterator)

        # Perform classification
        # cClassifier = RidgeClassifier()
        # dParamDistributions = {
        #     'alpha': 10 ** np.random.uniform(-5, 1, 100),
        #     'max_iter': np.random.uniform(1000, 100000, 100),
        # }
        # cAugRandomSearch = RandomizedSearchCV(
        #     cClassifier,
        #     dParamDistributions,
        #     cv=cAugmentedIterator,
        #     n_iter=50,
        #     n_jobs=1,
        #     verbose=0,
        #     scoring='roc_auc',
        #     refit=False
        # )
        # # Fit on each set of the cross-validated data
        # # cAugRandomSearch.fit(aAugX, aAugY)
        #
        # # refit where TSE was calculated across ALL data AND new augmented data for ALL is used
        # pdBestAugParams = pd.DataFrame(cAugRandomSearch.cv_results_)
        # dBestAugParams = pdBestAugParams.loc[pdBestAugParams['rank_test_score'] == 1]['params'].iloc[0]
        # cAugRefittedModel = RidgeClassifier(**dBestAugParams)
        if sDecomp!='None':
            lsAccTr = [None,None,None]
            lsAccVal = [None,None,None]
            lsROCAUCVal = [None,None,None]
            iCV=0
            for idxTr, idxTe in cAugmentedIterator:
                aXTr = aAugX[idxTr]
                aXTe = aAugX[idxTe]
                aYTr = aAugY[idxTr]
                aYTe = aAugY[idxTe]
                flROCAUC, cHistory = fRunDenseNetOnInput_v2('Dense', 8, sSavePath, iEpochs=iMaxEpochs,
                                        sIniPath=None, aXTr=aXTr, aYTr=aYTr, aXTe=aXTe, aYTe=aYTe, bCV=True,
                                                            sTag=f'CV{iCV}')
                idxBestCV = np.where(cHistory.history['val_acc']==np.max(cHistory.history['val_acc']))[0][0]
                lsAccTr[iCV] = cHistory.history['acc'][idxBestCV]
                lsAccVal[iCV] = cHistory.history['val_acc'][idxBestCV]
                lsROCAUCVal[iCV] = flROCAUC
                iCV+=1
            # cAugRefittedModel.fit(
            #     np.concatenate((aAllTSE, aAllAugTSE), axis=0), np.concatenate((aYAll, aYAllAug), axis=0)
            # )
            aXTr = np.concatenate((aAllTSE, aAllAugTSE), axis=0)
            aXTe = aAllTSETe
            aYTr = np.concatenate((aYAll, aYAllAug), axis=0)
            aYTe = aYAllTe
            flROCAUC, cHistory = fRunDenseNetOnInput_v2('Dense', 8, sSavePath, iEpochs=iMaxEpochs,
                                                        sIniPath=None, aXTr=aXTr, aYTr=aYTr, aXTe=aXTe, aYTe=aYTe,
                                                        bCV=False, sTag=f'Full')
            flAcc = cHistory.history['acc'][np.where(cHistory.history['acc']==np.max(cHistory.history['acc']))[0][0]]
        else:
            print('Have not built Non-augmented method yet')

        # Fill augmentation table
        pdAugmentationResults.loc[iAugExperiment]['Validation Train Acc'] = np.mean(np.array(lsAccTr))
        pdAugmentationResults.loc[iAugExperiment]['Validation Train std. error'] = np.std(np.array(lsAccTr))
        pdAugmentationResults.loc[iAugExperiment]['Validation Acc'] = np.mean(np.array(lsAccVal))
        pdAugmentationResults.loc[iAugExperiment]['Validation Acc std. error'] =np.std(np.array(lsAccVal))
        pdAugmentationResults.loc[iAugExperiment]['Validation AUC ROC'] = np.mean(np.array(lsROCAUCVal))
        pdAugmentationResults.loc[iAugExperiment]['Validation AUC ROC std. error'] = np.std(np.array(lsROCAUCVal))
        pdAugmentationResults.loc[iAugExperiment]['Final Train Acc'] = flAcc
        pdAugmentationResults.loc[iAugExperiment]['Final Test AUC ROC'] = flROCAUC
        pdAugmentationResults.loc[iAugExperiment]['Generalization Error (Val-Te ROC AUC)'] =\
            np.mean(np.array(lsROCAUCVal))-flROCAUC
    #        cAugRefittedModel.fit(aAllTSE, aYAll)
    #
    #     flAugScore = roc_auc_score(aYAllTe, fSoftmax(cAugRefittedModel.decision_function(aAllTSETe)))
    #     print(f'Score with augmentation of {2*iSamples} ({iSamples} controls, {iSamples} ASD) in {sDecomp} space:\n'
    #           f'(Noise method={sNoiseTo})\n'
    #           f'Mean Train score: {cAugRandomSearch.cv_results_["mean_train_score"][0]:.3f}+/-{cAugRandomSearch.cv_results_["std_train_score"][0]:.3f}\n'
    #           f'Mean CV score:{cAugRandomSearch.cv_results_["mean_test_score"][0]:.3f}+/-{cAugRandomSearch.cv_results_["std_test_score"][0]:.3f}'
    #           f'\nTest score: {flAugScore:.3f}'
    #           f'\nGeneralization Error: {(cAugRandomSearch.cv_results_["mean_train_score"][0]-cAugRandomSearch.cv_results_["mean_test_score"][0]):.3f}'
    #           )
    #
    #     pdAugmentationResults.loc[iAugExperiment]['Train AUC ROC'] = \
    #         pdBestAugParams.loc[pdBestAugParams['rank_test_score'] == 1]['mean_train_score'].values[0]
    #     pdAugmentationResults.loc[iAugExperiment]['Train std. error'] = \
    #         pdBestAugParams.loc[pdBestAugParams['rank_test_score'] == 1]['std_train_score'].values[0]
    #     pdAugmentationResults.loc[iAugExperiment]['Validation AUC ROC'] = \
    #         pdBestAugParams.loc[pdBestAugParams['rank_test_score'] == 1]['mean_test_score'].values[0]
    #     pdAugmentationResults.loc[iAugExperiment]['Validation std. error'] =\
    #         pdBestAugParams.loc[pdBestAugParams['rank_test_score'] == 1]['std_test_score'].values[0]
    #     pdAugmentationResults.loc[iAugExperiment]['Test AUC ROC'] = flAugScore
    #     pdAugmentationResults.loc[iAugExperiment]['Generalization Error'] = pdAugmentationResults.loc[iAugExperiment]['Train AUC ROC']-\
    #                                                                     pdAugmentationResults.loc[iAugExperiment]['Validation AUC ROC']
    #
    pkl.dump(pdAugmentationResults, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization'
                                            '/DataAugmentation/pdTestResults_Dense8_Test.p','wb'))
