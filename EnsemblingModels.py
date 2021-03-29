#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file Generically has a class to ensemble predictors

This file, if run alone, will ensemble the top 5 IMPAC TSE models

Originally developed for use in the ASD comparison project
Created by Cooper Mellema on 10 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

import os
import sys
import numpy as np
import sklearn as sk
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from BioMarkerIdentification import fLoadModels, fLoadPDData
import pickle as pkl
import pandas as pd
import random
import itertools
import glob
import time
import multiprocessing

from sklearn import linear_model
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

sys.path.append('/project/bioinformatics/DLLab/Cooper/Code/Utilities')
import DataManagementFunctions as dm

random.seed(42)
# Increase plot size
plt.style.use('seaborn-colorblind')
# lsFigSize = plt.rcParams["figure.figsize"]
# plt.rcParams["figure.figsize"] = [lsFigSize[1]*1.2*2, lsFigSize[1] * 2*2]
mpl.rcParams['text.color'] = 'black'
mpl.rcParams.update({'font.size': 18})


class cEnsembleModel:
    """
    class that makes an ensemble of models
    """
    def __init__(self, dModels):
        """
        Initialize the ensemble model class
        :param dModels: dictionary of models
        """
        for sModel in dModels.keys():
            setattr(self, f'{sModel}', dModels[sModel])

    def fPredictClassProb(self, dData, bReturn=True, bTest=True):
        """
        Predicts the probability of a given class given a dictionary of data
        :param dData: dictionary of data, should have the same keys as the
            dModels in the __init__ function; Each dict entry is an array for
            a model to make a prediction on
        :param bReturn: boolean, True if returning dict, false if just saving
        :return: dProbs: dictionary of probabilities of a class per predictor
        """
        # Initialize empty dict
        dProbs={}


        for sKey in dData.keys():
            # If there is no model type with the same key as the data, throw an error
            if not hasattr(self, f'{sKey}'):
                raise ValueError(f'There is no stored model corresponding to the '\
                                 f'data with key={sKey}')
            # Else, make a prediction with the data

            else:
                # If it is an sklearn model, use the predict_proba method
                if dm.fRecursiveHasattr(self, f'{sKey}.predict_proba'):
                    aProb=self.__dict__[sKey].predict_proba(dData[sKey])[:,0]
                    aProb=np.expand_dims(aProb, axis=1)
                    aProb=self._fConfirmShape(aProb)
                    dProbs.update({sKey: aProb})
                # If it is a keras model, use the predict method
                elif dm.fRecursiveHasattr(self, f'{sKey}.predict'):
                    aProb=self.__dict__[sKey].predict(dData[sKey])
                    aProb=self._fConfirmShape(aProb)
                    dProbs.update({sKey: aProb})
                # If it has no prediction methods, throw an error
                else:
                    raise AttributeError(f'Model:{sKey} has no prediction method')

            # Save the top model performance
            if sKey=='Model1':
                if bTest:
                    setattr(self, 'aTopProbs', aProb)
                else:
                    setattr(self, 'aTopTrainProbs', aProb)

        # Assign value
        if bTest:
            setattr(self, 'dProbs', dProbs)
        else:
            setattr(self, 'dTrainProbs', dProbs)

        # Return if flag is set
        if bReturn:
            return dProbs

    def fSoftVoter(self, dTrainData, dData, bReturn=False):
        """
        Performs soft-voting classification
        :param dData: dictionary of data, should have the same keys as the
            dModels in the __init__ function; Each dict entry is an array for
            a model to make a prediction on
        :param bReturn: boolean, True if returning dict, false if just saving
        :return: aPredicitons: array of probabilities of a class after soft voting
        """
        # check if Class probabilites have been calcuated, calculate them if not
        if not hasattr(self, 'dProbs'):
            self.fPredictClassProb(dData, bReturn=False)
        if not hasattr(self, 'dTrainProbs'):
            self.fPredictClassProb(dTrainData, bReturn=False, bTest=False)

        # Convert dict to array:
        aData=np.concatenate([self.dProbs[sKey] for sKey in self.dProbs.keys()], 1)
        aTrainData=np.concatenate([self.dTrainProbs[sKey] for sKey in self.dTrainProbs.keys()], 1)

        # Perform soft vote for final probabilities
        aPredictions=np.sum(aData, axis=1)/(aData.shape[1])
        aTrainPredictions=np.sum(aTrainData, axis=1)/(aTrainData.shape[1])

        # store predictions
        aPredictions=np.expand_dims(aPredictions, axis=1)
        self.aSoftVoteProbs=aPredictions

        aTrainPredictions=np.expand_dims(aTrainPredictions, axis=1)
        self.aSoftVoteProbsTrain=aTrainPredictions

        # return if flag is set
        if bReturn:
            return aPredictions

    def fHardVoter(self, dTrainData, dData, lsThresholds=np.linspace(0,1,1000), bReturn=False):
        """
        Performs hard-voting classification
        :param dData: dictionary of data, should have the same keys as the
            dModels in the __init__ function; Each dict entry is an array for
            a model to make a prediction on
        :param lsThresholds: list or generator, thresholds for voting
        :param bReturn: boolean, True if returning dict, false if just saving
        :return: aPredicitons: array of probabilities of a class after hard voting
        """
        # check if Class probabilites have been calcuated, calculate them if not
        if not hasattr(self, 'dProbs'):
            self.fPredictClassProb(dData, bReturn=False)
        if not hasattr(self, 'dTrainProbs'):
            self.fPredictClassProb(dTrainData, bReturn=False, bTest=False)

        # Convert dict to array:
        aData = np.concatenate([self.dProbs[sKey] for sKey in self.dProbs.keys()], 1)
        aTrainData=np.concatenate([self.dTrainProbs[sKey] for sKey in self.dTrainProbs.keys()], 1)

        dPredictions={}
        dTrainPredictions={}

        aThresholds=np.zeros(aData.shape[0:1])
        aTrainThresholds=np.zeros(aTrainData.shape[0:1])

        for flThreshold in lsThresholds:
            # Perform hard votes for final probabilities
            aVotes=(aData>flThreshold).astype('int')
            aTrainVotes=(aTrainData>flThreshold).astype('int')

            aPredictions=((np.sum(aVotes, axis=1) / (aVotes.shape[1]))>=0.5).astype('int')
            aTrainPredictions=((np.sum(aTrainVotes, axis=1) / (aTrainVotes.shape[1]))>=0.5).astype('int')

            for iRow in range(aThresholds.shape[0]):
                if aPredictions[iRow]==0:
                    if aThresholds[iRow]==0:
                        aThresholds[iRow]=flThreshold
                if aTrainPredictions[iRow]==0:
                    if aTrainThresholds[iRow]==0:
                        aTrainThresholds[iRow]=flThreshold

        aThresholds=np.expand_dims(aThresholds, axis=1)
        self.aHardVoteProbs=aThresholds

        aTrainThresholds=np.expand_dims(aTrainThresholds, axis=1)
        self.aHardVoteProbsTrain=aTrainThresholds

        # Return dict of binary predictions per threshold
        if bReturn:
            return aThresholds

    def fTrainStackingModel(self, dTrainData, dTestData, aTrainClasses, bReturn=False, sType='Random Forest'):
        # check if Class probabilites have been calcuated, calculate them if not
        if not hasattr(self, 'dTrainProbs'):
            self.fPredictClassProb(dTrainData, bReturn=False, bTest=False)
        if not hasattr(self, 'dProbs'):
            self.fPredictClassProb(dTestData, bReturn=False, bTest=True)

        # Convert dict to array:
        aTrainData = np.concatenate([self.dTrainProbs[sKey] for sKey in self.dTrainProbs.keys()], 1)
        aTestData = np.concatenate([self.dProbs[sKey] for sKey in self.dProbs.keys()], 1)

        # Perform the training of the stacking model
        aTrainPredictions, aTestPredictions, cModel = self._fStacker(aTrainData, aTrainClasses, aTestData, sType=sType)

        # save to attribute
        dm.fRecursiveSetattr(self, f'aStackingProbs{sType}', aTestPredictions)
        dm.fRecursiveSetattr(self, f'aStackingProbs{sType}Train', aTrainPredictions)
        dm.fRecursiveSetattr(self, f'{sType}StackingModel', cModel)

        # Return dict of binary predictions per threshold
        if bReturn:
            return aTestPredictions, cModel

    def _fRetreiveClassifier(self, sType):
        # generate the classifier and the distribution of parameters to search over
        if sType=='Lasso Logistic Regression':
            cClassifier = linear_model.LogisticRegression(penalty='l1')
            dParamDistributions = {
                'C': 1000 * (10 ** np.random.uniform(-5, 1, 100)),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
        elif sType=='Ridge Logistic Regression':
            cClassifier = linear_model.LogisticRegression(penalty='l2')
            dParamDistributions = {
                'C': 1000 * (10 ** np.random.uniform(-5, 1, 100)),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
        elif sType=='Linear SVM':
            cClassifier = SVC(probability=True)
            dParamDistributions = {
                'C': 10 ** np.random.uniform(-4, 4, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
        elif sType=='Gaussian SVM':
            cClassifier = SVC(probability=True)
            dParamDistributions = {
                'C': 10 ** np.random.uniform(-4, 4, 100),
                'gamma': 10 ** np.random.uniform(-2, 2, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
        elif sType=='Random Forest':
            cClassifier = RandomForestClassifier()
            dParamDistributions = {
                'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
                'max_leaf_nodes': np.random.randint(5, 50, 100)
            }
        elif sType=='Extremely Random Trees':
            cClassifier = ExtraTreesClassifier()
            dParamDistributions = {
                'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
                'max_leaf_nodes': np.random.randint(5, 50, 100),
            }
        # if not one of the above types, throw an error
        else:
            raise NotImplementedError(f'Only Logistic Regression (ridge and lasso),'
                                      f'SVM (linear and gaussain kernel), Random Forest,'
                                      f' and extremely random trees are implemented,'
                                      f' Model type {sType} has not yet '
                                      f'been implemented')

        # return model and model params to search over
        return cClassifier, dParamDistributions

    def _fStacker(self, aXTrainData, aYTrainData, aXTestData, sType='Random Forest'):

        # Retrieve the classifier
        cClassifier, dParamDistributions = self._fRetreiveClassifier(sType)

        # Run a random search
        cRandSearch = RandomizedSearchCV(
            cClassifier,
            dParamDistributions,
            cv=10, # TODO change to 10
            n_iter=50, # TODO change to 50
            n_jobs=1,
            verbose=2,
            scoring='roc_auc'
        )

        # Fit the model
        cRandSearch.fit(aXTrainData, aYTrainData)
        cModel = cRandSearch

        #Make predictions
        if dm.fRecursiveHasattr(cRandSearch, 'best_estimator_.predict_proba'):
            aTrainPredictions = np.expand_dims(cRandSearch.best_estimator_.predict_proba(aXTrainData)[:,1], axis=1)
            aTestPredictions = np.expand_dims(cRandSearch.best_estimator_.predict_proba(aXTestData)[:,1], axis=1)
        else:
            aTrainPredictions = cRandSearch.best_estimator_.predict(aXTrainData)
            aTestPredictions = cRandSearch.best_estimator_.predict(aXTestData)

        aTrainPredictions = self._fConfirmShape(aTrainPredictions)
        aTestPredictions = self._fConfirmShape(aTestPredictions)

        dm.fRecursiveSetattr(self, f'c{sType.replace(" ", "")}StackingModel', cModel)

        return aTrainPredictions, aTestPredictions, cModel

    def _fConfirmShape(self, aArray):
        # find dimensions of array
        iDims=len(aArray.shape)

        # reshape to a 2D array
        if iDims==1:
            aArray=np.expand_dims(aArray, axis=1)
        elif iDims==2:
            pass
        elif iDims>2:
            while iDims>2:
                aArray=np.squeeze(aArray, axis=iDims-1)
                iDims=iDims-1

        return aArray

    def fPlotROCAUC(self, aYTest, sTitle, sSavePath):
        """
        Plots ROC curve for classifier
        :param aYTest: Test data
        :param sTitle: title
        :param sSavePath: save location
        :return: none
        """
        #Plot the Top model alone ROC AUC
        flScore = sk.metrics.roc_auc_score(aYTest, dm.fRecursiveGetattr(self, 'aTopProbs'))
        lsFPR, lsTPR, lsThresholds = sk.metrics.roc_curve(aYTest,
                                                           dm.fRecursiveGetattr(self, 'aTopProbs'))
        fig, ax = plt.subplots()

        ax.plot(lsFPR, lsTPR, label=f'Top Model Alone (Dense FeedFwd), area={flScore:.4g}', alpha=0.5)

        dScore={}

        #Calculate ROC metrics
        lsAttr=list(a for a in dir(self) if (a.startswith('a') and not a.__contains__('Top') and not a.__contains__('Train')))
        for sAttr in lsAttr:
            flScore = sk.metrics.roc_auc_score(aYTest, dm.fRecursiveGetattr(self, sAttr))
            lsFPR, lsTPR, lsThresholds = sk.metrics.roc_curve(aYTest,
                                                                      dm.fRecursiveGetattr(self, sAttr))
            dScore.update({sAttr: flScore})

            # make plot
            sTitleLabel=sAttr[1:].replace('Probs', ' ')
            ax.plot(lsFPR, lsTPR, label=f'{sTitleLabel}, area={flScore:.4g}', alpha=0.5)

        ax.plot([0, 1], [0, 1], '--')

        # finalize plot
        # Labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        #ax.axis('equal')
        fig.set_size_inches(10,10)
        ax.set_xlabel('False Positive Rate', color='black')
        ax.set_ylabel('True Positive Rate', color='black')
        ax.set_title(sTitle)

        # set colors
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')


        leg=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=1,
                   facecolor='white', prop={'size':50})
        plt.tight_layout()

        #save and clear
        plt.savefig(sSavePath, facecolor='white', transparent=True)
        plt.close()

        for line in leg.get_lines():
            line.set_linewidth(12.0)
        fig2=leg.figure
        fig2.canvas.draw()
        bbox=leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig2.savefig(sSavePath.replace(".png", "Legend.png"), dpi="figure", bbox_inches=bbox)

        return dScore

    def fViolinComparison(self, sType, aYTrainData, aYTestData, sSavePath):
        #Get Test and train predictions
        if not (sType=='Soft Vote' or sType=='Hard Vote'):
            aStackingTestPredictions=dm.fRecursiveGetattr(self, f'aStackingProbs{sType}')
            aStackingTrainPredictions=dm.fRecursiveGetattr(self, f'aStackingProbs{sType}Train')
        else:
            aStackingTestPredictions=dm.fRecursiveGetattr(self, f'a{sType.replace(" ","")}Probs')
            aStackingTrainPredictions=dm.fRecursiveGetattr(self, f'a{sType.replace(" ","")}ProbsTrain')

        aTopTestPredictions=dm.fRecursiveGetattr(self, f'aTopProbs')
        aTopTrainPredictions=dm.fRecursiveGetattr(self, f'aTopTrainProbs')

        #Make a dataframe of Train and test data
        dfStackedTrain=pd.DataFrame.from_dict({'ASD Positive': aYTrainData.flatten(), 'Predictions':
            aStackingTrainPredictions.flatten()})
        dfStackedTrain['Model Type']='Stacked'
        dfStackedTrain['Data']='Train'

        dfTopTrain=pd.DataFrame.from_dict({'ASD Positive': aYTrainData.flatten(), 'Predictions':
            aTopTrainPredictions.flatten()})
        dfTopTrain['Model Type']='Top'
        dfTopTrain['Data']='Train'

        dfStackedTest = pd.DataFrame.from_dict({'ASD Positive': aYTestData.flatten(), 'Predictions':
            aStackingTestPredictions.flatten()})
        dfStackedTest['Model Type']='Stacked'
        dfStackedTest['Data']='Test'

        dfTopTest = pd.DataFrame.from_dict({'ASD Positive': aYTestData.flatten(), 'Predictions':
            aTopTestPredictions.flatten()})
        dfTopTest['Model Type']='Top'
        dfTopTest['Data']='Test'

        dfCombined = pd.concat([dfStackedTrain, dfStackedTest, dfTopTrain, dfTopTest])

        # Generate plot
        sns.set(style="whitegrid")
        cPlot = sns.catplot(
            x="Model Type",
            y="Predictions",
            hue="Data",
            col="ASD Positive",
            data=dfCombined,
            kind="violin",
            split=True,
            scale='count'
        )
        plt.ylim(-0.2, 1.2)
        plt.ylabel('Probability of ASD')
        plt.show()
        #save and clear
        plt.savefig(sSavePath)
        plt.close()

def _fConfirmShape(aArray):
    """
    Reshapes array to a 2 D array
    :param aArray: the array in question
    :return: aArray (reshaped)
    """
    # find dimensions of array
    iDims=len(aArray.shape)

    # reshape to a 2D array
    if iDims==1:
        aArray=np.expand_dims(aArray, axis=1)
    elif iDims==2:
        pass
    elif iDims>2:
        while iDims>2:
            aArray=np.squeeze(aArray, axis=iDims-1)
            iDims=iDims-1

    return aArray

def _fRetreiveClassifier(sType):
    # generate the classifier and the distribution of parameters to search over
    if sType=='Lasso Logistic Regression':
        cClassifier = linear_model.LogisticRegression(penalty='l1')
        dParamDistributions = {
            'C': 1000 * (10 ** np.random.uniform(-5, 1, 100)),
            'max_iter': np.random.uniform(1000, 100000, 100),
        }
    elif sType=='Ridge Logistic Regression':
        cClassifier = linear_model.LogisticRegression(penalty='l2')
        dParamDistributions = {
            'C': 1000 * (10 ** np.random.uniform(-5, 1, 100)),
            'max_iter': np.random.uniform(1000, 100000, 100),
        }
    elif sType=='Linear SVM':
        cClassifier = SVC(probability=True)
        dParamDistributions = {
            'C': 10 ** np.random.uniform(-4, 4, 100),
            'max_iter': np.random.uniform(10000, 100000, 100)
        }
    elif sType=='Gaussian SVM':
        cClassifier = SVC(probability=True)
        dParamDistributions = {
            'C': 10 ** np.random.uniform(-4, 4, 100),
            'gamma': 10 ** np.random.uniform(-2, 2, 100),
            'max_iter': np.random.uniform(10000, 100000, 100)
        }
    elif sType=='Random Forest':
        cClassifier = RandomForestClassifier()
        dParamDistributions = {
            'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
            'max_leaf_nodes': np.random.randint(5, 50, 100)
        }
    elif sType=='Extremely Random Trees':
        cClassifier = ExtraTreesClassifier()
        dParamDistributions = {
            'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
            'max_leaf_nodes': np.random.randint(5, 50, 100),
        }
    # if not one of the above types, throw an error
    else:
        raise NotImplementedError(f'Only Logistic Regression (ridge and lasso),'
                                  f'SVM (linear and gaussain kernel), Random Forest,'
                                  f' and extremely random trees are implemented,'
                                  f' Model type {sType} has not yet '
                                  f'been implemented')

    # return model and model params to search over
    return cClassifier, dParamDistributions

def fRandSearch(sType, aXTrain, aYTrain):
    # Retrieve the classifier
    cClassifier, dParamDistributions = _fRetreiveClassifier(sType)

    # Run a random search
    cRandSearch = RandomizedSearchCV(
        cClassifier,
        dParamDistributions,
        cv=10,
        n_iter=50,
        n_jobs=1,
        verbose=2,
        scoring='roc_auc'
    )

    # Fit the model
    cRandSearch.fit(aXTrain, aYTrain)
    return cRandSearch

def fMakePredictions(dModels, dXData):
    """
    Predicts the probability of a given class given a dictionary of data
    :param dModels: dictionary of the models
    :param dXData: dictionary of data, should have the same keys as the
        dModels
    """
    # Initialize empty dict
    dProbs={}

    # Loop through each model
    for sKey in dXData.keys():
        # If there is no model type with the same key as the data, throw an error
        if not sKey in dModels.keys():
            raise ValueError(f'There is no stored model corresponding to the '\
                             f'data with key={sKey}')

        # Else, make a prediction with the data
        else:
            print(f'Making prediction for model {sKey}')
            # If it is an sklearn model, use the predict_proba method
            if hasattr(dModels[sKey], 'predict_proba'):
                aProb=dModels[sKey].predict_proba(dXData[sKey])[:,0]
                # softmax the linear regression models
                if sKey.__contains__('LinRidge') or sKey.__contains__('LinLasso'):
                    aProb=np.exp(aProb)/sum(np.exp(aProb))
            # If it is a keras model, or doesn't have a predict_proba method, use the predict method
            elif hasattr(dModels[sKey], 'predict'):
                aProb=dModels[sKey].predict(dXData[sKey])
                # softmax the linear regression models
                if sKey.__contains__('LinRidge') or sKey.__contains__('LinLasso'):
                    aProb=np.exp(aProb)/sum(np.exp(aProb))
            # If it has no prediction methods, throw an error
            else:
                raise AttributeError(f'Model:{sKey} has no prediction method')
        #save result
        aProb = _fConfirmShape(aProb)
        dProbs.update({sKey: aProb})

    return dProbs

def fReducedEnsemble(lsModels, dPredictions, aYTrain, dTestPredictions=None, aYTest=None, bRandSearch=False,
                     iCrossVal=4):
    """
    Produces predictions for a reduced set of ensembled models without any random searching
    :param lsModels: list of which models to ensemble [3 only]
    :param dPredictions: dictionary of the predictions
    :param aYTrain: array of the true training labels
    :return:
    """
    # select subset of predictions to ensemble
    aPredictions=np.concatenate([value for key, value in dPredictions.items() if key in lsModels], 1)
    if dTestPredictions is not None:
        aTestPredictions = np.concatenate([value for key, value in dTestPredictions.items() if key in lsModels], 1)

    # Do ensembling
    if (dTestPredictions is None) and (aYTest is None):
        #Start values at one
        flSoftScore = 0
        flHardScore = 0
        flLinRidgeScore = 0
        flSVMScore = 0

        # Set up cross validation
        cSplitter=KFold(n_splits=iCrossVal)

        for aTrainIndex, aValidationIndex in cSplitter.split(aPredictions):
            # do ensembling
            aSoftVote=np.mean(aPredictions[aValidationIndex], axis=1)
            aHardVote=fHardVote(aPredictions[aValidationIndex])
            aLinRidgeStack=linear_model.LogisticRegression().fit(aPredictions[aTrainIndex], aYTrain[aTrainIndex]).predict_proba(
                aPredictions[aValidationIndex])[:,1]
            aSVMStack=SVC(probability=True).fit(aPredictions[aTrainIndex], aYTrain[aTrainIndex]).predict_proba(
                aPredictions[aValidationIndex])[:,1]

            # Calculate score
            flSoftScore = flSoftScore + sk.metrics.roc_auc_score(aYTrain[aValidationIndex], aSoftVote)
            flHardScore = flHardScore + sk.metrics.roc_auc_score(aYTrain[aValidationIndex], aHardVote)
            flLinRidgeScore = flLinRidgeScore + sk.metrics.roc_auc_score(aYTrain[aValidationIndex], aLinRidgeStack)
            flSVMScore = flSVMScore + sk.metrics.roc_auc_score(aYTrain[aValidationIndex], aSVMStack)

        # Find Average
        flSoftScore = float(flSoftScore/iCrossVal)
        flHardScore = float(flHardScore/iCrossVal)
        flLinRidgeScore = float(flLinRidgeScore/iCrossVal)
        flSVMScore = float(flSVMScore/iCrossVal)

    elif bRandSearch==False:
        aSoftVote=np.mean(aTestPredictions, axis=1)
        aHardVote=fHardVote(aTestPredictions)
        aLinRidgeStack=linear_model.LogisticRegression().fit(aPredictions, aYTrain).predict_proba(aTestPredictions)[:,1]
        aSVMStack=SVC(probability=True).fit(aPredictions, aYTrain).predict_proba(aTestPredictions)[:,1]

        # Calculate score
        flSoftScore=sk.metrics.roc_auc_score(aYTest, aSoftVote)
        flHardScore=sk.metrics.roc_auc_score(aYTest, aHardVote)
        flLinRidgeScore=sk.metrics.roc_auc_score(aYTest, aLinRidgeStack)
        flSVMScore=sk.metrics.roc_auc_score(aYTest, aSVMStack)
    else:
        aSoftVote=np.mean(aTestPredictions, axis=1)
        aHardVote=fHardVote(aTestPredictions)
        cLinRidgeStack=fRandSearch('Ridge Logistic Regression', aPredictions, aYTrain)
        aLinRidgeStack=cLinRidgeStack.best_estimator_.predict_proba(aTestPredictions)[:,1]
        cSVMStack=fRandSearch('Gaussian SVM', aPredictions, aYTrain)
        aSVMStack=cSVMStack.best_estimator_.predict_proba(aTestPredictions)[:,1]

        # Calculate score
        flSoftScore=sk.metrics.roc_auc_score(aYTest, aSoftVote)
        flHardScore=sk.metrics.roc_auc_score(aYTest, aHardVote)
        flLinRidgeScore=sk.metrics.roc_auc_score(aYTest, aLinRidgeStack)
        flSVMScore=sk.metrics.roc_auc_score(aYTest, aSVMStack)

    # save score
    sTag=f'{lsModels}'.replace('"','').replace("'","").replace(',','').replace('[','').replace(']','').replace(' ', '_')

    return {f'{sTag}_Scores': [flSoftScore, flHardScore, flLinRidgeScore, flSVMScore]}

def fHardVote(aData):
    """
    Does hard voting on prediction vector while varying the threshold
    :param aData: array of data
    :return: array of predictions with the threshold varied
    """
    aThresholds = np.zeros(aData.shape[0:1])

    for flThreshold in np.linspace(0,1,1000):
        # Perform hard votes for final probabilities, varying the threshold for voting
        aVotes = (aData > flThreshold).astype('int')

        aPredictions = ((np.sum(aVotes, axis=1) / (aVotes.shape[1])) >= 0.5).astype('int')

        # Fill the final prediction vector
        for iRow in range(aThresholds.shape[0]):
            if aPredictions[iRow] == 0:
                if aThresholds[iRow] == 0:
                    aThresholds[iRow] = flThreshold
    return aThresholds

def fFormatToDataFrame(sModelPerformanceDir):
    """
    Loops through a directory containing pickled model performance and places them in a dataframe
    :param sModelPerformanceDir: path to directory where the model performances are stored
    :return: pdPerformance
    """
    # Get list of files in dir
    lsFiles=glob.glob(f'{sModelPerformanceDir}/*.p')
    pdPerformance=pd.DataFrame(index=lsFiles, columns=['SoftVote', 'HardVote', 'LogisticRidge', 'SVM'])

    # load each into dataframe
    for sFile in lsFiles:
        pdPerformance.loc[sFile]=pkl.load(open(os.path.join(sModelPerformanceDir, sFile),'rb'))

    return pdPerformance.astype('float')

def fFormatListOfDicts(lsd):
    """
    Formats the list of dicts returned by the parallel processing
    :param lsd: list of dicts
    :return: dataframe of data, in form:
           | Soft Vote |  Hard Vote  | Logistic Ridge  |  SVM   |
    Model1 |    ...    |     .....   |    ......       |  ...   |
    Model2 |    ...    |     .....   |    ......       |  ...   |
    """
    dReformatted = dict((key, d[key]) for d in lsd for key in d)
    dfFinal=pd.DataFrame.from_dict(dReformatted).T
    dfFinal.columns=['Soft Vote', 'Hard Vote', 'Logistic Ridge', 'SVM']

    return dfFinal

if '__main__'==__name__:
    # Load the data that will be used
    dXData, aYData = fLoadPDData()
    del(dXData)

    # NOTE, the predictions alone are preferentially used over the entire model
    sDataPath='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles/EnsembleDataSet.p'

    # load indices of train/test
    sTrainLoc='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/aTrainingIndex.p'
    sTestLoc='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/aTestIndex.p'
    aTrainLoc=pkl.load(open(sTrainLoc,'rb'))
    aTestLoc=pkl.load(open(sTestLoc,'rb'))
    aYTrain=aYData[aTrainLoc.astype('int'),:]
    aYTest=aYData[aTestLoc.astype('int'),:]
    del(aYData)

    # load data for models
    dXData = pkl.load(open(sDataPath, 'rb'))

    # Get training and test X data
    dXTrainData={}
    dXTestData={}
    for sKey in dXData.keys():
        if 'Dense' in sKey:
            dXTrainData[sKey]=np.expand_dims(np.expand_dims(dXData[sKey].iloc[aTrainLoc].values, axis=1), axis=3)
            dXTestData[sKey]=np.expand_dims(np.expand_dims(dXData[sKey].iloc[aTestLoc].values, axis=1), axis=3)
        elif 'LSTM' in sKey:
            dXTrainData[sKey]=np.expand_dims(dXData[sKey].iloc[aTrainLoc].values, axis=1)
            dXTestData[sKey]=np.expand_dims(dXData[sKey].iloc[aTestLoc].values, axis=1)
        else:
            dXTrainData[sKey]=dXData[sKey].iloc[aTrainLoc].values
            dXTestData[sKey]=dXData[sKey].iloc[aTestLoc].values

    # make dictionary of predictions from models
    sPredictionPath='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles' \
                    '/EnsemblePredictionSetSoftmaxed.p'

    # If the predictions exist, load them. Otherwise, make the predictions
    if not os.path.isfile(sPredictionPath):
        sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles/EnsembleModelSet.p'
        dModels = pkl.load(open(sModelPath, 'rb'))
        dPredictions=fMakePredictions(dModels, dXTrainData)
        pkl.dump(dPredictions, open(sPredictionPath, 'wb'))
    else:
        dPredictions=pkl.load(open(sPredictionPath,'rb'))

    #Repeat for test data
    sTestPredictionPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles' \
                      '/EnsemblePredictionTestSetSoftmaxed.p'
    if not os.path.isfile(sTestPredictionPath):
        sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles' \
                     '/EnsembleModelSet.p'
        dModels = pkl.load(open(sModelPath, 'rb'))
        dTestPredictions=fMakePredictions(dModels, dXTestData)
        pkl.dump(dTestPredictions, open(sTestPredictionPath, 'wb'))
    else:
        dTestPredictions=pkl.load(open(sTestPredictionPath,'rb'))


    # Set number of parallel jobs
    nParallel=int(multiprocessing.cpu_count()/4)

    # if True, run ensembles of 3 models
    b3Models=False
    if b3Models:
        # set up all sets to loop through for 3 combinations in parallel
        lsTopThrees=list([x for x in dPredictions.keys() if not x.__contains__('SVM')])
        ls3Perms=list(itertools.combinations(lsTopThrees, 3))
        random.shuffle(ls3Perms)
        print(f'Running parallel ensemble jobs, nNodes={nParallel}')

        # run the ensembling
        lsDict3Perms=Parallel(n_jobs=nParallel)(delayed(fReducedEnsemble)(lsModels, dPredictions, aYTrain) for lsModels in ls3Perms)

        # Convert to the right form
        pd3Perms = fFormatListOfDicts(lsDict3Perms)

        # Save
        pkl.dump(pd3Perms, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
                                         'AlternateMetrics/Ensembles/pdIterativeSearchEnsembleOf3NoSVM.p',
                                'wb'))

    # if True, run ensembles of 5 models
    b5Models=True
    if b5Models:
        # set up sets to loop through for 5 combinations in parallel
        # only uses top two models per category rather than the top 10 per category
        lsTopTwos=list([x for x in dPredictions.keys() if ((x.__contains__('Rank0') or x.__contains__('Rank1')) and
                                                           not x.__contains__('SVM'))])
        ls5Perms=list(itertools.combinations(lsTopTwos,5))
        random.shuffle(ls5Perms)

        # Run the ensembling
        lsDict5Perms=Parallel(n_jobs=nParallel)(delayed(fReducedEnsemble)
                                   (lsModels, dPredictions, aYTrain) for lsModels in ls5Perms)

        # Convert to the right form
        pd5Perms = fFormatListOfDicts(lsDict5Perms)

        # Save
        pkl.dump(pd5Perms, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
                                         'AlternateMetrics/Ensembles/pdIterativeSearchEnsembleOf5NoSVM.p',
                                'wb'))

    # if bTest, load the top results and train the top results in a dictionary
    bTest=False
    bRandSearch=False
    if bTest:
        # set up sets to loop through top models
        dTop5s=pkl.load(open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
                              'AlternateMetrics/Ensembles/dIterativeSearchEnsembleOf5SoftmaxLinResults.p', 'rb'))
        dTop3s=pkl.load(open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
                              'AlternateMetrics/Ensembles/dIterativeSearchEnsembleOf3SoftmaxLinResults.p', 'rb'))
        # Run the ensembling
        lsDict5PermsTop=Parallel(n_jobs=nParallel)(delayed(fReducedEnsemble)
                                   (lsModels, dPredictions, aYTrain, dTestPredictions, aYTest, bRandSearch) for
                                                   lsModels in list(dTop5s.values()))
        lsDict3PermsTop=Parallel(n_jobs=nParallel)(delayed(fReducedEnsemble)
                                   (lsModels, dPredictions, aYTrain, dTestPredictions, aYTest, bRandSearch) for
                                                   lsModels in list(dTop3s.values()))

        pd5PermsTop=fFormatListOfDicts(lsDict5PermsTop)
        pd3PermsTop=fFormatListOfDicts(lsDict3PermsTop)

        # Save
        # pkl.dump(pd5PermsTop, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
        #                                  'AlternateMetrics/Ensembles/pdIterativeSearchEnsembleOf5TopResults.p','wb'))
        # # Save
        # pkl.dump(pd3PermsTop, open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
        #                                  'AlternateMetrics/Ensembles/pdIterativeSearchEnsembleOf3TopResults.p','wb'))


