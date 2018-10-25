""" This code takes the Paris IMPAC Autism study data,
runs multiple machine learning algorithms on it, and compares the results

The different machine learning algorithms that are compared are:
-Naive Bayes (gaussian)
-GLM
    -with ridge regression
    -with lasso regression
-SVM
    -with gaussian kernel
    -with linear kernel
-Random forest
-Extremely random trees
-Adaptive boosting with AdaBoost
-Gradient boosting with xgBoost

The different metrics used are:
- accuracy
- Precision Recall area under curve
- F1 score
- ROC area under curve

Written by Cooper Mellema in the Montillo Deep Learning Lab at UT Southwesten Medical Center
August 2018
"""

import numpy as np
import scipy as sp
import sklearn as sk
import datetime
import os
import sys
import pip
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import keras
import warnings
import xgboost
import random
import seaborn
import nibabel
import nilearn
import itertools
import multiprocessing
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict, cross_validate, \
    RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, auc, \
    f1_score
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from nilearn.connectome import ConnectivityMeasure
from sklearn.naive_bayes import GaussianNB
from keras.utils import to_categorical
from random import shuffle
import pickle

np.random.seed(42)

# Set up directory locations
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


def fSaveFigure(sFigureIdentification, sImagesPath, bTightLayout=True, sFigureExtension="png", intResolution=300):
    sPath = os.path.join(sImagesPath, sFigureIdentification + "." + sFigureExtension)
    print("Saving figure ", sFigureIdentification)
    if bTightLayout:
        plt.tight_layout()
        plt.savefig(sPath, format=sFigureExtension, dip=intResolution)


def fNormalize(dataset):
    f64mean = dataset.mean(axis=0)
    f64std = dataset.std(axis=0)
    f32norm_dataset = np.float32(dataset - f64mean) / (f64std + 0.000001)
    return f32norm_dataset


# for f(10, [1,2,3]) returns [10**1, 10**2, 10**3]
def fElementWiseExponential(f64Element, aArray):
    aExp = f64Element * np.ones(len(aArray))
    for i in range(len(aExp)):
        aExp[i] = aExp[i] ** aArray[i]
    return aExp


# Ignore irrelevant warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# Setting up a class that runs the different machine learning modules


class cMachineLearning(object):
    def __init__(self, xData, yData, xTest, yTest, data_name, sImagesPath):
        self.xData = xData
        self.yData = yData
        self.xTest = xTest
        self.yTest = yTest
        self.data_name = data_name
        self.images_path = sImagesPath

    # This segment fits the model, generates an accuracy score, F1 score, area under a
    # Precision recall curve, and area under a ROC curve. It also plots and saves a confusion matrix,
    # Precision-recall curve, and a ROC curve
    def fProcessData(self, cClassifier, dParamDistributions, sPrefix, fMRI=False, FeatureExtractor=None):
        # if there are parameters, a random search is done

        if fMRI == True:
            cClassifier = make_pipeline(FeatureExtractor, cClassifier)

        if dParamDistributions is not None:
            cRandSearch = RandomizedSearchCV(
                cClassifier,
                dParamDistributions,
                cv=10,  # change to 10
                n_iter=50,  # change to 50
                n_jobs=1,
                verbose=2,
                scoring='roc_auc'
            )

            cRandSearch.fit(self.xData, self.yData)
            skcModel = cRandSearch
            aPredicted = cRandSearch.best_estimator_.predict(self.xTest)
        else:
            skcModel = cClassifier.fit(self.xData, self.yData)
            aPredicted = cClassifier.predict(self.xTest)

        aRoundedPredicted = aPredicted
        aRoundedPredicted[aRoundedPredicted >= 0.5] = 1
        aRoundedPredicted[aRoundedPredicted < 0.5] = 0
        f64Accuracy = accuracy_score(self.yTest, aRoundedPredicted, normalize=False)

        self.fPlotNormalizedConfusionMatrix(aRoundedPredicted)
        f64PRAUC, f64F1score = self.fPrecisionRecall(aPredicted)
        f64ROCAUC = self.fROC(aPredicted)


        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # Generates a ROC curve AUC score and plots the curve if figure name is given
    def fROC(self, yPredicted, figure_name=None, label="None"):
        FPR, TPR, thresholds = roc_curve(self.yTest, yPredicted)
        score = roc_auc_score(self.yTest, yPredicted)
        if figure_name is not None:
            plt.plot(FPR, TPR, linewidth=2, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([0, 1, 0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(figure_name)
            fSaveFigure(figure_name, self.images_path)
            return score
        else:
            return score


    # Generates a Precision Recall curve AUC score and plots the curve if figure name is given
    def fPrecisionRecall(self, yPredicted, figure_name=None):
        precisions, recalls, thresholds = precision_recall_curve(self.yTest, yPredicted)
        AUCscore = auc(recalls, precisions)
        f1score = f1_score(self.yTest, np.rint(yPredicted))
        if figure_name is not None:
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], 'r-', label="Recall")
            plt.xlabel("Threshold")
            plt.legend(loc="center left")
            plt.ylim([0, 1])
            plt.title(figure_name)
            fSaveFigure(figure_name, self.images_path)
            return AUCscore, f1score
        else:
            return AUCscore, f1score


    # Generates a confusion matrix
    def fPlotNormalizedConfusionMatrix(self, yPredicted, figure_name=None):
        aConfusionMatrix = confusion_matrix(self.yTest, yPredicted)
        aConfusionMatrix = aConfusionMatrix.astype('float') / aConfusionMatrix.sum(axis=1)
        if figure_name is not None:
            plt.matshow(aConfusionMatrix)
            plt.title(figure_name)
            fSaveFigure(figure_name, self.images_path)

    # generates Naive Bayes classifier and process it
    def fNaiveBayes(self):
        cNaiveBayesClassifier = GaussianNB()

        sPrefix = "Naive Bayes "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cNaiveBayesClassifier,
                                                                                               None, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates GLM classifier with either ridge or lasso regularization and process it
    def fGLM(self, regularizer="ridge"):

        if regularizer == "ridge":
            cGLMClassifier = linear_model.Ridge()
            dParamDistributions = {
                'alpha': 10 ** np.random.uniform(-5, 1, 100),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
            sPrefix = "GLM with Ridge Regularization "

        elif regularizer == "lasso":
            cGLMClassifier = linear_model.Lasso()
            dParamDistributions = {
                'alpha': 10 ** np.random.uniform(-5, 1, 100),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
            sPrefix = "GLM with Lasso Regularization "

        else:
            print("Please specify 'ridge' or 'lasso' regularization")

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cGLMClassifier,
                                                                                               dParamDistributions,
                                                                                               sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates SVM classifier with either a linear or gaussian kernel and process it
    def fSVM(self, kernel="linear"):

        if kernel == "linear":
            cSVMClassifier = LinearSVC()
            dParamDistributions = {
                'C': 10 ** np.random.uniform(-4, 4, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
            sPrefix = "Linear "

        elif kernel == "gaussian":
            cSVMClassifier = SVC()
            dParamDistributions = {
                'C': 10 ** np.random.uniform(-4, 4, 100),
                'gamma': 10 ** np.random.uniform(-2, 2, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
            sPrefix = "Gaussian Kernel "

        else:
            print("Please specify 'linear' or 'gaussian' kernel")

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cSVMClassifier,
                                                                                               dParamDistributions,
                                                                                               sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates Random Forest classifier and process it
    def fRandForest(self):

        cRandForestClassifier = RandomForestClassifier()
        dParamDistributions = {'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
                               'max_leaf_nodes': np.random.randint(5, 50, 100)}
        sPrefix = "Random Forest "
        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cRandForestClassifier,
                                                                                               dParamDistributions,
                                                                                               sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates an Extremely Random Trees classifier and process it
    def fExtremeRandTrees(self):

        cExtremeRandTreesClassifier = ExtraTreesClassifier()
        dParamDistributions = {
            'n_estimators': np.round(5 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
            'max_leaf_nodes': np.random.randint(5, 50, 100),
        }
        sPrefix = "Extremely Random Trees "
        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(
            cExtremeRandTreesClassifier, dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates an AdaBoost Gradient Boosting classifier and process it
    def fAdaBoost(self):

        cAdaBoostClass = AdaBoostClassifier()
        dParamDistributions = {
            'n_estimators': np.round(2 * 10 ** np.random.uniform(1, 3, 100)).astype(int),
            'learning_rate': np.random.uniform(0.1, 0.9, 100),
        }
        sPrefix = "Ada Boost "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cAdaBoostClass,
                                                                                               dParamDistributions,
                                                                                               sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # #generates an xgBoost gradient boosting classifier and process it
    def fXGradBoost(self):
        cXGradBoost = XGBClassifier()

        dParamDistributions = {
            'n_estimators': np.round(5 * 10 ** np.random.uniform(0.1, 3, 100)).astype(int),
            'max_depth': np.random.randint(1, 10, 100),
            'subsample': np.random.uniform(0.2, 0.8, 100),
            'colsample_bytree': np.random.uniform(0.2, 1, 100),
            'learning_rate': 10 ** np.random.uniform(-2, 0, 100),
        }

        sPrefix = "XGradient Boost "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cXGradBoost,
                                                                                               dParamDistributions,
                                                                                               sPrefix)

        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC


######################### Getting the data ###################################

def fRunAnalysis(xData, xTest, yData, yTest, sTargetDirectory):
    # Running the analyses and saving in a table, which is then all saved in the TargetDirectory

    # making folder if it doesn't exist
    sTargetDirectory = os.path.join(sProjectRootDirectory, sProjectIdentification, sTargetDirectory)
    if not os.path.exists(sTargetDirectory):
        os.makedirs(sTargetDirectory)

        # setting up figure saving
        sImagesPath = os.path.join(sProjectRootDirectory, sProjectIdentification, sTargetDirectory, "Images")
        if not os.path.exists(sImagesPath):
            os.makedirs(sImagesPath)

        # Initializing Table
        lsResultsIndex = ['accuracy', 'Precision/recall area under curve', 'F1 score', 'ROC curve area under curve']
        lsResultsColumns = ['Naive Bayes', 'Linear Ridge Regression', 'Linear Lasso Regression',
                            'SVM with Linear Kernel', 'SVM with Gaussian Kernel', 'Random Forest', 'Extremely Random Trees',
                            'Ada Boost Gradient Boosting',
                            'xgBoost Gradient Boosting']
        pdResults = pd.DataFrame(index=lsResultsIndex, columns=lsResultsColumns)

        # create a results frame that stores the performance metrics for each alg tested
        FullProcess = cMachineLearning(xData, yData, xTest, yTest, "Paris Autism", sImagesPath)

        # 1
        aBayesPredicted, skcBayesModel, f64BayesAccuracy, f64BayesPrecisionAUC, f64BayesF1, f64BayesROCAUC = FullProcess.fNaiveBayes()
        pickle.dump(skcBayesModel, open(sTargetDirectory + "/NaiveBayes.p", 'wb'))
        pdResults['Naive Bayes'] = [f64BayesAccuracy, f64BayesPrecisionAUC, f64BayesF1, f64BayesROCAUC]

        #2
        aLinRidgePredicted, skcLinRidgeModel, f64LinRidgeAccuracy, f64LinRidgePrecisionAUC, f64LinRidgeF1, f64LinRidgeROCAUC = FullProcess.fGLM()
        pickle.dump(skcLinRidgeModel, open(sTargetDirectory + "/LinRidge.p", 'wb'))
        pdResults['Linear Ridge Regression'] = [f64LinRidgeAccuracy, f64LinRidgePrecisionAUC, f64LinRidgeF1,
                                                f64LinRidgeROCAUC]
        #3
        aLinLassoPredicted, skcLinLassoModel, f64LinLassoAccuracy, f64LinLassoPrecisionAUC, f64LinLassoF1, f64LinLassoROCAUC = FullProcess.fGLM(
            regularizer="lasso")
        pickle.dump(skcLinLassoModel, open(sTargetDirectory + "/LinLasso.p", 'wb'))
        pdResults['Linear Lasso Regression'] = [f64LinLassoAccuracy, f64LinLassoPrecisionAUC, f64LinLassoF1,
                                                f64LinLassoROCAUC]
        #4
        aLinSVMPredicted, skcLinSVMModel, f64LinSVMAccuracy, f64LinSVMPrecisionAUC, f64LinSVMF1, f64LinSVMROCAUC = FullProcess.fSVM()
        pickle.dump(skcLinSVMModel, open(sTargetDirectory + "/LinSVM.p", 'wb'))
        pdResults['SVM with Linear Kernel'] = [f64LinSVMAccuracy, f64LinSVMPrecisionAUC, f64LinSVMF1, f64LinSVMROCAUC]

        #5
        aGaussSVMPredicted, skcGaussSVMModel, f64GaussSVMAccuracy, f64GaussSVMPrecisionAUC, f64GaussSVMF1, f64GaussSVMROCAUC = FullProcess.fSVM(
            kernel="gaussian")
        pickle.dump(skcGaussSVMModel, open(sTargetDirectory + "/GaussSVC.p", 'wb'))
        pdResults['SVM with Gaussian Kernel'] = [f64GaussSVMAccuracy, f64GaussSVMPrecisionAUC, f64GaussSVMF1,
                                                 f64GaussSVMROCAUC]
        #6
        aRandForPredicted, skcRandForModel, f64RandForAccuracy, f64RandForPrecisionAUC, f64RandForF1, f64RandForROCAUC = FullProcess.fRandForest()
        pickle.dump(skcRandForModel, open(sTargetDirectory + "/RandFor.p", 'wb'))
        pdResults['Random Forest'] = [f64RandForAccuracy, f64RandForPrecisionAUC, f64RandForF1, f64RandForROCAUC]

        #7
        aExRanTreePredicted, skcExRanTreeModel, f64ExRanTreeAccuracy, f64ExRanTreePrecisionAUC, f64ExRanTreeF1, f64ExRanTreeROCAUC = FullProcess.fExtremeRandTrees()
        pickle.dump(skcExRanTreeModel, open(sTargetDirectory + "/ExRanTrees.p", 'wb'))
        pdResults['Extremely Random Trees'] = [f64ExRanTreeAccuracy, f64ExRanTreePrecisionAUC, f64ExRanTreeF1,
                                               f64ExRanTreeROCAUC]
        #8
        aAdaBPredicted, skcAdaBModel, f64AdaBAccuracy, f64AdaBPrecisionAUC, f64AdaBF1, f64AdaBROCAUC = FullProcess.fAdaBoost()
        pickle.dump(skcAdaBModel, open(sTargetDirectory + "/AdaBoost.p", 'wb'))
        pdResults['Ada Boost Gradient Boosting'] = [f64AdaBAccuracy, f64AdaBPrecisionAUC, f64AdaBF1, f64AdaBROCAUC]

        #9
        aXGBoostPredicted, skcXGBoostModel, f64XGBoostAccuracy, f64XGBoostPrecisionAUC, f64XGBoostF1, f64XGBoostROCAUC = FullProcess.fXGradBoost()
        pickle.dump(skcXGBoostModel, open(sTargetDirectory + "/XGBoost.p", 'wb'))
        pdResults['xgBoost Gradient Boosting'] = [f64XGBoostAccuracy, f64XGBoostPrecisionAUC, f64XGBoostF1,
                                                  f64XGBoostROCAUC]

        # The Data saved below is a Pandas Dataframe of the following form:
        #                                   'Machine Learning Model#1'  'Model #2' , etc.
        # accuracy                         |    float64 value         |  float64 value
        # Precision/recall area under curve|    float64 value         |      ...
        # F1 Score                         |        ...               |      ...
        # ROC curve area under curve       |        ...               |      ...

        pdResults.to_pickle(os.path.join(sTargetDirectory, '10xCV_50foldRandSearch_MachineLearningResults.p'))


# if a pickled set of the Connectivity data is available, it loads it,
# otherwise, it calculates the connectivity for every atlas, saves them all
# in a single dictionary, and then pickles it
def fFetchFMRIData(fMRIDataNames, sFileLoc, **ConnectArgOptions):
    if os.path.isfile(os.path.join(sDataPath, sFileLoc + '.p')):

        dTotalConnectivity = pd.read_pickle(os.path.join(sDataPath, sFileLoc + '.p'))
        return dTotalConnectivity

    else:
        def fGetConnectivityData(fMRIDataNames, fMRILabels, sAtlasName):

            PreExtractedFMRI = FeatureExtractor(**ConnectArgOptions)
            PreExtractedFMRI.fit(fMRIDataNames, fMRILabels, sAtlasName)
            Connectivity = PreExtractedFMRI.transform(fMRIDataNames, sAtlasName)
            return Connectivity

        # get the connectivity for each atlas
        dTotalConnectivity = {}
        for sAtlas in fMRIAtlases:
            print('Computing Connectivity for ' + sAtlas)
            pdConnectivity = fGetConnectivityData(fMRIData, fMRILabels, sAtlas)
            pdConnectivity.columns = [sAtlas + "_" + str(col) for col in pdConnectivity.columns]
            dTotalConnectivity[sAtlas] = pdConnectivity

        pickle.dump(dTotalConnectivity, open(sDataPath + '/' + sFileLoc + '.p', 'wb'))
        return dTotalConnectivity


def fPreprocess():
    # Import the raw data
    pdRawData, aRawLabels = get_train_data(path="/project/bioinformatics/DLLab/Cooper"
                                                 "/Jupyter_notebooks/autism-master/autism-master")

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
    pdSMRIData = pdSMRIData.drop('anatomy_select', axis=1)  # discard the QA column since not used for ML fitting
    pdFMRIData = pdFMRIData.iloc[aCombinedGoodStudiesIndex]
    pdFMRIData = pdFMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
    pdFMRIData = pdFMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well

    # Also fetch the corresponding Autism/not Autism labels for the data that
    # passed the QA metric
    pdRawLabels = pd.DataFrame(aRawLabels)
    pdAutLabel = pdRawLabels.iloc[aCombinedGoodStudiesIndex]
    aAutLabel = pdAutLabel.values

    # sMRI processing:
    #    - convert categorical variables to numerical predictors rather than 'male', 'female', etc
    #        - male converted to 0, female to 1
    #        - site where MRI was taken is one-hot encoded
    #    -convert the sMRI data to an array
    #    -append the sMRI data with the categorical variables
    #    -replace NAN values with 0
    #    -normalize the sMRI data

    pdParticipantsData = pdRawData[[col for col in pdRawData.columns if col.startswith('participants')]]
    pdParticipantsData = pdParticipantsData.iloc[aCombinedGoodStudiesIndex]

    aSiteData = pdParticipantsData.participants_site.values
    aSexData = pdParticipantsData.participants_sex.values

    aSiteData = keras.utils.to_categorical(aSiteData)
    aSexData = keras.utils.to_categorical(aSexData == 'F')
    aSexData = np.resize(aSexData, (len(aSexData), 1))
    # now site and gender are ready for being passed into a machine learning model

    # the sMRI data is converted from a pandas dataframe to a numpy matrix
    aSMRIData = pdSMRIData.values

    # Now, we normalize the data
    aSMRIData = sk.preprocessing.normalize(aSMRIData)

    # Next, we combine it all (append columns ) into an 2D array for each algorithm to work on
    aProcessedSMRIData = np.append(aSMRIData, aSiteData, axis=1)
    aProcessedSMRIData = np.append(aProcessedSMRIData, aSexData, axis=1)

    # fill NAN locations with 0's
    aNANloc = np.isnan(aProcessedSMRIData)
    aProcessedSMRIData[aNANloc] = 0

    # now, aProcesedSMRIData is ready to be split into test, training, and validation sets

    ##################################

    # Select out the names of the atlases used
    lsFMRIAtlases = list(pdFMRIData.columns)
    for i in range(len(lsFMRIAtlases)):
        lsFMRIAtlases[i] = lsFMRIAtlases[i][5:]

    # Fetch the fMRI connectivity data and place in a dictionary:
    # each dictionary key is the atlas name

    dFMRIConnectivityData = fFetchFMRIData(pdFMRIData, 'AnatAndFMRIQAConnectivityDictionary')

    # Now, we loop through all the connectivity data and normalize each atlas's conectivity
    # matrices
    for i in range(len(lsFMRIAtlases)):
        dFMRIConnectivityData[lsFMRIAtlases[i]] = sk.preprocessing.normalize(
            dFMRIConnectivityData[lsFMRIAtlases[i]])

    # Append all Connectivity data to the structural data for a good stratified split
    aAllData = aProcessedSMRIData
    for i in range(len(lsFMRIAtlases)):
        aAllData = np.concatenate((aAllData, dFMRIConnectivityData[lsFMRIAtlases[i]]), axis=1)

    cTestTrainSplit = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    iterator=0
    aTrainingIndex=np.zeros((10, int(len(aAutLabel)*0.8)))
    aTestIndex = np.zeros((10, int(len(aAutLabel)*0.2)))
    for train, test in cTestTrainSplit.split(aAllData, aAutLabel):
        aTrainingIndex[iterator,:] = train
        aTestIndex[iterator,:]=test
        iterator=iterator+1

    # Create a Dictionary to hold all the Test and Training data
    dXTrain = {}
    dXTest = {}

    # organize the anatomical data into test and training data
    pdProcessedSMRIData = pd.DataFrame(aProcessedSMRIData)
    dXTrain['anatomy'] = pdProcessedSMRIData.iloc[aTrainingIndex[0,:]].values
    dXTest['anatomy'] = pdProcessedSMRIData.iloc[aTestIndex[0,:]].values
    pdAutLabel = pd.DataFrame(aAutLabel)
    aYTrain = pdAutLabel.iloc[aTrainingIndex[0,:]].values
    aYTest = pdAutLabel.iloc[aTestIndex[0,:]].values

    # Set up a dictionary within the training set dictionary that will
    # contain as keys 'connectivty' and 'combined'. These keys will in turn
    # hold as their values a dictionary with keys 'atlases used' which contain
    # the connectiity matrix and combined connectivity+anatomical volumes.
    # for example, the dictionary dXTrain['connectivity']['msdl'] returns an array
    # that is the connectivity matrix calculated with the 'msdl' atlas. dXTest[
    # 'combined']['power_2011'] would return the matrix from of the Test set
    #  that was a concatenation of the sMRI data and the connectivity calculated
    # with the power_2011 atlas

    dXTrain['connectivity'] = {}
    dXTrain['combined'] = {}
    dXTest['connectivity'] = {}
    dXTest['combined'] = {}

    for i in range(len(lsFMRIAtlases)):
        dXTrain['connectivity'].update({lsFMRIAtlases[i]: dFMRIConnectivityData[lsFMRIAtlases[i]][aTrainingIndex[0,:].astype(int)]})
        dXTest['connectivity'].update({lsFMRIAtlases[i]: dFMRIConnectivityData[lsFMRIAtlases[i]][aTestIndex[0,:].astype(int)]})

        dXTrain['combined'].update({lsFMRIAtlases[i]: np.append(dXTrain['anatomy'],
                                                                dFMRIConnectivityData[lsFMRIAtlases[i]][aTrainingIndex[0,:].astype(int)],
                                                                axis=1)})
        dXTest['combined'].update({lsFMRIAtlases[i]: np.append(dXTest['anatomy'],
                                                               dFMRIConnectivityData[lsFMRIAtlases[i]][aTestIndex[0,:].astype(int)],
                                                               axis=1)})

    return dXTrain, dXTest, aYTrain, aYTest


# Get the Training and Test Data with fPreprocess (see fPreprocess comments for
# details)
dXTrain, dXTest, aYTrain, aYTest = fPreprocess()

# Create a directory to store results in- flagged with the day it was run
# named by parameters for random search with
# internal folders named for the data being used
sTargetDirectory = "10_fold_cv_50_random_hyperparameter_initializations_random_search" +  '0917' #datetime.date.today().strftime('%m''%d')

# Run the analysis on the anatomical data alone and then save the results in a sub-folder to
# sTarget Directory
sRunTag = 'anatomical_only'
fRunAnalysis(dXTrain['anatomy'], dXTest['anatomy'], aYTrain, aYTest, os.path.join(sTargetDirectory, sRunTag))

# Loops through all combinations of:
#   fMRI connectivity data alone used as Training and Test Data
#   fMRI connectivity data combined with structural data used as Training and Test Data
#
#   and then loops through each atlas used to generate the fMRI connectivity data matrices

for sCategory in dXTrain.keys():
    if sCategory != 'anatomy':
        for sAtlas in dXTrain[sCategory].keys():
            sRunTag = sCategory + '_' + sAtlas
            fRunAnalysis(dXTrain[sCategory][sAtlas], dXTest[sCategory][sAtlas], aYTrain, aYTest,
                         os.path.join(sTargetDirectory, sRunTag))
    else:
        fRunAnalysis(dXTrain[sCategory], dXTest[sCategory], aYTrain, aYTest,
                     os.path.join(sTargetDirectory, sRunTag))


