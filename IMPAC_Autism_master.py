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
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict,cross_validate, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, auc, f1_score
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from nilearn.connectome import ConnectivityMeasure
from sklearn.naive_bayes import GaussianNB
from keras.utils import to_categorical
from random import shuffle
import pickle

np.random.seed(42)

# Set up figure saving
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

def fSaveFigure(sFigureIdentification, sImagesPath, bTightLayout=True, sFigureExtension="png", intResolution = 300):
    sPath = os.path.join(sImagesPath, sFigureIdentification + "." + sFigureExtension)
    print("Saving figure ", sFigureIdentification)
    if bTightLayout:
        plt.tight_layout()
        plt.savefig(sPath, format=sFigureExtension, dip=intResolution)

def fNormalize(dataset):
    f64mean = dataset.mean(axis=0)
    f64std = dataset.std(axis=0)
    f32norm_dataset = np.float32(dataset-f64mean)/(f64std + 0.000001)
    return f32norm_dataset

# for f(10, [1,2,3]) returns [10**1, 10**2, 10**3]
def fElementWiseExponential(f64Element, aArray):
    aExp = f64Element*np.ones(len(aArray))
    for i in range(len(aExp)):
        aExp[i] = aExp[i]**aArray[i]
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

        if fMRI==True:
            cClassifier=make_pipeline(FeatureExtractor, cClassifier)

        if dParamDistributions is not None:
            cRandSearch = RandomizedSearchCV(
                cClassifier,
                dParamDistributions,
                cv=10, #change to 10
                n_iter=50, #change to 50
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

        self.fPlotNormalizedConfusionMatrix(aRoundedPredicted,
                                       (sPrefix + "Classifier Confusion Matrix for " + self.data_name + " Dataset"))
        plt.close()
        f64PRAUC, f64F1score = self.fPrecisionRecall(aPredicted,
                                              (sPrefix + "Classifier Precision vs Recall for " + self.data_name + " Dataset"))
        plt.close()
        f64ROCAUC = self.fROC(aPredicted,
                        (sPrefix + "Classifier ROC Curve for " + self.data_name + " Dataset"))
        plt.close()

        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    #Generates a ROC curve AUC score and plots the curve if figure name is given
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
    def fPlotNormalizedConfusionMatrix(self, yPredicted, figure_name):
        aConfusionMatrix = confusion_matrix(self.yTest, yPredicted)
        aConfusionMatrix = aConfusionMatrix.astype('float')/aConfusionMatrix.sum(axis=1)
        if figure_name is not None:
            plt.matshow(aConfusionMatrix)
            plt.title(figure_name)
            fSaveFigure(figure_name, self.images_path)

    #generates Naive Bayes classifier and process it
    def fNaiveBayes(self):
        cNaiveBayesClassifier=GaussianNB()

        sPrefix = "Naive Bayes "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cNaiveBayesClassifier, None, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates GLM classifier with either ridge or lasso regularization and process it
    def fGLM(self, regularizer="ridge"):

        if regularizer == "ridge":
            cGLMClassifier = linear_model.Ridge()
            dParamDistributions = {
                'alpha': 10**np.random.uniform(-5, 1, 100),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
            sPrefix = "GLM with Ridge Regularization "

        elif regularizer == "lasso":
            cGLMClassifier = linear_model.Lasso()
            dParamDistributions = {
                'alpha': 10**np.random.uniform(-5, 1, 100),
                'max_iter': np.random.uniform(1000, 100000, 100),
            }
            sPrefix = "GLM with Lasso Regularization "

        else:
            print("Please specify 'ridge' or 'lasso' regularization")

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cGLMClassifier, dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates SVM classifier with either a linear or gaussian kernel and process it
    def fSVM(self, kernel="linear"):

        if kernel == "linear":
            cSVMClassifier = LinearSVC()
            dParamDistributions = {
                'C': 10**np.random.uniform(-4, 4, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
            sPrefix = "Linear "

        elif kernel == "gaussian":
            cSVMClassifier = SVC()
            dParamDistributions = {
                'C': 10**np.random.uniform(-4, 4, 100),
                'gamma': 10**np.random.uniform(-2, 2, 100),
                'max_iter': np.random.uniform(10000, 100000, 100)
            }
            sPrefix = "Gaussian Kernel "

        else:
            print("Please specify 'linear' or 'gaussian' kernel")

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cSVMClassifier, dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates Random Forest classifier and process it
    def fRandForest(self):

        cRandForestClassifier = RandomForestClassifier()
        dParamDistributions = {'n_estimators': np.round(5*10**np.random.uniform(1, 3, 100)).astype(int),
                               'max_leaf_nodes': np.random.randint(5, 50, 100)}
        sPrefix = "Random Forest "
        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cRandForestClassifier,dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates an Extremely Random Trees classifier and process it
    def fExtremeRandTrees(self):

        cExtremeRandTreesClassifier = ExtraTreesClassifier()
        dParamDistributions = {
            'n_estimators': np.round(5*10**np.random.uniform(1, 3, 100)).astype(int),
            'max_leaf_nodes': np.random.randint(5, 50, 100),
        }
        sPrefix = "Extremely Random Trees "
        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cExtremeRandTreesClassifier, dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # generates an AdaBoost Gradient Boosting classifier and process it
    def fAdaBoost(self):

        cAdaBoostClass = AdaBoostClassifier()
        dParamDistributions = {
            'n_estimators': np.round(2*10**np.random.uniform(1, 3, 100)).astype(int),
            'learning_rate': np.random.uniform(0.1, 0.9, 100),
        }
        sPrefix = "Ada Boost "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cAdaBoostClass, dParamDistributions, sPrefix)
        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

    # #generates an xgBoost gradient boosting classifier and process it
    def fXGradBoost(self):
        cXGradBoost = XGBClassifier()

        dParamDistributions = {
            'n_estimators': np.round(5*10**np.random.uniform(0.1, 3, 100)).astype(int),
            'max_depth': np.random.randint(1, 10, 100),
            'subsample': np.random.uniform(0.2, 0.8, 100),
            'colsample_bytree': np.random.uniform(0.2, 1, 100),
            'learning_rate': 10**np.random.uniform(-2, 0, 100),
        }

        sPrefix = "XGradient Boost "

        aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC = self.fProcessData(cXGradBoost, dParamDistributions, sPrefix)

        return aPredicted, skcModel, f64Accuracy, f64PRAUC, f64F1score, f64ROCAUC

# Getting the data

RawData, RawLabels = get_train_data(path="/project/bioinformatics/DLLab/Cooper/Jupyter_notebooks/autism-master/autism-master")

###############  sMRI (anatomical MRI)  ###############  
#We select the anatomical data alone
AnatData = RawData[[col for col in RawData.columns if col.startswith('anatomy')]]

# Reselect only the trials that were acceptable (2) or good (1) while throwing out the
# (0) no MPRAGE available and (3) MPRAGE quality is unsatisfactory studies (subject's worth of data)

GoodStudiesIndex = np.where(AnatData.anatomy_select != (0 or 3))
AnatData = AnatData.iloc[GoodStudiesIndex[0]]
AnatLabels = RawLabels[GoodStudiesIndex[0]]
AnatData = AnatData.drop('anatomy_select', axis=1)
NormAnatData = fNormalize(AnatData.values)

# Next, we separate out the categorical variables (imaging site, gender) to one-hot encode them as something the
# algorithm will handle better. 
ParticipantsData = RawData[[col for col in RawData.columns if col.startswith('participants')]]
ParticipantsData = ParticipantsData.iloc[GoodStudiesIndex[0]]

SiteData = ParticipantsData.participants_site.values
SexData = ParticipantsData.participants_sex.values
AgeData = np.float32(ParticipantsData.participants_age.values)

SiteData = to_categorical(SiteData)
SiteData = np.resize(SiteData, (len(SiteData),1))
SexData = to_categorical(SexData == 'F')
SexData = np.resize(SexData, (len(SexData),1))
AgeData = fNormalize(AgeData)
AgeData = np.resize(AgeData, (len(AgeData),1))
#now site, sex, and age are ready for being passed into a machine learning model

#Next, we combine it all (append columns ) into an 2D array for each algorithm to work on
ProcessedData=NormAnatData
# ProcessedData = np.append(NormAnatData, SiteData, axis=1)
# ProcessedData = np.append(ProcessedData, AgeData, axis=1)
# ProcessedData = np.append(ProcessedData, SexData, axis=1)
aConfounders=np.append(SiteData, AgeData, axis=1)
aConfounders=np.append(aConfounders, SexData, axis=1)
#fill NAN locations with 0's
NANloc = np.isnan(aConfounders)
aConfounders[NANloc] = 0

#fill NAN locations with 0's
NANloc = np.isnan(ProcessedData)
ProcessedData[NANloc] = 0
ProcessedTruth = AnatLabels

#xData, xTest, yData, yTest = train_test_split(ProcessedData, ProcessedTruth, test_size=0.2)

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
    #
    aLinRidgePredicted, skcLinRidgeModel, f64LinRidgeAccuracy, f64LinRidgePrecisionAUC, f64LinRidgeF1, f64LinRidgeROCAUC = FullProcess.fGLM()
    pickle.dump(skcLinRidgeModel, open(sTargetDirectory + "/LinRidge.p", 'wb'))
    pdResults['Linear Ridge Regression'] = [f64LinRidgeAccuracy, f64LinRidgePrecisionAUC, f64LinRidgeF1, f64LinRidgeROCAUC]
    #
    aLinLassoPredicted, skcLinLassoModel, f64LinLassoAccuracy, f64LinLassoPrecisionAUC, f64LinLassoF1, f64LinLassoROCAUC = FullProcess.fGLM(regularizer="lasso")
    pickle.dump(skcLinLassoModel, open(sTargetDirectory + "/LinLasso.p", 'wb'))
    pdResults['Linear Lasso Regression'] = [f64LinLassoAccuracy, f64LinLassoPrecisionAUC, f64LinLassoF1, f64LinLassoROCAUC]
    #
    aLinSVMPredicted, skcLinSVMModel, f64LinSVMAccuracy, f64LinSVMPrecisionAUC, f64LinSVMF1, f64LinSVMROCAUC = FullProcess.fSVM()
    pickle.dump(skcLinSVMModel, open(sTargetDirectory + "/LinSVM.p", 'wb'))
    pdResults['SVM with Linear Kernel'] = [f64LinSVMAccuracy, f64LinSVMPrecisionAUC, f64LinSVMF1, f64LinSVMROCAUC]
    # 5
    aGaussSVMPredicted, skcGaussSVMModel, f64GaussSVMAccuracy, f64GaussSVMPrecisionAUC, f64GaussSVMF1, f64GaussSVMROCAUC = FullProcess.fSVM(kernel="gaussian")
    pickle.dump(skcGaussSVMModel, open(sTargetDirectory + "/GaussSVC.p", 'wb'))
    pdResults['SVM with Gaussian Kernel'] = [f64GaussSVMAccuracy, f64GaussSVMPrecisionAUC, f64GaussSVMF1, f64GaussSVMROCAUC]
    #
    aRandForPredicted, skcRandForModel, f64RandForAccuracy, f64RandForPrecisionAUC, f64RandForF1, f64RandForROCAUC = FullProcess.fRandForest()
    pickle.dump(skcRandForModel, open(sTargetDirectory + "/RandFor.p", 'wb'))
    pdResults['Random Forest'] = [f64RandForAccuracy, f64RandForPrecisionAUC, f64RandForF1, f64RandForROCAUC]
    #
    aExRanTreePredicted, skcExRanTreeModel, f64ExRanTreeAccuracy, f64ExRanTreePrecisionAUC, f64ExRanTreeF1, f64ExRanTreeROCAUC = FullProcess.fExtremeRandTrees()
    pickle.dump(skcExRanTreeModel, open(sTargetDirectory + "/ExRanTrees.p", 'wb'))
    pdResults['Extremely Random Trees'] = [f64ExRanTreeAccuracy, f64ExRanTreePrecisionAUC, f64ExRanTreeF1, f64ExRanTreeROCAUC]
    #
    aAdaBPredicted, skcAdaBModel, f64AdaBAccuracy, f64AdaBPrecisionAUC, f64AdaBF1, f64AdaBROCAUC = FullProcess.fAdaBoost()
    pickle.dump(skcAdaBModel, open(sTargetDirectory + "/AdaBoost.p", 'wb'))
    pdResults['Ada Boost Gradient Boosting'] = [f64AdaBAccuracy, f64AdaBPrecisionAUC, f64AdaBF1, f64AdaBROCAUC]
    # 9
    aXGBoostPredicted, skcXGBoostModel, f64XGBoostAccuracy, f64XGBoostPrecisionAUC, f64XGBoostF1, f64XGBoostROCAUC = FullProcess.fXGradBoost()
    pickle.dump(skcXGBoostModel, open(sTargetDirectory + "/XGBoost.p", 'wb'))
    pdResults['xgBoost Gradient Boosting'] = [f64XGBoostAccuracy, f64XGBoostPrecisionAUC, f64XGBoostF1, f64XGBoostROCAUC]

    # The Data saved below is a Pandas Dataframe of the following form:
    #                                   'Machine Learning Model#1'  'Model #2' , etc.
    # accuracy                         |    float64 value         |  float64 value
    # Precision/recall area under curve|    float64 value         |      ...
    # F1 Score                         |        ...               |      ...
    # ROC curve area under curve       |        ...               |      ...


    pdResults.to_pickle(os.path.join(sTargetDirectory, '10xCV_50foldRandSearch_MachineLearningResults.p'))

sTargetDirectory="anatomical_only_machine_learning_10x_cross_val_50_fold_Rand_Search_ROC_AUC_Metric"
# fRunAnalysis(xData, xTest, yData, yTest, sTargetDirectory)

###############  fMRI  ###############  
fMRIData = RawData
fMRILabels = RawLabels
# Here we select only the columns of fMRI_select=1 (1=fMRI present, 0=fMRI absent)
fMRIData = RawData[[col for col in RawData.columns if col.startswith('fmri')]]
GoodStudiesIndex = np.where(RawData.fmri_select != 0)
fMRIData = fMRIData.iloc[GoodStudiesIndex[0]]
fMRILabels = fMRILabels[GoodStudiesIndex[0]]

fMRIData = fMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
fMRIData = fMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well

 # Select out the names of the atlases used
fMRIAtlases = list(fMRIData.columns)
for i in range(len(fMRIAtlases)):
    fMRIAtlases[i] = fMRIAtlases[i][5:]


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

# if a pickled set of the Connectivity data is available, it loads it,
# otherwise, it calculates the connectivity for every atlas, saves them all
# in a single dictionary, and then pickles it
def fFetchFMRIData(fMRIDataNames, sFileLoc, **ConnectArgOptions):

    if os.path.isfile(os.path.join(sDataPath, sFileLoc + '.p')):

        dTotalConnectivity = pd.read_pickle(os.path.join(sDataPath, sFileLoc + '.p'))
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

        pickle.dump(dTotalConnectivity, open(sDataPath + '/' + sFileLoc + '.p', 'wb'))
        return dTotalConnectivity

dTotalConnectivity = fFetchFMRIData(fMRIData, 'TangentVectorQAfMRIsMRI', ConnectivityMetric='tangent', ConnectivityVector=True)
aMSDLConnectivity = dTotalConnectivity['msdl']
xData, xTest, yData, yTest = train_test_split(aMSDLConnectivity, fMRILabels, test_size=0.2)

sTargetDirectory="msdl_fMRI_connectivity_machine_learning_10x_cross_val_50_fold_Rand_Search_ROC_AUC_Metric"
# fRunAnalysis(xData, xTest, yData, yTest, sTargetDirectory)

########## Matching the Example data ###########

sTargetDirectory = "Duplicate_Example"
sTargetDirectory = os.path.join(sProjectRootDirectory, sProjectIdentification, sTargetDirectory)
if not os.path.exists(sTargetDirectory):
    os.makedirs(sTargetDirectory)

# for the example data, they used only the msdl atlas connectivity
aMSDLConnectivity = dTotalConnectivity['msdl']

#split the data into test and training sets
xData, xTest, yData, yTest = train_test_split(aMSDLConnectivity, fMRILabels, test_size=0.2)
xData, xVal, yData, yVal = train_test_split(xData, yData, test_size=0.2)

#perform a logistic regression
cLogRegression = linear_model.LogisticRegression(C=1.0)
cLogRegression.fit(xData, yData)

#predict the values and get an accuracy and ROC score
aTrainPredicted = cLogRegression.predict(xData)
aValPredicted = cLogRegression.predict(xVal)
f64TrainAcc = accuracy_score(yData, aTrainPredicted)
f64ValAcc = accuracy_score(yVal, aValPredicted)

aTrainPredicted = cLogRegression.predict_proba(xData)[:,1]
aValPredicted = cLogRegression.predict_proba(xVal)[:,1]
f64TrainROC = roc_auc_score(yData, aTrainPredicted)
f64ValROC = roc_auc_score(yVal, aValPredicted)
ReplicateLogRegResult = [f64TrainAcc, f64ValAcc, f64TrainROC, f64ValROC]

# The example logistic regression they performed had:
# Training score ROC-AUC: 1.000 +- 0.000
# Validation score ROC-AUC: 0.612 +- 0.019
# Training score accuracy: 1.000 +- 0.000
# Validation score accuracy: 0.587 +- 0.021

print("Train Acc: " + str(f64TrainAcc)
      + " Val Acc: " + str(f64ValAcc)
      + " Train ROC AUC: " + str(f64TrainROC)
      + " Val ROC AUC: " + str(f64ValROC)
     )


pickle.dump(ReplicateLogRegResult, open(sTargetDirectory + "/ReplicateLogRegResult.p", 'wb'))

# They also performed a random forest classifier with
# combined structural and fMRI data, so we also repeat
# that here

# re-grab the data
fMRIData = RawData[[col for col in RawData.columns if col.startswith('fmri')]]
sMRIData = RawData[[col for col in RawData.columns if col.startswith('anat')]]

# perform the same QA steps as above
fMRIGoodStudiesIndex = np.where(RawData.fmri_select != 0)
sMRIGoodStudiesIndex = np.where(RawData.anatomy_select != (0 or 3))
combinedGoodStudiesIndex = np.intersect1d(fMRIGoodStudiesIndex, sMRIGoodStudiesIndex)

sMRIData = sMRIData.iloc[combinedGoodStudiesIndex]
sMRIData = sMRIData.drop('anatomy_select', axis=1)
fMRIData = fMRIData.iloc[combinedGoodStudiesIndex]
fMRIData = fMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
fMRIData = fMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well
dTotalConnectivity = fFetchFMRIData(fMRIData, 'AnatAndFMRIQAConnectivityDictionary')
aMSDLConnectivity = dTotalConnectivity['msdl']
aProcessedData = np.concatenate((sMRIData, aMSDLConnectivity), axis=1)
sfMRILabels = RawLabels[combinedGoodStudiesIndex]


#split the data into test and training sets
xData, xTest, yData, yTest = train_test_split(aProcessedData, sfMRILabels, test_size=0.2)
xData, xVal, yData, yVal = train_test_split(xData, yData, test_size=0.2)

# fit the Random Forest Classifier with the same hyperparameters
cRandForestClassifier = RandomForestClassifier(n_estimators=100)
cRandForestClassifier.fit(xData, yData)

#predict the values and get an accuracy and ROC score
aTrainPredicted = cRandForestClassifier.predict(xData)
aValPredicted = cRandForestClassifier.predict(xVal)
f64TrainAcc = accuracy_score(yData, aTrainPredicted)
f64ValAcc = accuracy_score(yVal, aValPredicted)

aTrainPredicted = cRandForestClassifier.predict_proba(xData)[:,1]
aValPredicted = cRandForestClassifier.predict_proba(xVal)[:,1]
f64TrainROC = roc_auc_score(yData, aTrainPredicted)
f64ValROC = roc_auc_score(yVal, aValPredicted)
ReplicateRandForestResult = [f64TrainAcc, f64ValAcc, f64TrainROC, f64ValROC]

# The Random Forest Classification they performed had:
# Training score ROC-AUC: 1.000 +- 0.000
# Validation score ROC-AUC: 0.655 +- 0.028
# Training score accuracy: 1.000 +- 0.000
# Validation score accuracy: 0.613 +- 0.033


print("Train Acc: " + str(f64TrainAcc)
      + " Val Acc: " + str(f64ValAcc)
      + " Train ROC AUC: " + str(f64TrainROC)
      + " Val ROC AUC: " + str(f64ValROC)
     )

pickle.dump(ReplicateRandForestResult, open(sTargetDirectory + "/ReplicateRandForest.p", 'wb'))
