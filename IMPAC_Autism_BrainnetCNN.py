""" This code takes the Paris IMPAC Autism study data,and
uses the BrainNetCNN (Kawahara 2017) in a transfer learning paradigm

The different transfer learning paradigms that are compared are:

    A:) All layers but decision layer frozen (using the network as a fixed feature extractor)

    B:) Trained decision layer (from step above) and 1 block down
            are unfrozen and retrained (fine-tuning the existing network)

    [The third possibility is to simply use the architecture and retrain the whole model,
        but I will not be performing this due to the small sample size]

The two paradigms are used on each the data in the following ways:

    1:) Structural Data Alone

    2:) fMRI data Alone

    3:) fMRI combined with structural at input level

    4:) fMRI combined with structural at decision level (i.e. the
            results from 1:) and 2:) with a new decision layer)

The performance metrics being used are:

    - accuracy
    - Precision-Recall area under curve
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
import caffe
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
from keras import models, layers, regularizers
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
from keras.utils import to_categorical, normalize
from random import shuffle
from scipy.stats.stats import pearsonr
import _pickle as cPickle
import pickle

np.random.seed(42)

# Set up figure saving
sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
sProjectIdentification = "AutismProject"
sAutismStartingKit = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master"
sDataPath = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master/data"
sAutismCode = "/project/bioinformatics/DLLab/Cooper/Libraries/paris-saclay-cds-ramp-workflow-v0.2.0-41-g31d4037/paris-saclay-cds-ramp-workflow-31d4037"
sBrainNetCNNCode = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master'
sBrainNetCNNCode2 = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master/ann4brains'
sTargetDirectory = os.path.join(sProjectRootDirectory, sProjectIdentification, 'Neural_Network_Run1')
if not os.path.exists(sTargetDirectory):
    os.makedirs(sTargetDirectory)
sImagesPath = os.path.join(sTargetDirectory, "Images")
if not os.path.exists(sImagesPath):
    os.makedirs(sImagesPath)


sys.path.append(sBrainNetCNNCode)
sys.path.append(sBrainNetCNNCode2)
sys.path.append(sAutismStartingKit)
sys.path.append(sAutismCode)
sys.path.append(sDataPath)

# import last few modules that were in the path added above (sAutismCode)
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from problem import get_train_data, get_cv
from download_data import fetch_fmri_time_series
# import BrainNetCNNCode
from ann4brains.nets import BrainNetCNN

# Initialize the pandas dataframe for information to be stored
lsResultsIndex = ['accuracy', 'Precision/recall area under curve', 'F1 score', 'ROC curve area under curve']
lsResultsColumns = ['BrainNetCNN', 'ConvolutionalNet', 'DenseNet']
pdResults = pd.DataFrame(index=lsResultsIndex, columns=lsResultsColumns)

# fetching the entire dataset
RawData, RawLabels = get_train_data(path="/project/bioinformatics/DLLab/Cooper/Jupyter_notebooks/autism-master/autism-master")
# Next, we narrow down the whole dataset to the fMRI data only
fMRIData = RawData
fMRILabels = RawLabels
fMRIData = RawData[[col for col in RawData.columns if col.startswith('fmri')]]
# Here we select only the columns of fMRI_select=1 (1=fMRI present, 0=fMRI absent)
GoodStudiesIndex = np.where(RawData.fmri_select != 0)
fMRIData = fMRIData.iloc[GoodStudiesIndex[0]]
fMRILabels = fMRILabels[GoodStudiesIndex[0]]

fMRIData = fMRIData.drop('fmri_select', axis=1)  # discard the QA column since not used for ML fitting
fMRIData = fMRIData.drop('fmri_motions', axis=1)  # discard the motion QA column as well

# Now we pull out the names of the atlases used from the fMRI data
fMRIAtlases = list(fMRIData.columns)
for i in range(len(fMRIAtlases)):
    fMRIAtlases[i] = fMRIAtlases[i][5:]


# Next we calculate the connectivity matrices by loading the datasets and running this analysis

# This function was included in the IMPAC autism starting set, I altered it to grab the data from the correct folder
def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(os.path.join('/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master', subject_filename),
                                 header=None).values
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

dTotalConnectivity = fFetchFMRIData(fMRIData, 'TangentMatrixQAfMRIsMRI', ConnectivityMetric='tangent', ConnectivityVector=False)

# next, we normalize the data
for sAtlas in fMRIAtlases:
    dTotalConnectivity[sAtlas] = normalize(dTotalConnectivity[sAtlas].values)
    dTotalConnectivity[sAtlas] = dTotalConnectivity[sAtlas].astype('float32')


UnprocessedData = dTotalConnectivity['msdl']
ProcessedData = UnprocessedData.reshape(UnprocessedData.shape[0], int(np.sqrt(UnprocessedData.shape[1])), int(np.sqrt(UnprocessedData.shape[1])))
ProcessedTruth = fMRILabels.astype('float32')


# then we split the data into fitting data, validation data, and test data
xData, xTest, yData, yTest = train_test_split(ProcessedData, ProcessedTruth, test_size=0.2)
xData, xVal, yData, yVal = train_test_split(xData, yData, test_size=0.2)

# Here we expand the dimensions for the neural net to work properly
# the added 1st dimension is the number of channels from the
# patient, in this case, all measurements are of one connectivity
# 'channel', not multiple as RGB images would be.

# The required dimensions for BrainNetCNN is size
# N x C x H x W, where N is the number of samples, C is
# the number of channels in each sample, and, H and W are the
# spatial dimensions for each sample.
xData = np.expand_dims(xData, axis=1)
xTest = np.expand_dims(xTest, axis=1)
xVal = np.expand_dims(xVal, axis=1)

# initializing the architexture
BrainNetArch = [
    ['e2n', {'n_filters': 16,
             'kernel_h': xData.shape[2], #2
             'kernel_w': xData.shape[3]}], #3
    ['dropout', {'dropout_ratio': 0.5}],
    ['relu', {'negative_slope': 0.33}],
    ['fc', {'n_filters': 30}],
    ['relu', {'negative_slope': 0.33}],
    ['out', {'n_filters': 1}]
]

BrainNetFullNetwork = BrainNetCNN('practice', BrainNetArch)
BrainNetFullNetwork.fit(xData, yData, xVal, yVal)
yPredicted = BrainNetFullNetwork.predict(xTest)

# Now we print out the performance by several metrics
BrainNetROCAUCScore = roc_auc_score(yTest, yPredicted)
precisions, recalls, thresholds = precision_recall_curve(yTest, yPredicted)
BrainNetPRAUCScore = auc(recalls, precisions)
BrainNetF1Score = f1_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
BrainNetAccuracyScore = accuracy_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))), normalize=True)
print('ROC AUC: ', BrainNetROCAUCScore, ' Precision/Recall Auc: ', BrainNetPRAUCScore, ' F1 Score: ', BrainNetF1Score, ' Accuracy: ', BrainNetAccuracyScore)

#Then we save the performace metrics
pickle.dump(BrainNetFullNetwork, open(sTargetDirectory + "/BrainNet.p", 'wb'))
pdResults['BrainNetCNN'] = [BrainNetAccuracyScore, BrainNetPRAUCScore, BrainNetF1Score, BrainNetROCAUCScore]


################# Convolutional Network ###################

# As the data is entered in the same form as above, we can start with
# creating our architexture
ConvNetArch = models.Sequential()
ConvNetArch.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', data_format='channels_first', input_shape=(xData.shape[1], xData.shape[2], xData.shape[3])))
ConvNetArch.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
ConvNetArch.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
ConvNetArch.add(layers.MaxPooling2D(pool_size=(2, 2)))
ConvNetArch.add(layers.Flatten())
ConvNetArch.add(layers.Dense(32, activation='relu'))
ConvNetArch.add(layers.Dense(1, activation='sigmoid'))
ConvNetArch.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

ConvNetArch.fit(xData, yData, epochs=10, batch_size=256, validation_data=(xVal, yVal))
yPredicted = ConvNetArch.predict(xTest)

# Now we print out the performance by several metrics
ConvNetROCAUCScore = roc_auc_score(yTest, yPredicted)
precisions, recalls, thresholds = precision_recall_curve(yTest, yPredicted)
ConvNetPRAUCScore = auc(recalls, precisions)
ConvNetF1Score = f1_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
ConvNetAccuracyScore = accuracy_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
print('ROC AUC: ', ConvNetROCAUCScore, ' Precision/Recall Auc: ', ConvNetPRAUCScore, ' F1 Score: ', ConvNetF1Score, ' Accuracy: ', ConvNetAccuracyScore)

#Then we save the performace metrics
#pickle.dump(ConvNetArch, open(sTargetDirectory + "/ConvNet.p", 'wb'))
pdResults['ConvolutionalNet'] = [ConvNetAccuracyScore, ConvNetPRAUCScore, ConvNetF1Score, ConvNetROCAUCScore]

#################### Dense Network ######################

# First, we collapse the extra dimension added above
xData = np.squeeze(xData, axis=1)
xTest = np.squeeze(xTest, axis=1)
xVal = np.squeeze(xVal, axis=1)

# We must reshape the data into a vector rather than a square matrix - further, we collapse the lower triangle
# of the connectivity because the matrix is symmetric
def fUpperTriangVectorize(x):
    UpperTriangX = np.zeros((int(x.shape[0]), int((x.shape[1]*(x.shape[2]+1)/2))))
    for index in range(x.shape[0]):
        UpperTriangIndex = np.triu(x[index,:,:]).nonzero()
        UpperTriangVect = x[index, UpperTriangIndex[0], UpperTriangIndex[1]]
        UpperTriangX[index, :] = UpperTriangVect
    return UpperTriangX

xData = fUpperTriangVectorize(xData)
xTest = fUpperTriangVectorize(xTest)
xVal = fUpperTriangVectorize(xVal)

#initialize the model
DenseNetArch = models.Sequential()
DenseNetArch.add(layers.Dense(16, kernel_regularizer=regularizers.l1(0.001), activation='relu', input_shape=(xData.shape[1],)))
DenseNetArch.add(layers.Dropout(0.5))
DenseNetArch.add(layers.Dense(32, kernel_regularizer=regularizers.l1(0.001), activation='relu', input_shape=(xData.shape[1],)))
DenseNetArch.add(layers.Dropout(0.5))
DenseNetArch.add(layers.Dense(1, activation='sigmoid'))
DenseNetArch.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

DenseNetArch.fit(xData, yData, epochs=10, batch_size=256, validation_data=(xVal, yVal))
yPredicted = DenseNetArch.predict(xTest)

# Now we print out the performance by several metrics
DenseNetROCAUCScore = roc_auc_score(yTest, yPredicted)
precisions, recalls, thresholds = precision_recall_curve(yTest, yPredicted)
DenseNetPRAUCScore = auc(recalls, precisions)
DenseNetF1Score = f1_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
DenseNetAccuracyScore = accuracy_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
print('ROC AUC: ', DenseNetROCAUCScore, ' Precision/Recall Auc: ', DenseNetPRAUCScore, ' F1 Score: ', DenseNetF1Score, ' Accuracy: ', DenseNetAccuracyScore)


#Then we save the performace metrics
# pickle.dump(DenseNetArch, open(sTargetDirectory + "/ConvNet.p", 'wb'))
pdResults['DenseNet'] = [DenseNetAccuracyScore, DenseNetPRAUCScore, DenseNetF1Score, DenseNetROCAUCScore]

# Save the pd file of all performance metrics
pdResults.to_pickle(os.path.join(sTargetDirectory, 'Neural_Network_Metrics.p'))

# Plot the performances
from IMPAC_Autism_MLFigureGenerator import fPlot

lsAlgorithm = lsResultsColumns
sTitle = "Performance of Network by: "
for i in range(4):
    fPlot(i, pdResults, lsAlgorithm, sTitle, sImagesPath)