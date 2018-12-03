""" This file takes the preprocessed data and appends the confounding information
(age, sex, MRI site) to each saved feature vector

"""
import pickle as pkl
import numpy as np
import keras as k
import pandas as pd
import sys
import os
import sklearn as sk
import IMPAC_Autism_Run_multiple_test

sAutStartingKit = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master"
sAutDataPath = "/project/bioinformatics/DLLab/shared/Autism_IMPAC/autism-master/data"
sAutCode = "/project/bioinformatics/DLLab/Cooper/Libraries/paris-saclay-cds-ramp-workflow-v0.2.0-41-g31d4037/paris-saclay-cds-ramp-workflow-31d4037"
sys.path.append(sAutStartingKit)
sys.path.append(sAutCode)
sys.path.append(sAutDataPath)

# import last few modules that were in the path added above (sAutismCode)
from problem import get_train_data, get_cv
from download_data import fetch_fmri_time_series

#initialize the path to the current Train/Test split
sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

def fLoadData():

    # Load the master file of the data pre-organized into train, test
    [dXData, dXTest, aYData, aYTest] = pkl.load(open(sDataPath, 'rb'))

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
    aAgeData = np.float32(pdParticipantsData.participants_age.values)

    aSiteData = k.utils.to_categorical(aSiteData)
    aSexData = k.utils.to_categorical(aSexData == 'F')
    aSexData = np.resize(aSexData, (len(aSexData), 1))
    aAgeData = np.resize(aAgeData, (len(aSexData), 1))
    # now site, gender, and age are ready for being passed into a machine learning model

    aConfounders = np.append(aSiteData, aSexData, axis=1)
    aConfounders = np.append(aConfounders, aAgeData, axis=1)

    # the sMRI data is converted from a pandas dataframe to a numpy matrix
    aSMRIData = pdSMRIData.values

    # Now, we normalize the data
    aSMRIData = sk.preprocessing.normalize(aSMRIData)

    # Next, we combine it all (append columns ) into an 2D array for each algorithm to work on
    # aProcessedSMRIData = np.append(aSMRIData, aSiteData, axis=1)
    # aProcessedSMRIData = np.append(aProcessedSMRIData, aSexData, axis=1)
    aProcessedSMRIData = aSMRIData

    # fill NAN locations with 0's
    aNANloc = np.isnan(aProcessedSMRIData)
    aProcessedSMRIData[aNANloc] = 0
    aNANloc = np.isnan(aConfounders)
    aConfounders[aNANloc] = 0
    # now, the structural and confounding variables are ready to be split
    # into the test, training, and validation sets
    return aConfounders, aSMRIData, pdFMRIData, aAutLabel, dXData, dXTest, aYData, aYTest

aConfounders, aSMRIData, pdFMRIData, aAutLabel, dOrigXData, dOrigXTest, aOrigYData, aOrigYTest = fLoadData()

def fProcessFMRI():
    # Select out the names of the atlases used
    lsFMRIAtlases = list(pdFMRIData.columns)
    for i in range(len(lsFMRIAtlases)):
        lsFMRIAtlases[i] = lsFMRIAtlases[i][5:]

    # Fetch the fMRI connectivity data and place in a dictionary:
    # each dictionary key is the atlas name

    dFMRIConnectivityData = IMPAC_Autism_Run_multiple_test.fFetchFMRIData(pdFMRIData, 'AnatAndFMRIQAConnectivityDictionary')

    # Now, we loop through all the connectivity data and normalize each atlas's conectivity
    # matrices
    for i in range(len(lsFMRIAtlases)):
        dFMRIConnectivityData[lsFMRIAtlases[i]] = sk.preprocessing.normalize(
            dFMRIConnectivityData[lsFMRIAtlases[i]])

    return dFMRIConnectivityData, lsFMRIAtlases

dFMRIConnectivityData, lsFMRIAtlases = fProcessFMRI()

def fSplit(lsFMRIAtlases, aAutLabel):
    # Append all Connectivity data to the structural data for a good stratified split
    aAllData = np.append(aConfounders, aSMRIData, axis=1)
    for i in range(len(lsFMRIAtlases)):
        aAllData = np.concatenate((aAllData, dFMRIConnectivityData[lsFMRIAtlases[i]]), axis=1)

    cTestTrainSplit = sk.model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    iterator = 0
    aTrainingIndex = np.zeros((10, int(len(aAutLabel) * 0.8)))
    aTestIndex = np.zeros((10, int(len(aAutLabel) * 0.2)))
    for train, test in cTestTrainSplit.split(aAllData, aAutLabel):
        aTrainingIndex[iterator, :] = train
        aTestIndex[iterator, :] = test
        break

    return aTrainingIndex, aTestIndex

def fOrganizeToDict(aProcessedSMRIData, aAutLabel, lsFMRIAtlases):

    aTrainingIndex, aTestIndex=fSplit(lsFMRIAtlases, aAutLabel)

    # Create a Dictionary to hold all the Test and Training data
    dXTrain = {}
    dXTest = {}

    # organize the anatomical data into test and training data
    pdProcessedSMRIData = pd.DataFrame(aProcessedSMRIData)
    dXTrain['anatomy'] = np.append(aConfounders[aTrainingIndex[0, :].astype(int)],
                                   pdProcessedSMRIData.iloc[aTrainingIndex[0, :]].values,
                                   axis=1)
    dXTest['anatomy'] = np.append(aConfounders[aTestIndex[0, :].astype(int)],
                                  pdProcessedSMRIData.iloc[aTestIndex[0, :]].values,
                                  axis=1)
    pdAutLabel = pd.DataFrame(aAutLabel)
    aYTrain = pdAutLabel.iloc[aTrainingIndex[0, :]].values
    aYTest = pdAutLabel.iloc[aTestIndex[0, :]].values

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
        dXTrain['connectivity'].update(
            {lsFMRIAtlases[i]: np.append(aConfounders[aTrainingIndex[0, :].astype(int)],
                                         dFMRIConnectivityData[lsFMRIAtlases[i]][aTrainingIndex[0, :].astype(int)],
                                         axis=1)})
        dXTest['connectivity'].update(
            {lsFMRIAtlases[i]: np.append(aConfounders[aTestIndex[0, :].astype(int)],
                                         dFMRIConnectivityData[lsFMRIAtlases[i]][aTestIndex[0, :].astype(int)],
                                         axis=1)})

        dXTrain['combined'].update({lsFMRIAtlases[i]: np.append(dXTrain['anatomy'],
                                                                dFMRIConnectivityData[lsFMRIAtlases[i]][
                                                                    aTrainingIndex[0, :].astype(int)],
                                                                axis=1)})
        dXTest['combined'].update({lsFMRIAtlases[i]: np.append(dXTest['anatomy'],
                                                               dFMRIConnectivityData[lsFMRIAtlases[i]][
                                                                   aTestIndex[0, :].astype(int)],
                                                               axis=1)})
    return dXTrain, dXTest

dXTrain, dXTest = fOrganizeToDict(aSMRIData, aAutLabel, lsFMRIAtlases)

aYTrain=aOrigYData
aYTest=aOrigYTest

pkl.dump([dXTrain, dXTest, aYTrain, aYTest],
                open('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestDataWithConfounds.p', 'wb'))


