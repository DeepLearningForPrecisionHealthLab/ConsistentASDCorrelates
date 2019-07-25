""" This file takes the preprocessed data and appends the confounding information
(age, sex, MRI site) to each saved feature vector

"""
import pickle as pkl
import numpy as np
import keras as k
import pandas as pd
import sys
import os
import datetime
import sklearn as sk
import IMPAC_Autism_Run_multiple_atlases_v2

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

lsAtlases=['basc064', 'basc122', 'basc197', 'craddock_scorr_mean', 'harvard_oxford_cort_prob_2mm', 'msdl', \
           'power_2011']

def fFlattenEC(aEC):
    """
    Returns a flattened asymmetric EC matrix, with diagonals deleted
    :param aEC: EC array
    :return: vEC: EC vector
    """
    iLen=aEC.shape[0]
    vSkips=np.arange(0,(iLen**2), iLen+1)
    vEC=aEC.flatten()
    vEC=np.delete(vEC, vSkips)
    return vEC

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

    lsSites = []
    for i in range(35):
        lsSites.append(f'Site{i+1:02}')

    lsSites.append('Sex(F=1)')
    lsSites.append('Age')

    pdConfounders = pd.DataFrame(columns=lsSites, index=pdSMRIData.index)

    for i in range(35):
        pdConfounders.iloc[:,i] = aSiteData[:,i]

    pdConfounders['Sex(F=1)'] = aSexData[:,0]
    pdConfounders['Age'] = aAgeData[:,0]

    aConfounders = np.append(aSiteData, aSexData, axis=1)
    aConfounders = np.append(aConfounders, aAgeData, axis=1)

    # the sMRI data is converted from a pandas dataframe to a numpy matrix
    aSMRIData = pdSMRIData.values

    # Now, we normalize the data
    aSMRIData = sk.preprocessing.normalize(aSMRIData)

    iRow, iCol = aSMRIData.shape

    for nRow in range(iRow):
        for nCol in range(iCol):
            pdSMRIData.iloc[nRow, nCol]=aSMRIData[nRow, nCol]

    # Next, we combine it all (append columns ) into an 2D array for each algorithm to work on
    # aProcessedSMRIData = np.append(aSMRIData, aSiteData, axis=1)
    # aProcessedSMRIData = np.append(aProcessedSMRIData, aSexData, axis=1)
    aProcessedSMRIData = aSMRIData

    # fill NAN locations with 0's
    pdConfounders = pdConfounders.fillna(0)
    pdSMRIData = pdSMRIData.fillna(0)
    # now, the structural and confounding variables are ready to be split
    # into the test, training, and validation sets
    return pdConfounders, pdSMRIData, pdFMRIData, aAutLabel, dXData, dXTest, aYData, aYTest

def fFetchLSGC(pdFMRIData, sLoc=None):
    """
    Fetches the lsGC scores of patients
    :param pdFMRIData: pandas dataframe of patients being used (will be using the index)
    :return: dlsGCReduced: dictionary of lsGC scores by patient, including only the patients in the index of pdFMRIData
    """
    # Load the lsGC Data
    if sLoc==None:
        sLoc = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/lsGC'
    dlsGCFMRIData = pkl.load(open(os.path.join(sLoc, 'dlsGCData_default_param.p'), 'rb'))
    dFlattenedGC = {}

    # extract the non-diagonal elements into a flattened feature vector
    for sKey1 in dlsGCFMRIData.keys():
        dFlattenedGC.update({sKey1: {}})
        for sKey2 in dlsGCFMRIData[sKey1].keys():
            aGC = dlsGCFMRIData[sKey1][sKey2]
            dFlattenedGC[sKey1].update({sKey2: fFlattenEC(aGC)})

    # format the lsGC results into a data frame
    dlsGC = {}
    for sAtlas in lsAtlases:
        dlsGC.update({sAtlas: pd.DataFrame(dFlattenedGC[sAtlas]).transpose()})

    # select only the ones that pass the quality control metric
    dlsGCReduced = {}
    for sAtlas in lsAtlases:
        dlsGCReduced.update({sAtlas: dlsGC[sAtlas].loc[map(str, pdFMRIData.index)]})

    return dlsGCReduced

def fRenamelsGC(dlsGC):
    # rename the columns by Roi-Roi connections
    for sAtlas in dlsGC.keys():
        iLen = int((1 + (1 + 4 * len(dlsGC[sAtlas].iloc[0])) ** (1 / 2)) / 2)
        lsNames=list()
        for iRoi1 in range(iLen):
            for iRoi2 in range(iLen):
                if not iRoi1 == iRoi2:
                    lsNames.append(f'ROI{iRoi1+1:03}-ROI{iRoi2+1:03}')

        dlsGC[sAtlas].columns=lsNames

    #return lsGC measures in dictionary
    return dlsGC

pdConfounders, pdSMRIData, pdFMRIData, aAutLabel, dOrigXData, dOrigXTest, aOrigYData, aOrigYTest = fLoadData()

dlsGCReduced = fFetchLSGC(pdFMRIData)
dlsGCReduced = fRenamelsGC(dlsGCReduced)

def fProcessFMRI():
    # Select out the names of the atlases used
    lsFMRIAtlases = list(pdFMRIData.columns)
    for i in range(len(lsFMRIAtlases)):
        lsFMRIAtlases[i] = lsFMRIAtlases[i][5:]

    # Fetch the fMRI connectivity data and place in a dictionary:
    # each dictionary key is the atlas name

    dFMRIConnectivityData = IMPAC_Autism_Run_multiple_atlases_v2.fFetchFMRIData(pdFMRIData,
                                                                       'AnatAndFMRIQAConnectivityDictionary')

    # Now, we loop through all the connectivity data and normalize each atlas's conectivity
    # matrices
    for i in range(len(lsFMRIAtlases)):
        dFMRIConnectivityData[lsFMRIAtlases[i]] = sk.preprocessing.normalize(
            dFMRIConnectivityData[lsFMRIAtlases[i]])

    return dFMRIConnectivityData, lsFMRIAtlases

def pdConnectivityFromUpperTraingVect(aVect):

    iRow, iCol = aVect.shape
    iRoi = int((np.sqrt(1+8*iCol)-1)/2)
    lsRoi=[]

    for iRoi1 in range(iRoi):
        for iRoi2 in range(iRoi):
            if iRoi1<=iRoi2:
                lsRoi.append(f'ROI{iRoi1+1:03}-ROI{iRoi2+1:03}')

    pdFMRIVects = pd.DataFrame(index=pdSMRIData.index, columns=lsRoi)
    pdFMRIVects.iloc[:, :] = aVect[:, :]

    return pdFMRIVects

dFMRIArrayData, lsFMRIAtlases = fProcessFMRI()

dXData={}
for sKey in dFMRIArrayData.keys():
    #pdFMRI = pdConnectivityFromUpperTraingVect(dFMRIArrayData[sKey])
    dlsGCReduced[sKey].index = list(map(int, dlsGCReduced[sKey].index))
    dXData.update({sKey: pd.concat([pdConfounders.drop(['Age'], axis=1), pdSMRIData, dlsGCReduced[sKey]], axis=1)})

#aAllData=pd.concat([pdConfounders.drop(['Age'], axis=1), pdSMRIData, pdFMRI], axis=1)

def fSplit(aAutLabel, pdConfounders, pdSMRIData, dlsGCReduced):
    # Append all Connectivity data to the structural data for a good stratified split
    # aAllData = np.append(aConfounders, aSMRIData, axis=1)
    # for i in range(len(lsFMRIAtlases)):
    #     aAllData = np.concatenate((aAllData, dFMRIConnectivityData[lsFMRIAtlases[i]]), axis=1)

    aAllData=pd.concat([pdConfounders.drop(['Age'], axis=1),
                        pdSMRIData,
                        dlsGCReduced['basc064'],
                        dlsGCReduced['basc122'],
                        dlsGCReduced['basc197'],
                        dlsGCReduced['craddock_scorr_mean'],
                        dlsGCReduced['harvard_oxford_cort_prob_2mm'],
                        dlsGCReduced['msdl'],
                        dlsGCReduced['power_2011']
                        ], axis=1)

    cTestTrainSplit = sk.model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    iterator = 0
    aTrainingIndex = np.zeros((10, int(len(aAutLabel) * 0.8)))
    aTestIndex = np.zeros((10, int(len(aAutLabel) * 0.2)))
    for train, test in cTestTrainSplit.split(aAllData, aAutLabel):
        aTrainingIndex[iterator, :] = train
        aTestIndex[iterator, :] = test
        break

    aTrainingIndex=aTrainingIndex[0,:]
    aTestIndex = aTestIndex[0, :]

    return aTrainingIndex, aTestIndex

def fOrganizeToDict(pdConfounders, pdSMRIData, dConnData, aAutLabel, lsFMRIAtlases, aTrainingIndex, aTestIndex,
                    sType='LSGC'):

    # Create a Dictionary to hold all the Test and Training data
    dXTrain = {}
    dXTest = {}

    #standard scale the sMRI data
    ss=sk.preprocessing.StandardScaler()
    # train
    pdSMRIDataTrain=pd.DataFrame(ss.fit_transform(pdSMRIData.iloc[aTrainingIndex]))
    pdSMRIDataTrain=pdSMRIDataTrain.set_index(pdSMRIData.iloc[aTrainingIndex].index)
    pdSMRIDataTrain.columns=pdSMRIData.columns
    # test
    pdSMRIDataTest=pd.DataFrame(ss.transform(pdSMRIData.iloc[aTestIndex]))
    pdSMRIDataTest=pdSMRIDataTest.set_index(pdSMRIData.iloc[aTestIndex].index)
    pdSMRIDataTest.columns=pdSMRIData.columns

    # standard scale the connectivity data
    dConnectivityTrain={}
    dConnectivityTest={}
    for sAtlas in lsAtlases:
        ss = sk.preprocessing.StandardScaler()
        if sType=='LSGC':
            #train
            dConnectivityTrain[sAtlas]=pd.DataFrame((ss.fit_transform(dConnData[sAtlas].iloc[
                                                                         aTrainingIndex].values))).fillna(0)
            dConnectivityTrain[sAtlas]=dConnectivityTrain[sAtlas].set_index(dConnData[sAtlas].iloc[aTrainingIndex].index)
            dConnectivityTrain[sAtlas].columns = dConnData[sAtlas].columns
            #test
            dConnectivityTest[sAtlas]=pd.DataFrame((ss.transform(dConnData[sAtlas].iloc[aTestIndex].values))).fillna(0)
            dConnectivityTest[sAtlas]=dConnectivityTest[sAtlas].set_index(dConnData[sAtlas].iloc[aTestIndex].index)
            dConnectivityTest[sAtlas].columns = dConnData[sAtlas].columns
        else:
            #train
            dConnectivityTrain[sAtlas]=pd.DataFrame((ss.fit_transform(
                dConnData[sAtlas].loc[pdConfounders.iloc[aTrainingIndex].index].values
            ))).fillna(0)
            dConnectivityTrain[sAtlas].index=pdConfounders.iloc[aTrainingIndex].index
            dConnectivityTrain[sAtlas].columns = dConnData[sAtlas].columns
            #test
            dConnectivityTest[sAtlas]=pd.DataFrame((ss.fit_transform(
                dConnData[sAtlas].loc[pdConfounders.iloc[aTestIndex].index].values
            ))).fillna(0)
            dConnectivityTest[sAtlas].index=pdConfounders.iloc[aTestIndex].index
            dConnectivityTest[sAtlas].columns = dConnData[sAtlas].columns

    # organize the anatomical data into test and training data
    dXTrain.update({'anatomy': pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTrainingIndex],
                               pdSMRIDataTrain,
                               ], axis=1).values})
    dXTest.update({'anatomy': pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTestIndex],
                              pdSMRIDataTest,
                              ], axis=1).values})

    pdAutLabel = pd.DataFrame(aAutLabel)

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

    for sAtlas in lsAtlases:
        dXTrain['connectivity'].update(
            {sAtlas: pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTrainingIndex],
                                dConnectivityTrain[sAtlas]], axis=1).values})
        dXTest['connectivity'].update(
            {sAtlas: pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTestIndex],
                                dConnectivityTest[sAtlas]], axis=1).values})
        dXTrain['combined'].update(
            {sAtlas: pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTrainingIndex],
                                pdSMRIDataTrain, dConnectivityTrain[sAtlas]], axis=1).values})
        dXTest['combined'].update(
            {sAtlas: pd.concat([pdConfounders.drop(['Age'], axis=1).iloc[aTestIndex],
                                pdSMRIDataTest,dConnectivityTest[sAtlas]], axis=1).values})

    aYTrain=aAutLabel[aTrainingIndex.astype('int')]
    aYTest=aAutLabel[aTestIndex.astype('int')]

    return dXTrain, dXTest, aYTrain, aYTest

def fRenameConn(dConn):
    for sConn in dConn.keys():
        # rename the columns by Roi-Roi connections
        for sAtlas in dConn[sConn].keys():
            iLen = int((-1 + (1+8*dConn[sConn][sAtlas].shape[1])**(1/2))/2)
            lsNames=list()
            for iRoi1 in range(iLen):
                for iRoi2 in range(iLen):
                    if not iRoi1 < iRoi2:
                        lsNames.append(f'ROI{iRoi1+1:03}-ROI{iRoi2+1:03}')

            dConn[sConn][sAtlas].columns=lsNames

        #return Connectivity measures in dictionary
    return dConn

if '__main__'==__name__:

    #Set which things to run
    bLSGC=False

    # Get indices of Training and test data
    aTrainingIndex, aTestIndex = fSplit(aAutLabel, pdConfounders, pdSMRIData, dlsGCReduced)
    #
    # #Load data
    # sRoot = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject'
    # dDataByConnType=pkl.load(open(os.path.join(sRoot, 'AllNewConnectivities.p'), 'rb'))
    #
    # #Rename features
    # dDataByConnType=fRenameConn(dDataByConnType)
    #
    # #Add lsGC to processing
    # dDataByConnType.update({'LSGC': dlsGCReduced})
    #
    # # Sort Train/Test for each type
    # for sKey in dDataByConnType.keys():
    #     # cut out train/test data
    #     dXTrain, dXTest, aYTrain, aYTest = fOrganizeToDict(pdConfounders, pdSMRIData, dDataByConnType[sKey],
    #                                                        aAutLabel, lsAtlases, aTrainingIndex, aTestIndex, sType=sKey)
    #     # fill in Nan's
    #     for sAtlas in dXTrain['connectivity']:
    #         sFlag='connectivity'
    #         print(f'{sKey},{sAtlas} number of NAN: {np.sum(np.isnan(dXTrain[sFlag][sAtlas]))}')
    #         print(f'{sKey},{sAtlas} number of NAN: {np.sum(np.isnan(dXTest[sFlag][sAtlas]))}')
    #         dXTrain[sFlag][sAtlas]=np.nan_to_num(dXTrain[sFlag][sAtlas])
    #         dXTest[sFlag][sAtlas]=np.nan_to_num(dXTest[sFlag][sAtlas])
    #
    #     # dump into directory
    #     # Dump data into directory
    #     pkl.dump([dXTrain, dXTest, aYTrain, aYTest],
    #              open(f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/{sKey}AllData.p',
    #                   'wb'))
    #
    #




