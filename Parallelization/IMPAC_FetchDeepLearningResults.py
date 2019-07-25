""" This file fetches and saves the deep learning model results

The validation and test data are both fetched, saved, and then
appended to the standard machine learning results to be used
in a comparative analysis

"""
import pandas as pd
import numpy as np
import keras
import os
import pickle
import sklearn.metrics as skm

sProjectDirectory = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
sDeepLearningLocation = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun'
sMLModelLocation = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Results/3_fold_cv_50_random_hyperparameter_initializations_random_search0924'
sSaveLocation = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/MLDeepSummaryData'

# The names of the Deep Learning algorithms used, to label data frame
# lsDLAlgorithms = [
#     'Dense Network',
#     'LSTM Network'
# ]

lsDLAlgorithms = [
    'Dense Network'
]

#
#lsDLTags = ['Dense', 'LSTM']
lsDLTags = ['Dense']

# The names of the types of input data used, to label data frame
# lsInputNames = [
#     'Anatomical Volumetric Data Alone',
#     'Connectivity using the BASC Atlas with 64 Parcellations',
#     'Connectivity using the BASC Atlas with 122 Parcellations',
#     'Connectivity using the BASC Atlas with 197 Parcellations',
#     'Connectivity using the Craddock Atlas with X Parcellations',
#     'Connectivity using the Harvard-Oxford Atlas with X Parcellations',
#     'Connectivity using the MSDL Atlas with X Parcellations',
#     'Connectivity using the Power Atlas with X Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 64 Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 122 Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 197 Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the Craddock Atlas with X Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the Harvard-Oxford Atlas with X Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the MSDL Atlas with X Parcellations',
#     'Combined Anatomical Volumetric Data and Connectivity with the Power Atlas with X Parcellations',
# ]
lsInputNames = [
    'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 122 Parcellations',
]

# The tags of the types of input data used, to fetch the data
# lsInputTags = [
#     'anatomy', 'connectivitybasc064', 'connectivitybasc122', 'connectivitybasc197',
#     'connectivitycraddock_scorr_mean', 'connectivityharvard_oxford_cort_prob_2mm', 'connectivitymsdl',
#     'connectivitypower_2011', 'combinedbasc064', 'combinedbasc122', 'combinedbasc197',
#     'combinedcraddock_scorr_mean', 'combinedharvard_oxford_cort_prob_2mm', 'combinedmsdl',
#     'combinedpower_2011'
# ]

lsInputTags = [
    'combinedbasc064', 'combinedbasc122', 'combinedbasc197',
    'combinedcraddock_scorr_mean', 'combinedharvard_oxford_cort_prob_2mm', 'combinedmsdl',
    'combinedpower_2011'
]


# The names of the Metrics used, to label dataFrame
#lsMetrics = ['Accuracy', 'Area Under Precision-Recall Curve',
#             'F1 Score', 'Area Under ROC Curve']

lsMetrics = ['Area Under ROC Curve']

# Initialize dictionary to hold data for each DL Framework
dPerformanceByDLAlg = {}

# Initialize a dictionary containing the deep learning algorithm that itself
# contains a dictionary containing the different metrics used and a pandas
# dataframe. The dataframe has columns for each input (anatomical alone,
# atlas1, atlas2, etc) and rows for each architecture attempted
# for sDLAlgorithm in lsDLAlgorithms:
#     dPerformanceByDLAlg.update({sDLAlgorithm: {lsMetrics[0]: pd.DataFrame(columns=lsInputNames, index=range(50)),
#                                                lsMetrics[1]: pd.DataFrame(columns=lsInputNames, index=range(50)),
#                                                lsMetrics[2]: pd.DataFrame(columns=lsInputNames, index=range(50)),
#                                                lsMetrics[3]: pd.DataFrame(columns=lsInputNames, index=range(50))
#                                                }})
for sDLAlgorithm in lsDLAlgorithms:
    dPerformanceByDLAlg.update({sDLAlgorithm: {lsMetrics[0]: pd.DataFrame(columns=lsInputNames, index=range(50)),
                                               }})

# This function will calculate a performance metric for a given prediction
# By the DL algorithm
def fCalculateMetric(iMetric, aPredicted):
    """
    :param iMetric: must be 0(accuracy), 1(area under Precision-Recall), 2(F1 score), or 3(ROC AUC)
    :param aPredicted: array of the predicted categorization (autism vs not)
    :return: float value of teh metric desited
    """
    dXData, dXTest, aYData, aYTest = pickle.load(open(sProjectDirectory + '/TrainTestData.p', 'rb'))

    if iMetric==0:
        flResults= skm.accuracy_score(aYTest, np.round(aPredicted).astype(int), normalize=False)
        return flResults

    elif iMetric==1:
        flResults= skm.f1_score(aYTest, np.round(aPredicted).astype(int))
        return flResults

    elif iMetric==2:
        aPrecisions, aRecalls, aThresholds = skm.precision_recall_curve(aYTest, aPredicted)
        flResults = skm.auc(aRecalls, aPrecisions)
        return flResults

    elif iMetric==3:
        flResults= skm.roc_auc_score(aYTest, aPredicted)
        return flResults

    else:
        raise Exception('Not a valid selection of metric, must be 0(accuracy), 1(area under Precision-Recall), 2(F1 score), or 3(ROC AUC)')


#First, we check if the file has already been saved
if not os.path.isfile(os.path.join(sDeepLearningLocation, 'Results', 'DeepLearningResults')+'.p'):
    # Now, we loop through the Trial Directory and fill in results if they exist.
    # If the results do not exist, the performance metric is set to 0
    for iDLModel, sDLModel in enumerate(lsDLAlgorithms):
        sDLTag = lsDLTags[iDLModel]
        for iMetric, sMetric in enumerate(lsMetrics):
            iMetric = 3
            for iInputNumber, sInputName in enumerate(lsInputNames):
                sInputTag = lsInputTags[iInputNumber]
                for iTrial in range(50):
                    sFile = os.path.join(sDeepLearningLocation,sDLTag, sDLTag) + f'_{iTrial:02}' + sInputTag + \
                                                                         'PredictedResults.p'
                    if os.path.isfile(sFile):
                        aPredicted = pickle.load(open(sFile, 'rb'))
                        flPerformance = fCalculateMetric(iMetric, aPredicted)
                        dPerformanceByDLAlg[sDLModel][sMetric].iloc[iTrial, iInputNumber] = flPerformance
                        print('model ' + sDLModel + ' ' + str(iTrial) + ' was successful for input ' + sInputName + ' - ' + sMetric + ' Calculated')
                    elif not os.path.isfile(sFile):
                        dPerformanceByDLAlg[sDLModel][sMetric].iloc[iTrial, iInputNumber] = np.nan
                        print('model ' + sDLModel + ' ' + str(iTrial) + ' failed - ' + sMetric + ' NOT determined')

    # Save the summary file
    pickle.dump(dPerformanceByDLAlg, open(os.path.join(sDeepLearningLocation, 'Results', 'DeepLearningResults')+'.p', 'wb'))

# If the summary file already exists, load it
elif os.path.isfile(os.path.join(sDeepLearningLocation, 'Results', 'DeepLearningResults')+'.p'):
    dPerformanceByDLAlg = pickle.load(open(os.path.join(sDeepLearningLocation, 'Results', 'DeepLearningResults')+'.p', 'rb'))












#
#
#
#
#
#
# pickle.dump(dPerformanceByMetric, open(os.path.join(sProjectRootDirectory, sProjectIdentification, 'dMLSummaryData.p'), 'wb'))
#
#
# #############################Validation Data##############################
#
# # Create a dictionary to hold the validation Data
# dValidationByInput={}
#
# # Now we load the validation data
# # first, we loop over the possible inputs (anatomical, connectivity with atlas 1, etc)
# for iInputNumber in range(len(lsInputTags)):
#     sInputTag = lsInputTags[iInputNumber]
#     sInputName = lsInputNames[iInputNumber]
#
#     #then we test if there is already a validation data summary file
#     if not os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification,
#                                            sInputTag, 'BestModelsValidationSummary')+'.p'):
#
#         # then we create a dataframe to hold the validation data
#         pdValidationScores = pd.DataFrame(index=lsMLAlgorithms[1:], columns=['Mean Validation Score of Best Model',
#                                                                          'Standard Deviation of Validation Scores of Best Model'])
#
#         # then we loop over the different ML algorithms used except for the naive bayes (ML model 0)
#         # becuase the Naive bayes model doesn't have validation data
#         for iMLModel in range(len(lsMLAlgorithms[1:])):
#             iMLModel = iMLModel + int(1)
#             sMLTag = lsMLTags[iMLModel]
#             sMLName = lsMLAlgorithms[iMLModel]
#
#             # if the model has been run, we fetch the mean validation performance of the best run of the random
#             # search as well as the standard deviation,
#             # if the model has not been run, we set the values to 0
#             if os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification,
#                                            sInputTag, sMLTag)+'.p'):
#                 skcModel=pickle.load(open(os.path.join(sProjectRootDirectory, sProjectIdentification,
#                                                        sTrialIdentification, sInputTag, sMLTag) + '.p', 'rb'))
#                 iBestModelIndex = np.where(skcModel.cv_results_['rank_test_score'] == 1)[0][0]
#                 mean = skcModel.cv_results_['mean_test_score'][iBestModelIndex]
#                 std = skcModel.cv_results_['std_test_score'][iBestModelIndex]
#                 pdValidationScores.loc[sMLName, 'Mean Validation Score of Best Model'] = mean
#                 pdValidationScores.loc[sMLName, 'Standard Deviation of Validation Scores of Best Model'] = std
#             else:
#                 pdValidationScores.loc[sMLName, 'Mean Validation Score of Best Model'] = 0
#                 pdValidationScores.loc[sMLName, 'Standard Deviation of Validation Scores of Best Model'] = 0
#
#         # finally, we save the validation score table
#         pickle.dump(pdValidationScores, open(os.path.join(sProjectRootDirectory, sProjectIdentification,
#                                                           sTrialIdentification, sInputTag,
#                                                           'BestModelsValidationSummary')+'.p', 'wb'))
#     # if the validation data summary file exists, we simply load it
#     else:
#         pdValidationScores = pickle.load(open(os.path.join(sProjectRootDirectory, sProjectIdentification,
#                                               sTrialIdentification, sInputTag,
#                                               'BestModelsValidationSummary')+'.p', 'rb'))
#
#     # update the master dictionary contatining all input types
#     dValidationByInput.update({sInputTag: pdValidationScores})
#
# pdValidationByInput = pd.concat(dValidationByInput)