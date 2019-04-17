"""This code takes the metrics from the Machine Learning Algorithms used
on the Paris IMPAC Autism study data (as detailed in IMPAC_Autism_Master)
and plots the metrics used to evaluate which models best predicted
the presence or absence of autism in the patients

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
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
from textwrap import wrap
from matplotlib import rcParams
import pandas as pd
rcParams.update({'figure.autolayout': True})


# The path where the data is stored
sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainedModels'
# The path of the particular run being analysed
# sTargetDirectory = "anatomical_only_machine_learning_10x_cross_val_50_fold_Rand_Search_ROC_AUC_Metric"
sTargetDirectory = "msdl_fMRI_connectivity_machine_learning_10x_cross_val_50_fold_Rand_Search_ROC_AUC_Metric"

# The path For images to be saved to
sImagesPath = os.path.join(sDataPath, sTargetDirectory, 'Images')
# The name of the file holding the performance metrics
sFileName = '10xCV_50foldRandSearch_MachineLearningResults'

# Loading the pickled file
pdData = pd.read_pickle(os.path.join(sDataPath, sTargetDirectory, sFileName + '.p'))

# The Data is a Pandas Dataframe of the following form:
#                                   'Machine Learning Model#1'  'Model #2' , etc.
# accuracy                         |    float64 value         |  float64 value
# Precision/recall area under curve|    float64 value         |      ...
# F1 Score                         |        ...               |      ...
# ROC curve area under curve       |        ...               |      ...

# The names of all the Models used, corresponding to 'Machine Learning Model #1, #2, etc.
# as above. This list is to be later used in titles of figures
lsMLAlgorithms = ['Naive Bayes', 'Linear Ridge Regression', 'Linear Lasso Regression',
                  'SVM with Linear Kernel', 'SVM with Gaussian Kernel', 'Random Forest',
                  'Extremely Random Trees', 'Ada Boost Gradient Boosting', 'xgBoost Gradient Boosting']


def fPlot(iMetric, pdData, lsAlgorithm, sTitle, sImagesPath):

    # The names of the metrics as stored in the Dataframe, to be used to index through
    # the data
    lsMetricIDs = ['accuracy', 'Precision/recall area under curve',
                   'F1 score', 'ROC curve area under curve']

    # The names of the Metrics used, to Name figures later
    lsMetrics = ['Accuracy', 'Area Under Precision-Recall Curve',
                 'F1 Score', 'Area Under ROC Curve']

    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, ax = plt.subplots()

    # sort the row by values for that row
    lsX = pdData.iloc[iMetric, :].sort_values(axis=0)

    plt.xticks(range(len(lsX)), lsX.index, size='small', rotation=45)

    for iAlg in range(len(lsAlgorithm)):
        x = lsX.index[iAlg]
        y = lsX[iAlg]

        # Make the names of the Algorithms the ticks on the x axis


        ax.scatter(x, y)

    ax.set_xlabel('Model')
    ax.set_ylabel(lsMetrics[iMetric])
    ax.set_title('\n'.join(wrap(sTitle + lsMetrics[iMetric])))
    fig.savefig(os.path.join(sImagesPath, lsMetrics[iMetric] + '.png'))
    del ax
    del fig
    del plt

# Now we generate and save the plots
sTitle ='Performance of Machine Learning Algorithms by '
fPlot(0, pdData, lsMLAlgorithms, sTitle, sImagesPath)
fPlot(1, pdData, lsMLAlgorithms, sTitle, sImagesPath)
fPlot(2, pdData, lsMLAlgorithms, sTitle, sImagesPath)
fPlot(3, pdData, lsMLAlgorithms, sTitle, sImagesPath)

# Here we load up the individual models we trained in order to extract the validation scores

# Note: the bayesian model doesn't have validation data as there was no cross validation
# skcBayesModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/NaiveBayes.p", 'rb'))
skcLinRidgeModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/LinRidge.p", 'rb'))
skcLinLassoModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/LinLasso.p", 'rb'))
skcLinSVMModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/LinSVM.p", 'rb'))
skcGaussSVMModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/GaussSVC.p", 'rb'))
skcRandForModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/RandFor.p", 'rb'))
skcExRanTreeModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/ExRanTrees.p", 'rb'))
skcAdaBModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/AdaBoost.p", 'rb'))
skcXGBoostModel = pickle.load(open(os.path.join(sDataPath, sTargetDirectory) + "/XGBoost.p", 'rb'))

lsModels = [skcLinRidgeModel, skcLinLassoModel, skcLinSVMModel, skcGaussSVMModel,
            skcRandForModel, skcExRanTreeModel, skcAdaBModel, skcXGBoostModel]
lsValidationScores=['skcLinRidgeModel', 'skcLinLassoModel', 'skcLinSVMModel', 'skcGaussSVMModel',
                   'skcRandForModel', 'skcExRanTreeModel', 'skcAdaBModel', 'skcXGBoostModel']
pdValidationScores = pd.DataFrame(columns=lsValidationScores, index=('Mean Validation Score', 'Validation Score Std'))

# We cycle through the models and save the validation scores as an array in a dictionary
for ModelIndex in range(len(lsValidationScores)):

    iBestModelIndex = np.where(lsModels[ModelIndex].cv_results_['rank_test_score']==1)[0][0]
    mean = lsModels[ModelIndex].cv_results_['mean_test_score'][iBestModelIndex]
    std = lsModels[ModelIndex].cv_results_['std_test_score'][iBestModelIndex]


    pdValidationScores[lsValidationScores[ModelIndex]]['Mean Validation Score'] = mean
    pdValidationScores[lsValidationScores[ModelIndex]]['Validation Score Std'] = std


# Sort the Pandas frame columns by mean validation error
pdValidationScores = pdValidationScores.sort_values('Mean Validation Score', axis=1)

# Now we redefine a plotting function for the slightly different dataset
# (now data is a pandas dataframe with mean validation error and std
def fValPlot(pdValidationScores):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig, ax = plt.subplots()

    # sort the row by values for that row
    lsX = list(pdValidationScores.columns.values)
    for iterator in range(len(lsX)):
        lsX[iterator] = lsX[iterator][3:]

    plt.xticks(range(len(lsX)), tuple(lsX), size='small', rotation=45)

    for iModel in range(len(lsX)):
        x = iModel
        y = pdValidationScores['skc'+lsX[iModel]][0]
        error = pdValidationScores['skc'+lsX[iModel]][1]
        # Make the names of the ML Algorithms the ticks on the x axis





        ax.scatter(x, y)
        ax.errorbar(x, y, error)

    ax.set_xlabel('Model')
    ax.set_ylabel('Area Under ROC Curve')
    ax.set_title('\n'.join(wrap('Model Performance on Validation Data in 10x Cross-Validation')))
    fig.savefig(os.path.join(sImagesPath, 'ValidationErrors.png'))
    del ax
    del fig
    del plt

fValPlot(pdValidationScores)