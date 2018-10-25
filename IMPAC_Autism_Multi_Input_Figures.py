"""This code takes the metrics from the Machine Learning Algorithms used
on the Paris IMPAC Autism study data (as detailed in IMPAC_Autism_Master)
and plots the performance as measured by a performance metric for multiple
input data types

The different input types are:
-Anatomical Volumetric Data
-Connectivity Data using the following atlases:
    -Basc atlas - 64 parcelations
                - 122 parcellations
                - 197 parcellations
    -Craddock Score atlas
    -Harvard-Oxford atlas
    -MSDL atlas
    -Power atlas
-Combined anatomical and probability data for the above atlases


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
September 2018

"""

import os
import pickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import matplotlib.patches as matplotpatches
import matplotlib.cm as cm
from textwrap import wrap
from matplotlib import rcParams
import pandas as pd

# Set up project directory locations
sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
sProjectIdentification = "AutismProject/Results"
sTrialIdentification = '3_fold_cv_50_random_hyperparameter_initializations_random_search0924'
sImagesPath = os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification, "SummaryImages")
sSummaryFileName = '10xCV_50foldRandSearch_MachineLearningResults.p'
if not os.path.exists(sImagesPath):
    os.makedirs(sImagesPath)

#set up figure global properties
rcParams.update({'figure.autolayout': True})

# The Tags used to save data for a specific ML algorithm,
# used to fetch the correct data
lsMLTags = [
    'NaiveBayes',
    'RandFor',
    'ExRanTrees',
    'AdaBoost',
    'XGBoost',
    'GaussSVM',
    'LinSVM',
    'LinLasso',
    'LinRidge'

]

# The names of the ML algorithms used, to label figures later
lsMLAlgorithms = [
    'Naive Bayes',
    'Random Forest',
    'Extremely Random Trees',
    'Ada Boost Gradient Boosting',
    'xgBoost Gradient Boosting',
    'SVM with Gaussian Kernel',
    'SVM with Linear Kernel',
    'Linear Lasso Regression',
    'Linear Ridge Regression'
]

# The Tags used to save data for a specific type of input data used
# used to fetch the correct data
lsInputTags = [
    'anatomical_only', 'connectivity_basc064', 'connectivity_basc122', 'connectivity_basc197',
    'connectivity_craddock_scorr_mean', 'connectivity_harvard_oxford_cort_prob_2mm', 'connectivity_msdl',
    'connectivity_power_2011', 'combined_basc064', 'combined_basc122', 'combined_basc197',
    'combined_craddock_scorr_mean', 'combined_harvard_oxford_cort_prob_2mm', 'combined_msdl',
    'combined_power_2011'
]

# The names of the types of input data used, to label figures later
lsInputNames = [
    'Anatomical Volumetric Data Alone',
    'Connectivity using the BASC Atlas with 64 Parcellations',
    'Connectivity using the BASC Atlas with 122 Parcellations',
    'Connectivity using the BASC Atlas with 197 Parcellations',
    'Connectivity using the Craddock Atlas with X Parcellations',
    'Connectivity using the Harvard-Oxford Atlas with X Parcellations',
    'Connectivity using the MSDL Atlas with X Parcellations',
    'Connectivity using the Power Atlas with X Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 64 Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 122 Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the BASC Atlas with 197 Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the Craddock Atlas with X Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the Harvard-Oxford Atlas with X Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the MSDL Atlas with X Parcellations',
    'Combined Anatomical Volumetric Data and Connectivity with the Power Atlas with X Parcellations',
]

# The Tags used to save data for a specific performance metric,
# used to fetch the correct data
lsMetricTags = ['accuracy', 'Precision/recall area under curve',
               'F1 score', 'ROC curve area under curve']

# The names of the Metrics used, to label figures later
lsMetrics = ['Accuracy', 'Area Under Precision-Recall Curve',
             'F1 Score', 'Area Under ROC Curve']

# Initialize dictionary to hold data for each performance metric
dPerformanceByMetric = {}

# Initialize several empty dataframes within a dictionary to hold the results about to be loaded
#   -one for each performance metric used
for metrics in lsMetrics:
    dPerformanceByMetric.update({metrics: pd.DataFrame(columns=lsMLAlgorithms, index=lsInputNames)})


# Now, we loop through the Trial Directory and fill in results if they exist.
# If the results do not exist, the performance metric is set to 0
for iInputNumber in range(len(lsInputTags)):
    sInputTag = lsInputTags[iInputNumber]
    sInputName = lsInputNames[iInputNumber]
    if os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification,
                                   sInputTag, sSummaryFileName)):
        pdSummary = pd.read_pickle(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                                sTrialIdentification, sInputTag, sSummaryFileName))
        for iMetricNumber in range(len(lsMetricTags)):
            sMetricTag = lsMetricTags[iMetricNumber]
            sMetricName = lsMetrics[iMetricNumber]
            for iMLModel in range(len(lsMLAlgorithms)):
                sMLTag = lsMLTags[iMLModel]
                sMLName = lsMLAlgorithms[iMLModel]
                dPerformanceByMetric[sMetricName].loc[sInputName, sMLName] = pdSummary.loc[sMetricTag, sMLTag]

    # If the results exist for just a single ML algorithm, but not the full summary,
    # just load the results that have completed
    else:
        for iIndividualResults in range(len(lsMLTags)):
            sIndividualResultsFile = lsMLTags[iIndividualResults] +'Stats.p'
            sIndividualResultsName = lsMLAlgorithms[iIndividualResults]
            sIndividualResultsTag = lsMLTags[iIndividualResults]
            if os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                           sTrialIdentification, sInputTag, sIndividualResultsFile)):
                pdSummary = pd.read_pickle(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                                        sTrialIdentification, sInputTag, sIndividualResultsFile))
                for iMetricNumber in range(len(lsMetricTags)):
                    sMetricTag = lsMetricTags[iMetricNumber]
                    sMetricName = lsMetrics[iMetricNumber]
                    dPerformanceByMetric[sMetricName].loc[sInputName, sIndividualResultsName] = pdSummary.loc[sMetricTag]

for keys in dPerformanceByMetric.keys():
    dPerformanceByMetric[keys]=dPerformanceByMetric[keys].fillna(int(0))

pickle.dump(dPerformanceByMetric, open(os.path.join(sProjectRootDirectory, sProjectIdentification, 'dMLSummaryData.p'), 'wb'))

# Now that we have saved everything in a single dictionary, we plot a nested bar graph of all
# the results
def fBarPlotTestPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms, sFigureTag='All', iTopNumber=None):
    for sMetricName in lsMetrics:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        #ax = dPerformanceByMetric[sMetricName].plot.bar()
        aMetricPerformance = dPerformanceByMetric[sMetricName].values
        iHeight = aMetricPerformance.shape[0]
        iWidth = aMetricPerformance.shape[1]

        ax = dPerformanceByMetric[sMetricName].plot.bar()
        ax.set_ylabel(sMetricName)
        pdYValues = dPerformanceByMetric[sMetricName]
        pdYValues = pdYValues.fillna(int(0))
        aYValues = pdYValues.values
        ax.set_ylim(np.min(aYValues[np.nonzero(aYValues)])-0.01, aYValues.max() + 0.01)
        ax.set_title(('Performance of ML Algorithms on Different Input Data by ' + sMetricName), loc='left', size='small')
        ax.set_xticklabels(lsInputTags)

        plt.xticks(size='xx-small', rotation=90)



        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, prop={'size': 6})
        plt.subplots_adjust(left=0.5)
        plt.tight_layout()

        savefig(os.path.join(sImagesPath, sMetricName + sFigureTag + '.png'))
        savefig(os.path.join(sImagesPath, sMetricName + sFigureTag + '.pdf'))


def fPlotTopIPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms, sFigureTag='All', iTopNumber=3, sPlotType='bar'):
    for sMetricName in lsMetrics:

        import matplotlib.pyplot as plt

        # select only the current metric results
        pdTop = dPerformanceByMetric[sMetricName]

        # select the top iTopNumber largest values per input type, set the rest to NAN
        pdTop = pdTop.stack().groupby(level=0).nlargest(iTopNumber).unstack().reset_index(level=1, drop=True).reindex(columns=pdTop.columns)

        # replace NA with 0
        pdTop = pdTop.fillna(int(0))

        # plot the data
        if sPlotType=='bar':
            ax = pdTop.plot.bar()
        elif sPlotType=='scatter':
            ax = pdTop.plot.scatter()
        ax.set_ylabel(sMetricName)
        aYValues = pdTop.values
        ax.set_ylim(np.min(aYValues[np.nonzero(aYValues)])-0.01, pdTop.values.max() + 0.01)
        ax.set_title(('Performance of Top 3 ML Algorithms on Different Input Data by ' + sMetricName), loc='left', size='small')
        ax.set_xticklabels(lsInputTags)
        plt.xticks(size='xx-small', rotation=90)

        # attach a legend and
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, prop={'size': 6})
        plt.subplots_adjust(left=0.5)
        plt.tight_layout()

        savefig(os.path.join(sImagesPath, sMetricName + 'Top' + str(iTopNumber) + '.png'))
        savefig(os.path.join(sImagesPath, sMetricName + 'Top' + str(iTopNumber) + '.png'))
        plt.close()

###########################
def fScatterPlotTestPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms, bConnectLines=False, sFigureTag='All', iTopNumber=None):
    for sMetricName in lsMetrics:

        # pull up the performance data for a specific performance metric
        pdPerformance = dPerformanceByMetric[sMetricName]

        fig, ax = plt.subplots()

        aPerformance = pdPerformance.values

        # Turn the performance data into an array
        aYValues = aPerformance


        # Numerically associate the input data type with the x position
        aXValues = np.array(range(aYValues.shape[0]*aYValues.shape[1]))
        aXValues = np.floor(aXValues/aYValues.shape[1])
        aXValues = aXValues.reshape((aYValues.shape[0], aYValues.shape[1]))

        if not len(aYValues[aYValues==0])==0:
            aXValues[aYValues==0] = 'nan'
            aYValues[aYValues==0] = 'nan'

        lsColors = cm.rainbow(np.linspace(0, 1, len(pdPerformance.columns)))

        # Make a scatterplot with each color corresponding to a ML algorithm
        for iMLAlgorithm in range(len(pdPerformance.columns)):
            aColor=lsColors[iMLAlgorithm]
            for iInputNumber in range(len(pdPerformance.index)):
                if bConnectLines==True:
                    ax.plot(aXValues[:, iMLAlgorithm],
                        aYValues[:, iMLAlgorithm],
                        color=aColor, linestyle='-', alpha=0.03)
                ax.scatter(aXValues[:, iMLAlgorithm],
                           aYValues[:, iMLAlgorithm],
                           color=aColor, alpha=0.5)



        ax.set_ylabel(sMetricName)
        ax.set_title(('Performance of ML Algorithms on Different Input Data by ' + sMetricName), loc='left', size='small')

        plt.xticks(range(len(lsInputTags)), lsInputTags, size='xx-small', rotation=90)

        #ax.set_ylim(np.min(aYValues[np.nonzero(aYValues)]) - 0.01,
                    #dPerformanceByMetric[sMetricName].values.max() + 0.01)

        lsLegendColors=list()

        for iMLAlgorithm in range(len(lsMLAlgorithms)):
            sMLAlgorithm = lsMLAlgorithms[iMLAlgorithm]
            lsLegendColors.append(matplotpatches.Patch(color=lsColors[iMLAlgorithm], label=sMLAlgorithm))

        plt.legend(handles=lsLegendColors, bbox_to_anchor=(1.05, 1), loc=2, prop={'size':6})
        plt.tight_layout()

        savefig(os.path.join(sImagesPath, sMetricName + ' ' + sFigureTag + '.png'))
        savefig(os.path.join(sImagesPath, sMetricName + ' ' + sFigureTag + '.pdf'))
        plt.close()


fScatterPlotTestPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms)
fScatterPlotTestPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms,
                            bConnectLines=True, sFigureTag='All Connected')
fBarPlotTestPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms, sFigureTag='Bar')
#plot top 3 not currently working - need to debug
#fPlotTopIPerformance(dPerformanceByMetric, lsMetrics, lsInputTags, lsMLAlgorithms)

#############################Validation Data Plots##############################

# Create a dictionary to hold the validation Data
dValidationByInput={}

# Now we load the validation data
# first, we loop over the possible inputs (anatomical, connectivity with atlas 1, etc)
for iInputNumber in range(len(lsInputTags)):
    sInputTag = lsInputTags[iInputNumber]
    sInputName = lsInputNames[iInputNumber]

    #then we test if there is already a validation data summary file
    if not os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification,
                                           sInputTag, 'BestModelsValidationSummary')+'.p'):

        # then we create a dataframe to hold the validation data
        pdValidationScores = pd.DataFrame(index=lsMLAlgorithms[1:], columns=['Mean Validation Score of Best Model',
                                                                         'Standard Deviation of Validation Scores of Best Model'])

        # then we loop over the different ML algorithms used except for the naive bayes (ML model 0)
        # becuase the Naive bayes model doesn't have validation data
        for iMLModel in range(len(lsMLAlgorithms[1:])):
            iMLModel = iMLModel + int(1)
            sMLTag = lsMLTags[iMLModel]
            sMLName = lsMLAlgorithms[iMLModel]

            # if the model has been run, we fetch the mean validation performance of the best run of the random
            # search as well as the standard deviation,
            # if the model has not been run, we set the values to 0
            if os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, sTrialIdentification,
                                           sInputTag, sMLTag)+'.p'):
                skcModel=pickle.load(open(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                                       sTrialIdentification, sInputTag, sMLTag) + '.p', 'rb'))
                iBestModelIndex = np.where(skcModel.cv_results_['rank_test_score'] == 1)[0][0]
                mean = skcModel.cv_results_['mean_test_score'][iBestModelIndex]
                std = skcModel.cv_results_['std_test_score'][iBestModelIndex]
                pdValidationScores.loc[sMLName, 'Mean Validation Score of Best Model'] = mean
                pdValidationScores.loc[sMLName, 'Standard Deviation of Validation Scores of Best Model'] = std
            else:
                pdValidationScores.loc[sMLName, 'Mean Validation Score of Best Model'] = 0
                pdValidationScores.loc[sMLName, 'Standard Deviation of Validation Scores of Best Model'] = 0

        # finally, we save the validation score table
        pickle.dump(pdValidationScores, open(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                                          sTrialIdentification, sInputTag,
                                                          'BestModelsValidationSummary')+'.p', 'wb'))
    # if the validation data summary file exists, we simply load it
    else:
        pdValidationScores = pickle.load(open(os.path.join(sProjectRootDirectory, sProjectIdentification,
                                              sTrialIdentification, sInputTag,
                                              'BestModelsValidationSummary')+'.p', 'rb'))

    # update the master dictionary contatining all input types
    dValidationByInput.update({sInputTag: pdValidationScores})

pdValidationByInput = pd.concat(dValidationByInput)

pickle.dump(pdValidationByInput, open(os.path.join(sProjectRootDirectory, sProjectIdentification, 'pdMLSummaryValidationData.p'), 'wb'))

# Now we redefine a plotting function for the slightly different dataset
# (now data is a pandas dataframe with mean validation error and std
def fValPlot(pdValidationByInput, sImagesPath):
    fig, ax = plt.subplots()

    for iInput in range(len(lsInputTags)):
        sInput = lsInputTags[iInput]

        # Retreive the Mean and std for a given input
        lsMeans = list(pdValidationByInput.loc[sInput]['Mean Validation Score of Best Model'].values)
        lsStd = list(pdValidationByInput.loc[sInput]['Standard Deviation of Validation Scores of Best Model'].values)
        lsNames = list(pdValidationByInput.loc[sInput]['Mean Validation Score of Best Model'].index)
        lsColors = cm.rainbow(np.linspace(0,1,len(lsMeans)))

        # Sort by mean validation area under ROC curve
        # aOrder=np.argsort(lsMeans)
        # lsMeans[:] = [lsMeans[x] for x in aOrder]
        # lsStd[:] = [lsStd[x] for x in aOrder]
        # lsNames[:] = [lsNames[x] for x in aOrder]
        # lsColors[:] = [lsColors[x] for x in aOrder]

        for iPoint in range(len(lsMeans)):
            aColor=lsColors[iPoint]
            ax.scatter(iPoint, lsMeans[iPoint], edgecolors='none', color=aColor)
            ax.errorbar(iPoint, lsMeans[iPoint], lsStd[iPoint], ls='none', color=aColor)

    plt.xticks(range(len(lsMeans)), lsNames, size='small', rotation=45)

    ax.set_ylim(0.5, 1)
    ax.set_xlabel('Machine Learning Model')
    ax.set_ylabel('Area Under ROC Curve')
    ax.set_title('\n'.join(wrap('Model Performance on Validation Data in 10x Cross-Validation for All Input Types')))
    fig.savefig(os.path.join(sImagesPath, 'All Input Validation Performance.png'))
    fig.savefig(os.path.join(sImagesPath, 'All Input Validation Performance.pdf'))

fValPlot(pdValidationByInput, sImagesPath)

def fHeatMap(dPerformance, sMetric, lsXlabels, lsYlabels, sImagesPath):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(dPerformance[sMetric].values)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Performance by ' + sMetric, rotation=-90, va='bottom')

    ax.set_xticklabels(lsXlabels, rotation=45, ha='right')
    ax.set_yticklabels(lsYlabels, rotation=0)
    plt.xticks(np.arange(0, len(lsXlabels)))
    plt.yticks(np.arange(0, len(lsYlabels)))
    fig.savefig(os.path.join(sImagesPath, sMetric) +' Performance Heatmap.png')
    fig.savefig(os.path.join(sImagesPath, sMetric) + ' Performance Heatmap.pdf')

for sMetricName in lsMetrics:
    fHeatMap(dPerformanceByMetric, sMetricName, lsMLTags, lsInputTags, sImagesPath)


