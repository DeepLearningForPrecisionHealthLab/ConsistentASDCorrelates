# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../../../tmp/c5a686aa-754f-4a71-a3a0-bd8a3022a164'))
	print(os.getcwd())
except:
	pass
# %%
from IPython import get_ipython

# %% [markdown]
# Material for accuracy of models supplementary figure

# %%
import numpy as np
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle as pkl
from sklearn.metrics import roc_curve, roc_auc_score


# %%
# Load raw data
sData='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AllDataWithConfounds.p'
#Dictionary that containes the whole dataset (train and test) in pd dataframe
[dXData, aYData] = pkl.load(open(sData, 'rb'))

# Load predictions for all models
# sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles' \
#                 '/EnsembleModelSet.p'
# dModels = pkl.load(open(sModelPath, 'rb'))
sTestPredictionPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/Ensembles'                 '/EnsemblePredictionTestSet.p'
dTestPredictions = pkl.load(open(sTestPredictionPath, 'rb'))

# load test data as well
sTestLoc='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/aTestIndex.p'
aTestLoc=pkl.load(open(sTestLoc,'rb'))
aYTest=aYData[aTestLoc.astype('int'),:]


# %%
pdResults=pd.DataFrame(columns=['Atlas', 'MeanVal', 'Predicted'])
for sAtlas in ['basc064', 'basc122', 'basc197']:
    for i in range(50):
        flAvgCV=np.mean([pkl.load(open(f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun/Dense/Dense_{i:02}combined{sAtlas}TrainROCScoreCrossVal{iCV}.p', 'rb')) for iCV in range(1,4)])
        aTestPredictions = pkl.load(open(f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun/Dense/Dense_{i:02}combined{sAtlas}PredictedResults.p', 'rb'))
        pdResults=pdResults.append(pd.DataFrame.from_dict(
            {
                'Atlas': [sAtlas],
                'MeanVal': [flAvgCV],
                'Predicted': [aTestPredictions]
            }
        ))
        


# %%
aPredicted = pdResults[pdResults['Atlas']=='basc064'].sort_values(by='MeanVal', ascending=False).head(5)['Predicted'].iloc[0]
fpr, tpr, thresholds = roc_curve(aYTest, aPredicted)
flScore = roc_auc_score(aYTest, aPredicted)
Sn = tpr
Sp = 1-fpr
Sn[Sp>.6][-1], Sn[Sp>.7][-1], Sn[Sp>.8][-1]


# %%
from sklearn.metrics import roc_curve, roc_auc_score
pdChart = pd.DataFrame(columns=['Atlas', 'AUROC', 'Sn at 60 Sp', 'Sn at 70 Sp', 'Sn at 80 Sp'])
for sAtlas in ['basc064', 'basc122', 'basc197']:
    for i in range(5):
        aPredicted = pdResults[pdResults['Atlas']==sAtlas].sort_values(by='MeanVal', ascending=False).head(5)['Predicted'].iloc[i]
        fpr, tpr, thresholds = roc_curve(aYTest, aPredicted)
        flScore = roc_auc_score(aYTest, aPredicted)
        Sn = tpr
        Sp = 1-fpr
        pdChart=pdChart.append(pd.DataFrame.from_dict({
            'Atlas':[sAtlas],
            'AUROC':[flScore],
            'Sn at 60 Sp':[Sn[Sp>.6][-1]],
            'Sn at 70 Sp': [Sn[Sp>.7][-1]],
            'Sn at 80 Sp': [Sn[Sp>.8][-1]]
        }))


# %%
pdTable=pd.DataFrame(index = ['basc064', 'basc122', 'basc197'], columns=['AUROC', 'Sn at 60 Sp', 'Sn at 70 Sp', 'Sn at 80 Sp'])
for sAtlas in ['basc064', 'basc122', 'basc197']:
    for s in pdTable.columns:
        pdTable.loc[sAtlas, s]=f"{pdChart[pdChart['Atlas']==sAtlas][s].min()*100:0.0f}-{pdChart[pdChart['Atlas']==sAtlas][s].max()*100:0.0f}"


# %%
pdTable


# %%
plt.plot(fpr, tpr)


# %%



