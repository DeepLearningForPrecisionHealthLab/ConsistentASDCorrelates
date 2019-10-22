""" This file finds and plots the performances of models trained on the IMPAC dataset


"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import keras

sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun'

lsIndex = list(np.zeros(50))

for iModel in range(50):
    lsIndex[iModel] = 'Model number ' + str(iModel)

pdResults = pd.DataFrame(index=lsIndex, columns=['Mean Cross Val', 'Cross Val 1', 'Cross Val 2', 'Cross Val 3'])

for i in range(50):
    sCVFile1 = sDataPath+'/Dense_'+str(i)+'AllAtlasesROCScoreCrossVal1.p'
    sCVFile2 = sDataPath+'/Dense_'+str(i)+'AllAtlasesROCScoreCrossVal2.p'
    sCVFile3 = sDataPath+'/Dense_'+str(i)+'AllAtlasesROCScoreCrossVal3.p'
    if os.path.isfile(sCVFile1) and os.path.isfile(sCVFile2) and os.path.isfile(sCVFile3):
        flCV1 = pickle.load(open(sCVFile1, 'rb'))
        flCV2 = pickle.load(open(sCVFile2, 'rb'))
        flCV3 = pickle.load(open(sCVFile3, 'rb'))
        flMeanCVROC = (flCV1+flCV2+flCV3)/3
        pdResults.loc[('Model number ' + str(i)), 'Mean Cross Val'] = flMeanCVROC
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 1'] = flCV1
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 2'] = flCV2
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 3'] = flCV3

    else:
        pdResults.loc[('Model number ' + str(i)), 'Mean Cross Val'] = 0
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 1'] = 0
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 2'] = 0
        pdResults.loc[('Model number ' + str(i)), 'Cross Val 3'] = 0

pdResults = pdResults.infer_objects()
lsIndex = pdResults.index
iBestModel = lsIndex.get_loc(pdResults['Mean Cross Val'].idxmax())

for iBestModel in range(10):
    kmModelHistory = pickle.load(open(sDataPath+'/Dense_'+str(iBestModel)+'AllAtlasesModelHistory.p', 'rb'))
    kmModelHistoryCV1 = pickle.load(open(sDataPath+'/Dense_'+str(iBestModel)+'AllAtlasesModelHistoryCrossVal1.p', 'rb'))
    kmModelHistoryCV2 = pickle.load(open(sDataPath+'/Dense_'+str(iBestModel)+'AllAtlasesModelHistoryCrossVal2.p', 'rb'))
    kmModelHistoryCV3 = pickle.load(open(sDataPath+'/Dense_'+str(iBestModel)+'AllAtlasesModelHistoryCrossVal3.p', 'rb'))

    fig, ax = plt.subplots()
    x1 = kmModelHistory.history['acc']
    x2 = kmModelHistoryCV1.history['acc']
    x3 = kmModelHistoryCV2.history['acc']
    x4 = kmModelHistoryCV3.history['acc']
    ax.plot(range(len(x1)), x1, range(len(x2)), x2, range(len(x3)), x3, range(len(x4)), x4)

    plt.show()
