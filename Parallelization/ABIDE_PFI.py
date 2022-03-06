#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file Generically has a class to run permutation feature importance testing

This file, if run alone, will perform the PFI testing on the top 5 TSE models in the IMPAC study

Originally developed for use in the ASD comparison project
Created by Cooper Mellema on 11 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

import os
import sys
# import matplotlib.pyplot as plt
# plt.style.use('dark_background')
# from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
import pickle as pkl
import sklearn.metrics as skm
# import seaborn as sb
import pandas as pd
import numpy as np
import keras
import keras as k
import multiprocessing
import time
import glob
import types


from joblib import Parallel, delayed
from joblib import parallel_backend
from joblib import wrap_non_picklable_objects
sys.path.append('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/')
from IMPAC_DenseNetwork import read_config_file
from IMPAC_DenseNetwork import format_dict
from IMPAC_DenseNetwork import metrics_from_string
from IMPAC_DenseNetwork import get_optimizer_from_config
from IMPAC_DenseNetwork import add_layer_from_config
# from IMPAC_LSTM import fLoadLSTM
from IMPAC_DenseNetwork import network_from_ini_2
from sklearn.inspection import permutation_importance
from BioMarkerIdentification import *

def fLoadTunedModel(sGlobbablePath='/archive/bioinformatics/DLLab/CooperMellema/results/Autism/ABIDE_Tuned/ABIDE*/Model*_CV0', idx=0):
    lsModels = glob.glob(sGlobbablePath)
    lsModels.sort()
    cModel=keras.models.load_model(lsModels[idx])
    sModel=lsModels[idx].split('/')[-1]
    iAbide=int(lsModels[idx].split('/')[-2][-1])
    return cModel, sModel, iAbide

def fFormatTunedX_single(sModel, dAbide, idxAbide, iAbide):
    lsAtlases=['064', '122', '197']
    iModel = int(sModel.split('Model')[1].split('_')[0])
    sAtlas = f'BASC{lsAtlases[(iModel-1)%3]}'
    pdX=dAbide[sAtlas].iloc[idxAbide[iAbide],11:].drop([x for x in dAbide[sAtlas].iloc[idxAbide[iAbide],11:].columns if ('Age' in x)], axis=1)
    return pdX

def fPermuteModelSklearn(nPermutations, aYDat, pdXDat, cModel, sModel, sSaveDir):
    # define scorer function for PFI
    def _scorer(self,x,y):
        return skm.roc_auc_score(y, self.predict(np.expand_dims(np.expand_dims(x, axis=1), axis=3)))

    # create object with predict and score
    cPredictor = k.models.Model(inputs=cModel.inputs, outputs=cModel.outputs)
    cPredictor._make_predict_function()
    cPredictor.score = types.MethodType(_scorer, cPredictor)
    assert hasattr(cPredictor, 'score')

    # perform PFI
    pfi=permutation_importance(cPredictor, pdXDat.values, aYDat, n_repeats=nPermutations, random_state=0)
    pdPFI = pd.DataFrame.from_dict({'mean pfi':pfi.importances_mean, 'std pfi':pfi.importances_std})
    pdPFI.index=pdXDat.columns
    pdPFI.to_csv(os.path.join(sSaveDir, f'{sModel}.csv'))
    return


if '__main__'==__name__:
    idx = int(sys.argv[1])

    nPermutations=64

    #load models
    import tensorflow as tf
    from ABIDE_TestIMPACModels import fLoadABIDE_Data
    cModel, sModel, iAbide = fLoadTunedModel(idx=idx)
    dAbide = fLoadABIDE_Data()
    aYData = np.expand_dims(dAbide['BASC122']['ASD'].values, axis=-1)

    # Reformat the data to the right form
    idxAbide={}
    idxAbide[1]=(dAbide['BASC122']['ABIDE']==1).values
    idxAbide[2]=(dAbide['BASC122']['ABIDE']==2).values
    pdX = fFormatTunedX_single(sModel, dAbide, idxAbide, iAbide)


    # Permute the models
    sSaveDir=f'/archive/bioinformatics/DLLab/CooperMellema/results/Autism/ABIDE_Tuned/'\
        f'ABIDE{iAbide}/PFI_Tuned_{nPermutations}Permutations'
    os.makedirs(sSaveDir, exist_ok=True)
    print(f'Performing PFI for model: {sModel} on dataset: ABIDE{iAbide}')
    fPermuteModelSklearn(nPermutations, aYData[idxAbide[iAbide]], pdX, cModel, sModel, sSaveDir=sSaveDir)
    print(f'Saving results to: {sSaveDir} as {sModel}.csv')
