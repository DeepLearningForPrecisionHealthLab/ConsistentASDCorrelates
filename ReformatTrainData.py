#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 28 Aug, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import pickle as pkl
import numpy as np
import gc
import glob
from joblib import Parallel, delayed

def fReformatHistory(sHistoryFile, sMetric='acc'):
    """
    Takes the keras history object and just saves the training score instead of the big object
    :param sHistoryFile: path to the history file
    :return: dumps a new pickled file of just the training score
    """
    # disable gc for a speedup
    cFile=open(sHistoryFile,'rb')
    gc.disable()
    cModelHist=pkl.load(cFile)
    gc.enable()
    cFile.close()

    # dump training perf
    flTrainPerf=cModelHist.history[sMetric][np.argmax(cModelHist.history[f'val_{sMetric}'])]
    pkl.dump(flTrainPerf, open(sHistoryFile.replace('ModelHistory', f'Train{sMetric.upper()}Score'),'wb'))

if '__main__' == __name__:
    for sConnMetric in ['Correlation', 'Covariance', 'PartialCorrelation', 'Precision', 'LSGC']:
        for sType in ['Dense', 'LSTM']:
            lsTestResults = glob.glob(f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/'
                      f'Parallelization/TrainedModels/{sConnMetric}/{sType}/*ModelHistoryCrossVal*')

            Parallel(n_jobs=32)(delayed(fReformatHistory)(sFile) for sFile in lsTestResults)
