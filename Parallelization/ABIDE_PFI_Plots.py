#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This compares the IMPAC and ABIDE PFI results

Originally developed for use in the ASD comparison project
Created by Cooper Mellema on Dec 28,2021
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

import os
import sys
import matplotlib.pyplot as plt
import sklearn as sk
import pickle as pkl
import sklearn.metrics as skm
import seaborn as sns
sns.set_theme()
sns.set_style()
import pandas as pd
import numpy as np
import multiprocessing
import time
from scipy.stats import spearmanr, pearsonr
import glob

sys.path.append('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/')
from FeatureImportanceRevamp import fLoadImportances

def fLoadABIDEImportances(sImportanceLoc:str="/archive/bioinformatics/DLLab/CooperMellema/results/Autism/ABIDE_Tuned/ABIDE***/PFI_Tuned_64Permutations/Model*_CV0.csv", iAbide=1):
    lsFiles = glob.glob(sImportanceLoc.replace('***', f'{iAbide}'))
    lsImportances = [pd.read_csv(x, index_col=0)[['mean pfi']].T for x in lsFiles]
    pdABIDE = pd.concat(lsImportances)
    pdABIDE.index=[int(x.split('Model')[-1].split('_')[0]) for x in lsFiles]
    return pdABIDE.sort_index()

if '__main__'==__name__:
    pdIMPAC = fLoadImportances('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison8Permutations').drop('Atlas', axis=1)
    pdABIDE1 = fLoadABIDEImportances(iAbide=1)
    pdABIDE2 = fLoadABIDEImportances(iAbide=2)