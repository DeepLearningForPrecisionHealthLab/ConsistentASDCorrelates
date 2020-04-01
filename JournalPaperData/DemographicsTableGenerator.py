#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 13 Mar, 2020
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import sys
import os
from Parallelization.BioMarkerIdentification import fLoadPDData
import pandas as pd
import seaborn as sns
import numpy as np

# Load IMPAC data
dXData, aYData = fLoadPDData()
pdIMPAC = dXData['basc064'][[col for col in dXData['basc064'].columns if not (('ROI' in col) or ('Site' in col) or (
        'anatomy' in col))]]
pdIMPAC['DX']=aYData

# Load ABIDE I data
sRoot = '/project/bioinformatics/DLLab/STUDIES/'
sABIDE1 = f'{sRoot}ABIDE1/Metadata/Phenotypic_V1_0b.csv'
pdABIDE1 = pd.read_csv(sABIDE1)
pdABIDE1 = pdABIDE1[pdABIDE1['SITE_ID']=='NYU'][['SITE_ID', 'SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']]

# Load ABIDE II data
sABIDE2 = f'{sRoot}ABIDE2/Metadata/ABIDEII_Composite_Phenotypic_Data.csv'
pdABIDE2 = pd.read_csv(sABIDE2, encoding="ISO-8859-1")
pdABIDE2 = pd.concat((pdABIDE2[pdABIDE2['SITE_ID']=='ABIDEII-NYU_1'][['SITE_ID', 'SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN ', 'SEX']],
           pdABIDE2[pdABIDE2['SITE_ID']=='ABIDEII-NYU_2'][['SITE_ID', 'SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN ', 'SEX']]))

print(len(pdIMPAC[pdIMPAC['Dx']==0].index), len(pdIMPAC[pdIMPAC['Dx']==1].index))

print(len(pdABIDE2[pdABIDE2['DX_GROUP']==1].index), len(pdABIDE2[pdABIDE2['DX_GROUP']==2].index))

print(len(pdABIDE1[pdABIDE1['DX_GROUP']==1].index), len(pdABIDE1[pdABIDE1['DX_GROUP']==2].index))
