#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ***Purpose*** 

This file *** 
    Example:
    
***Details***
    Subheadings
    
Originally developed for use in the *** project
Created by Cooper Mellema on 11 Nov, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import numpy as np
import pandas as pd
import sys
import os
from Parallelization.BioMarkerIdentification import fLoadPDData, fLoadModels, fExpandDataframe

def fAnalyze():
    analysis=0
    return analysis

if '__main__'==__name__:

    dXData, aYData = fLoadPDData()

    # NOTE, this uses the full model, not the output predictions
    # load up the models and the data
    dModels = {
               'Model1': fLoadModels('Dense', 'connectivity', 'basc064', 46)[0]#,
               # 'Model2': fLoadModels('Dense', 'connectivity', 'basc122', 2)[0],
               # 'Model3': fLoadModels('Dense', 'connectivity', 'basc197', 32)[0],
               # 'Model4': fLoadModels('LinRidge', 'connectivity', 'basc064')[0],
               # 'Model5': fLoadModels('LinRidge', 'connectivity', 'basc122')[0],
               # 'Model6': fLoadModels('LinRidge', 'connectivity', 'basc197')[0],
               # 'Model7': fLoadModels('Dense', 'combined', 'basc064', 43)[0],
               # 'Model8': fLoadModels('Dense', 'combined', 'basc122', 39)[0],
               # 'Model9': fLoadModels('Dense', 'combined', 'basc197', 15)[0],
               # 'Model10': fLoadModels('LinRidge', 'combined', 'basc064')[0],
               # 'Model11': fLoadModels('LinRidge', 'combined', 'basc122')[0],
               # 'Model12': fLoadModels('LinRidge', 'combined', 'basc197')[0],
               # 'Model13': fLoadModels('Dense', 'anatomy', iModelNum=44)[0],
               # 'Model14': fLoadModels('LinRidge', 'anatomy')[0]
    }

    # Reformat the data to the right form
    dFormattedXData = {
        'Model1': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('anatomy') or
                                                                                   x.__contains__('Site') or
                                                                                   x.__contains__('Sex(F=1)') or
                                                                                   x.__contains__('Age'))],axis=1)#,
        # 'Model2': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('anatomy') or
        #                                                                            x.__contains__('Site') or
        #                                                                            x.__contains__('Sex(F=1)') or
        #                                                                            x.__contains__('Age'))],axis=1),
        # 'Model3': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('anatomy') or
        #                                                                            x.__contains__('Site') or
        #                                                                            x.__contains__('Sex(F=1)') or
        #                                                                            x.__contains__('Age'))],axis=1),
        # 'Model4': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('anatomy') or
        #                                                                            x.__contains__('Site') or
        #                                                                            x.__contains__('Sex(F=1)') or
        #                                                                            x.__contains__('Age'))],axis=1),
        # 'Model5': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('anatomy') or
        #                                                                            x.__contains__('Site') or
        #                                                                            x.__contains__('Sex(F=1)') or
        #                                                                            x.__contains__('Age'))],axis=1),
        # 'Model6': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('anatomy') or
        #                                                                            x.__contains__('Site') or
        #                                                                            x.__contains__('Sex(F=1)') or
        #                                                                            x.__contains__('Age'))],axis=1),
        # 'Model7': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model8': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model9': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model10': dXData['basc064'].drop([x for x in dXData['basc064'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model11': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model12': dXData['basc197'].drop([x for x in dXData['basc197'].columns if (x.__contains__('Age'))], axis=1),
        # 'Model13': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('ROI')or
        #                                                                             x.__contains__('Age'))], axis=1),
        # 'Model14': dXData['basc122'].drop([x for x in dXData['basc122'].columns if (x.__contains__('ROI')or
        #                                                                             x.__contains__('Age'))], axis=1)
    }
