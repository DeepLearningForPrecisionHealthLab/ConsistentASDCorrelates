#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file reformats keras history objects to arrays rather than the native keras files
    Example: initial history object=180k kB, after, history object = 3kB

    
Originally developed for use in the IMPAC Analysis project
Created by Cooper Mellema on 03 Dec, 2018
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import keras
import os
import pickle as pkl
import tensorflow as tf

config = tf.ConfigProto()
session = tf.Session(config=config)
config.gpu_options.allow_growth = True

def fReformatModelHist(iModel, sDataPath):
    if iModel<10:
        sModel='Stack_0'+str(iModel)
    else:
        sModel='Stack_'+str(iModel)
    print('   '+sModel)

    for iCV in range(3):
        if os.path.isfile(os.path.join(sDataPath, (sModel+'CrossVal'+str(iCV)+'ModelHistory.p'))):
            ModelHist=pkl.load(open(os.path.join(sDataPath, (sModel+'CrossVal'+str(iCV)+'ModelHistory.p')), 'rb'))
            if not type(ModelHist)==dict:
                ModelHist=ModelHist.history
                pkl.dump(ModelHist, open(os.path.join(sDataPath, (sModel+'CrossVal'+str(iCV)+'ModelHistory.p'), 'wb')))

        else:
            pass

if __name__ == '__main__':

    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Stacked' \
                '/RegularizationIncluded'

    for root, dirs, files in os.walk(sDataPath):
        for dir in dirs:
            sPath = os.path.join(sDataPath, dir)

            for i in range(50):
                fReformatModelHist(i, sPath)
