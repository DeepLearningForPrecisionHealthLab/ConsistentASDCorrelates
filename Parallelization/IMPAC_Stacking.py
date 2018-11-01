"""This file implements a stacking scheme for predicting ASD vs HC on the IMPAC Dataset

The stacking scheme comes in two phases, first is the base models for prediction.
The Base models are chosen from our broad search of the different ML models. DL models
performed the best, and as such, we are constraining the base models to DL frameworks.
Each highest performing DL network is combined in the following way into a stacked
framework, depicted here graphically:

featureset1 ->  Highest performing ---> Prediction \
                    network on                      \
                featureset1 from                     \
                previous experiment                   \
                                                       \
featureset2 ->  Highest performing ---> Prediction -------> Another network ----> Final
                    network on                        /    (trained in this     Prediction
                featureset2 from                     /           code)
                previous experiment                 /
                                                   /
featureset3 ->  Highest performing ---> Prediction
                    network on
                featureset3 from
                previous experiment

...(etc)

The feature-sets used are as follows:
- 243 anatomical features
- 7503 features from the FC matrix calculated from the BASC Atlas with 122 parcellations
- 31125 features from the FC matrix calculated from the Craddock Atlas with 249 parcellations
- 1176 features from the FC matrix calculated from the Harvard-Oxford Atlas with 69 parcellations
- 780 features from the FC matrix calculated from the MSDL Atlas with 39 parcellations
- 34980 features from the FC matrix calculated from the Power Atlas with 264 parcellations

Several training schemes will be investigated:
- Training only the top network
- Removing the decision layer and training only the top network (transfer learning rather than
        stacking)
- Training the top network and then tuning the other networks

"""
# Imports and path initializations
import numpy as np
import keras as ker
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import os
import pickle as pkl
from Parallelization.IMPAC_DenseNetwork import fReproduceModel

sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/IndividualInputs'

sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

lsFeatures = [
    'Anatomical',
    'BASC 122 ROI',
    'Craddock 449 ROI',
    'Harvard-Oxford 69 ROI',
    'MSDL 39 ROI',
    'Power 264 ROI'
]

lsFeatureTags = [
    'anatomy',
    'basc122',
    'craddock_scorr_mean',
    'harvard_oxford_cort_prob_2mm',
    'msdl',
    'power_2011'
]

dModels={}

# Load the master file of the data pre-organized into train, test
[dXData, dXTest, aYData, aYTest] = pkl.load(open(sDataPath, 'rb'))

# Reshape and Retype the data for use
aYData = np.float32(aYData)
aYTest = np.float32(aYTest)

for key in dXData.keys():
    # The required dimensions for the dense network is size
    # N x H x W x C, where N is the number of samples, C is
    # the number of channels in each sample, and, H and W are the
    # spatial dimensions for each sample.
    if key=='anatomy':
        dXData[key] = np.expand_dims(dXData[key], axis=1)
        dXData[key] = np.expand_dims(dXData[key], axis=3)

        dXTest[key] = np.expand_dims(dXTest[key], axis=1)
        dXTest[key] = np.expand_dims(dXTest[key], axis=3)

        dXData[key] = np.float32(dXData[key])
        dXTest[key] = np.float32(dXTest[key])
    elif key=='connectivity':
        for innerkey in dXData[key].keys():
            dXData[key][innerkey] = np.expand_dims(dXData[key][innerkey], axis=1)
            dXData[key][innerkey] = np.expand_dims(dXData[key][innerkey], axis=3)

            dXTest[key][innerkey] = np.expand_dims(dXTest[key][innerkey], axis=1)
            dXTest[key][innerkey] = np.expand_dims(dXTest[key][innerkey], axis=3)

            dXData[key][innerkey] = np.float32(dXData[key][innerkey])
            dXTest[key][innerkey] = np.float32(dXTest[key][innerkey])
    else:
        None

# Function to return the best fitting model for a given featureset
def fGetBestModel(sDataPath, sFeatureName, bTrainable=False):

    # Initialize a dataframe to store values
    lsModelIndex=[]
    for i in range(50):
        lsModelIndex = lsModelIndex +['Model number '+str(i)]

    pdResults=pd.DataFrame(index=lsModelIndex, columns=['Mean Cross Val', 'Cross Val 1', 'Cross Val 2', 'Cross Val 3'])

    # Load each cross validation score and store it in a dataframe
    for i in range(50):
        sCVFile1 = sDataPath + '/Dense_' + str(i) + sFeatureName + 'CrossVal1.p'
        sCVFile2 = sDataPath + '/Dense_' + str(i) + sFeatureName + 'CrossVal2.p'
        sCVFile3 = sDataPath + '/Dense_' + str(i) + sFeatureName + 'CrossVal3.p'
        if os.path.isfile(sCVFile1) and os.path.isfile(sCVFile2) and os.path.isfile(sCVFile3):
            flCV1 = pkl.load(open(sCVFile1, 'rb'))
            flCV2 = pkl.load(open(sCVFile2, 'rb'))
            flCV3 = pkl.load(open(sCVFile3, 'rb'))
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

    # Get '00' for model 0, '01' for model 1, etc. to fetch correct .ini files
    if iBestModel < 10:
        sBestModel = '0'+str(iBestModel)
    else:
        sBestModel = str(iBestModel)

    if sFeatureName == 'anatomy':
        sWeightsPath = sModelPath + '/Dense_' + sBestModel + '_' + sFeatureName + 'weights.h5'
        kerModel = fReproduceModel(sFeatureName, sBestModel, sWeightsPath)
    else:
        sWeightsPath = sModelPath + '/Dense_' + sBestModel + '_connectivity' + sFeatureName + 'weights.h5'
        kerModel = fReproduceModel('connectivity', sBestModel, sWeightsPath, sSubInputName=sFeatureName)

    kerModel.trainable=bTrainable
    return kerModel

# Save each most succesful model into a dictionary
for sFeatureTag in lsFeatureTags:
    dModels[sFeatureTag] = fGetBestModel(sDataPath, sFeatureTag)

# concatenate model here, concatenating only 1 layer below the decision layer
kerConcatLayer = ker.layers.concatenate([dModels['anatomy'].layers[-1].output,
                                         dModels['basc122'].layers[-1].output,
                                         dModels['craddock_scorr_mean'].layers[-1].output,
                                         dModels['harvard_oxford_cort_prob_2mm'].layers[-1].output,
                                         dModels['msdl'].layers[-1].output,
                                         dModels['power_2011'].layers[-1].output
                                         ], axis=-1)

###########################Test Model architecture########################################
# Replace later with a more thought-out search for best architecture at this point
iWidth = 16
kerDenseLayer = ker.layers.Dense(iWidth)(kerConcatLayer)
#kerLeakyReLU = ker.layers.advanced_activations.LeakyReLU(alpha=0.1)(kerDenseLayer)

#kerDenseLayer2 = ker.layers.Dense(iWidth/2)(kerLeakyReLU)
#kerLeakyReLU2 = ker.layers.advanced_activations.LeakyReLU(alpha=0.1)(kerDenseLayer2)

kerDecisionLayer = ker.layers.Dense(1, activation='sigmoid')(kerDenseLayer)
###########################End Test Arch###########################################

# Create Input layers for the taking in the data
def fInputLayer(sDataType, sAtlas):
    if sDataType=='anatomy':
        aDataShape = [1, dXData[sDataType].shape[1], 1]
    else:
        aDataShape = [1, dXData[sDataType][sAtlas].shape[1], 1]
    kerInput = ker.layers.Input(shape=aDataShape, name=sDataType+sAtlas)
    return kerInput

# Feed in individual inputs to the model along with validation data
kerModel = ker.Model(inputs=[fInputLayer('anatomy'),
                             fInputLayer('connectivity', lsFeatureTags[1]),
                             fInputLayer('connectivity', lsFeatureTags[2]),
                             fInputLayer('connectivity', lsFeatureTags[3]),
                             fInputLayer('connectivity', lsFeatureTags[4]),
                             fInputLayer('connectivity', lsFeatureTags[5])
                             ],
                     outputs=kerDecisionLayer)

# Compile the model
kerModel.compile(optimizer='nadam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
