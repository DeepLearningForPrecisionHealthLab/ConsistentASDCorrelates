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
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle as pkl
import sys
sys.path.append('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/')
from Parallelization.IMPAC_DenseNetwork import fReproduceModel
from Parallelization.IMPAC_DenseNetwork import read_config_file, format_dict,\
    add_layer_from_config, get_optimizer_from_config, metrics_from_string

np.random.seed(42)

# Function to return the best fitting model for a given featureset
def fGetBestModel(sDataPath, sFeatureName, bTrainable=True):
    """ Fetches the highest ROC AUC models in the IMPAC study for use in a stacking model

    :param sDataPath: the full path name to where the models are stored
    :param sFeatureName: the features used to train the model
    :param bTrainable: sets whether the model is trainable, i.e.
        whether the weights are locked and not trainable or not
    :return: the model
    """

    if not sFeatureName=='anatomy':
        sPerformanceTag='connectivity'+sFeatureName
    else:
        sPerformanceTag=sFeatureName

    # Initialize a dataframe to store values
    lsModelIndex=[]
    for i in range(50):
        lsModelIndex = lsModelIndex +['Model number '+str(i)]

    pdResults=pd.DataFrame(index=lsModelIndex, columns=['Mean Cross Val', 'Cross Val 1', 'Cross Val 2', 'Cross Val 3'])

    # Load each cross validation score and store it in a dataframe
    for i in range(50):
        sCVFile1 = sDataPath + '/Dense_' + str(i) + sPerformanceTag + 'ROCScoreCrossVal1.p'
        sCVFile2 = sDataPath + '/Dense_' + str(i) + sPerformanceTag + 'ROCScoreCrossVal2.p'
        sCVFile3 = sDataPath + '/Dense_' + str(i) + sPerformanceTag + 'ROCScoreCrossVal3.p'
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
    print(iBestModel)

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

#institute new stacked architecture from the ini file
def fStackedNetworkFromConfig(ini_path, kerConcatLayer, dModels, aInputShape=None, compiled=True):
    """ By Alex Treacher, altered by Cooper Mellema
    Should only work for this file

    This is an updated version of network_from_ini that needs more testing, but should be more stable, include more
        features and more automated in the long run
    This is designed to take a ini file and create a neural network from it
    Specific format is needed, the format can be seen from outputs of the above functions that create the ini files
    In order to be consistent a sequental model should start with a dedicated InputLayer
    :param ini_path: The location of the ini file
    :param api: The api that the model will be made from. Currently supports functional and sequential
    :return: keras model based on the ini. If api=='functional' and compiled==False:
         returns [input_layer, output_layer] so the network can be manipulated or compiled
    """
    # First, we load the .ini file as 'config'
    config = read_config_file(ini_path)

    #make sure the first layer is an input
    layers = [s for s in config.sections() if s.startswith('layer/')]
    strAPI = config['model']['class'].lower()

    #Initiate the model with the input layer
    if strAPI=='sequential':
        kerasModel = ker.models.Sequential()
        #add the input layer
        add_layer_from_config(kerasModel, config, layers[0], aInputShape, api=strAPI)

    if strAPI=='functional':
        input_config = dict(config[layers[0]])
        input_class = input_config['class']
        input_config.pop('class')
        format_dict(input_config)
        if input_class=='Dense':
            kerInputDenseLayer = ker.layers.Dense(input_config['units'])(kerConcatLayer)
            kerLayers = ker.layers.advanced_activations.LeakyReLU(alpha=input_config['alpha'])(kerInputDenseLayer)
        elif input_class=='BatchNormalization':
            kerLayers = ker.layers.BatchNormalization(momentum=input_config['momentum'],
                                                      epsilon=input_config['epsilon'])(kerConcatLayer)

    #remove the input layer so as to not try to add it again later
    layers = layers[1:]

    # add the layers
    for intI, strLayer in enumerate(layers):
        input_config = dict(config[layers[intI]])
        input_class = input_config['class']
        input_config.pop('class')
        format_dict(input_config)
        if not intI + 1 == layers.__len__(): #last layer
            if input_class == 'Dense':
                kerLayers = ker.layers.Dense(input_config['units'])(kerLayers)
                kerLayers = ker.layers.advanced_activations.LeakyReLU(alpha=input_config['alpha'])(kerLayers)
            elif input_class == 'Dropout':
                kerLayers = ker.layers.Dropout(input_config['rate'])(kerLayers)
            elif input_class == 'BatchNormalization':
                kerLayers = ker.layers.BatchNormalization(momentum=input_config['momentum'], epsilon=input_config['epsilon'])(kerLayers)
        if intI + 1 == layers.__len__(): #last layer
            kerLayers = ker.layers.Dense(1, activation='sigmoid')(kerLayers)


    #compile if compiled==True.
    if compiled == True:
        #get the optimzier
        optimizer = get_optimizer_from_config(config)
        if isinstance(optimizer, type(None)):
            raise KeyError("Opimizer not found in %s. Please include one or set compiled = False."%ini_path)
        #get the metrics
        try:
            strMetrics = config['model']['metrics']
        except KeyError:
            strMetrics = 'accuracy'
        metrics = metrics_from_string(strMetrics)
        if strAPI == 'functional':
            kerModel = ker.Model(inputs=[dModels[lsFeatureTags[0]].layers[0].input,
                                         dModels[lsFeatureTags[1]].layers[0].input,
                                         dModels[lsFeatureTags[2]].layers[0].input,
                                         dModels[lsFeatureTags[3]].layers[0].input,
                                         dModels[lsFeatureTags[4]].layers[0].input,
                                         dModels[lsFeatureTags[5]].layers[0].input
                                         ],
                                 outputs=kerLayers)
        kerModel.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=metrics)
        return kerModel
    #return non-compiled model
    else:
        if strAPI == 'functional':
            kerModel = ker.Model(inputs=[dModels[lsFeatureTags[0]].layers[0].input,
                                         dModels[lsFeatureTags[1]].layers[0].input,
                                         dModels[lsFeatureTags[2]].layers[0].input,
                                         dModels[lsFeatureTags[3]].layers[0].input,
                                         dModels[lsFeatureTags[4]].layers[0].input,
                                         dModels[lsFeatureTags[5]].layers[0].input
                                         ],
                                 outputs=kerLayers)
            return kerModel
        else:
            return kerasModel

def fCrossVal3Split(aX, aY):
    """ Splits data into 3x cross validation

    :param aX: array of X input feature data
    :param aY: array of Y target data
    :return: list of arrays of splits 1-3 for the X and Y data
    """
    # Initialize for splitting for 3x cross validation
    iSplitSize = int(aX.shape[0] / 3)

    # Split the Data for 3x cross validation
    aX1 = aX[iSplitSize:, :, :, :]  # skip over beginning
    aX2 = np.append(aX[:iSplitSize, :], aX[(2 * iSplitSize):, :], axis=0)  # split over middle
    aX3 = aX[:2 * iSplitSize, :, :, :]  # skip over end
    lsXCV = [aX1, aX2, aX3]

    aY1 = aY[iSplitSize:, :] # skip over beginning
    aY2 = np.append(aY[:iSplitSize, :], aY[(2 * iSplitSize):, :], axis=0)  # split over middle
    aY3 = aY[:(2 * iSplitSize), :] # skip over end
    lsYCV = [aY1, aY2, aY3]

    aXV1 = aX[:iSplitSize, :, :, :]  # include only beginning
    aXV2 = aX[iSplitSize:(2 * iSplitSize), :, :, :]  # include only middle
    aXV3 = aX[(2 * iSplitSize):, :, :, :]  # include only end
    lsXVal = [aXV1, aXV2, aXV3]

    aYV1 = aY[:iSplitSize, :] # include only beginning
    aYV2 = aY[iSplitSize:(2 * iSplitSize), :]  # include only middle
    aYV3 = aY[(2 * iSplitSize):, :]  # include only end
    lsYVal = [aYV1, aYV2, aYV3]

    return lsXCV, lsYCV, lsXVal, lsYVal

def fStackedCrossValSplit(dXData, aYData, lsFeatureTags):
    """ Splits the data into training and validation data for the stacking model
    :param dXData: a dictionary of the features used
    :param aYData: an array of the targets in the training Data
    :return: dXDataCV, dYDataCV, dXVal, dYVal: Dictionaries containing the train/validation
    data splits
    """

    # Initialize dictionaries to store the input features
    dXDataCV={0: [], 1: [], 2: []}
    dXVal={0: [], 1: [], 2: []}

    # create individual cross validation splits and then combine them in the
    # dictionary for cross val number i
    for i in range(len(lsFeatureTags)):
        if i==0:
            aXData = dXData[lsFeatureTags[i]]
            lsXCV, lsYCV, lsXVal, lsYVal = fCrossVal3Split(aXData, aYData)

        else:
            aXData = dXData['connectivity'][lsFeatureTags[i]]
            lsXCV, lsYCV, lsXVal, lsYVal = fCrossVal3Split(aXData, aYData)

        for i in range(3):
            dXDataCV[i].append(lsXCV[i])
            dXVal[i].append(lsXVal[i])

    return dXDataCV, lsYCV, dXVal, lsYVal

def fRunStacked(sDataPath, sINIPath, sSavePath, sModel, bFitFull=False, iCV=0):
    """Trains the stacked model

    :param dModels: dictionary of the models to be used for stacking
    :param sINIPath:
    :param sSavePath: the full path where the models are to be saved
    :param bFitFull: False if doing cross validation, True if fitting using all data
    :return: NA, but saves the history and weights (to sSavePath)
    """

    lsFeatureTags = [
        'anatomy',
        'basc122',
        'craddock_scorr_mean',
        'harvard_oxford_cort_prob_2mm',
        'msdl',
        'power_2011'
    ]

    dModels = {}

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
        if key == 'anatomy':
            dXData[key] = np.expand_dims(dXData[key], axis=1)
            dXData[key] = np.expand_dims(dXData[key], axis=3)

            dXTest[key] = np.expand_dims(dXTest[key], axis=1)
            dXTest[key] = np.expand_dims(dXTest[key], axis=3)

            dXData[key] = np.float32(dXData[key])
            dXTest[key] = np.float32(dXTest[key])
        elif key == 'connectivity':
            for innerkey in dXData[key].keys():
                dXData[key][innerkey] = np.expand_dims(dXData[key][innerkey], axis=1)
                dXData[key][innerkey] = np.expand_dims(dXData[key][innerkey], axis=3)

                dXTest[key][innerkey] = np.expand_dims(dXTest[key][innerkey], axis=1)
                dXTest[key][innerkey] = np.expand_dims(dXTest[key][innerkey], axis=3)

                dXData[key][innerkey] = np.float32(dXData[key][innerkey])
                dXTest[key][innerkey] = np.float32(dXTest[key][innerkey])
        else:
            None

    # Save each most succesful model into a dictionary
    sTrainedModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/IndividualInputs'
    for sFeatureTag in lsFeatureTags:
        dModels[sFeatureTag] = fGetBestModel(sTrainedModelPath, sFeatureTag)

    # concatenate model here, concatenating only 1 layer below the decision layer
    kerConcatLayer = ker.layers.concatenate([ker.layers.Flatten()(dModels['anatomy'].layers[-5].output),
                                             ker.layers.Flatten()(dModels['basc122'].layers[-5].output),
                                             ker.layers.Flatten()(dModels['craddock_scorr_mean'].layers[-5].output),
                                             ker.layers.Flatten()(dModels['harvard_oxford_cort_prob_2mm'].layers[-5].output),
                                             ker.layers.Flatten()(dModels['msdl'].layers[-5].output),
                                             ker.layers.Flatten()(dModels['power_2011'].layers[-5].output)
                                             ], axis=-1)

    kerDecisionLayer = fStackedNetworkFromConfig(sINIPath, kerConcatLayer, dModels, compiled=False)


    # Feed in individual inputs to the model along with validation data
    kerModel = ker.Model(inputs=[dModels[lsFeatureTags[0]].layers[0].input,
                                 dModels[lsFeatureTags[1]].layers[0].input,
                                 dModels[lsFeatureTags[2]].layers[0].input,
                                 dModels[lsFeatureTags[3]].layers[0].input,
                                 dModels[lsFeatureTags[4]].layers[0].input,
                                 dModels[lsFeatureTags[5]].layers[0].input
                                 ],
                         outputs=kerDecisionLayer.output)

    #Get the total number of layers to lock
    iLockLayers = len(dModels[lsFeatureTags[0]].layers)\
                  +len(dModels[lsFeatureTags[1]].layers)\
                  +len(dModels[lsFeatureTags[2]].layers)\
                  +len(dModels[lsFeatureTags[3]].layers)\
                  +len(dModels[lsFeatureTags[4]].layers)\
                  +len(dModels[lsFeatureTags[5]].layers)

    # Initialize the optimizer
    nadam=ker.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # Lock the old layers
    for layer in kerModel.layers[:iLockLayers]:
        layer.trainable = False

    # Initialize the early stopping criteria
    kerStopping = ker.callbacks.EarlyStopping(monitor='acc', min_delta=0.01, patience=20, restore_best_weights=True)

    # Compile the model
    kerModel.compile(optimizer=nadam,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    # Fit each of 3 cross-validations
    if not bFitFull:
        dXDataCV, lsYDataCV, dXVal, lsYVal = fStackedCrossValSplit(dXData, aYData, lsFeatureTags)
        ##for iCV in range(3): # Do 3x Cross Validation
            # Do not fit the model if the model has already been trained
        if not os.path.isfile(os.path.join(sSavePath, sModel, ('CrossVal'+str(iCV)+'ModelHistory.p'))):
            History = kerModel.fit(dXDataCV[iCV], lsYDataCV[iCV],
                                    validation_data=(dXVal[iCV], lsYVal[iCV]),
                                    epochs=500, batch_size=128,
                                    callbacks=[kerStopping])

            History=History.history

            # Predict on the Validation data
            aPredicted = kerModel.predict(dXVal[iCV])

            flROCScore = sk.metrics.roc_auc_score(lsYVal[iCV], aPredicted)

            # Save the model weights and history
            kerModel.save_weights(os.path.join(sSavePath, (sModel + 'CrossVal'+str(iCV)+'ModelWeights.h5')), 'wb')
            pkl.dump(aPredicted, open(os.path.join(sSavePath, (sModel + 'CrossVal'+str(iCV)+'ModelPredicted.p')), 'wb'))
            pkl.dump(flROCScore, open(os.path.join(sSavePath, (sModel + 'CrossVal'+str(iCV)+'ModelROCScore.p')), 'wb'))
            pkl.dump(History, open(os.path.join(sSavePath, (sModel + 'CrossVal'+str(iCV)+'ModelHistory.p')), 'wb'))

            del(kerModel)
            tf.reset_default_graph()
            ker.backend.clear_session()

    # Fit the FULL model (only the best model after cross validation)
    elif bFitFull:
        # Do not fit the model if the model has already been trained
        if not os.path.isfile(os.path.join(sSavePath, sModel, 'FullModelHistory.p')):
            FullHistory = kerModel.fit([dXData[lsFeatureTags[0]],
                                        dXData['connectivity'][lsFeatureTags[1]],
                                        dXData['connectivity'][lsFeatureTags[2]],
                                        dXData['connectivity'][lsFeatureTags[3]],
                                        dXData['connectivity'][lsFeatureTags[4]],
                                        dXData['connectivity'][lsFeatureTags[5]]
                                        ], aYData, epochs=500, batch_size=128,
                                        callbacks=[kerStopping])

            FullHistory=FullHistory.history

            # Predict on the test data
            aPredicted = kerModel.predict([dXTest[lsFeatureTags[0]],
                                           dXTest['connectivity'][lsFeatureTags[1]],
                                           dXTest['connectivity'][lsFeatureTags[2]],
                                           dXTest['connectivity'][lsFeatureTags[3]],
                                           dXTest['connectivity'][lsFeatureTags[4]],
                                           dXTest['connectivity'][lsFeatureTags[5]]
                                           ])

            flROCScore = sk.metrics.roc_auc_score(aYTest, aPredicted)

            # Save the model weights and history
            kerModel.save_weights(os.path.join(sSavePath, (sModel + 'FullModelWeights.h5')))
            pkl.dump(aPredicted, open(os.path.join(sSavePath, (sModel + 'FullModelPredicted.p')), 'wb'))
            pkl.dump(flROCScore, open(os.path.join(sSavePath, (sModel + 'FullModelROCScore.p')), 'wb'))
            pkl.dump(FullHistory, open(os.path.join(sSavePath, (sModel + 'FullModelHistory.p')), 'wb'))

##################################Below actually runs the code####################################

if '__main__' == __name__:

    # Initialize the paths used
    sModelPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/IndividualInputs'

    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

    # Fetch the name of the input features and model architecture number from the parallel wrapper
    sInfo = sys.argv[1]
    sModel = sInfo.split('.')[-2]
    sModel = sModel.split('/')[-1]
    # sModel='Stack_18'

    sINIPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sModel + '.ini'

    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Stacked/EarlyChopped'

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

    # Fit the models
    bCV0 = False
    bCV1 = True
    bCV2 = False
    bFull = False

    if bCV0:
        fRunStacked(sDataPath, sINIPath, sSavePath, sModel, bFitFull=False, iCV=0)
    elif bCV1:
        fRunStacked(sDataPath, sINIPath, sSavePath, sModel, bFitFull=False, iCV=1)
    elif bCV2:
        fRunStacked(sDataPath, sINIPath, sSavePath, sModel, bFitFull=False, iCV=2)
    elif bFull:
        fRunStacked(sDataPath, sINIPath, sSavePath, sModel, bFitFull=True)
