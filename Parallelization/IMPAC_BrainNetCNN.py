""" This code takes the Paris IMPAC Autism study data,and
trains a dense network as a classifier

The performance metrics being used are:

    - accuracy
    - Precision-Recall area under curve
    - F1 score
    - ROC area under curve

Written by Cooper Mellema in the Montillo Deep Learning Lab at UT Southwesten Medical Center
Sept 2018
"""

import numpy as np
import os
import sys
import ast
import configparser
import pandas as pd
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, auc, f1_score
import pickle

sBrainNetCNNCode = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master'
sBrainNetCNNCode2 = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master/ann4brains'
sys.path.append(sBrainNetCNNCode)
sys.path.append(sBrainNetCNNCode2)
# import BrainNetCNNCode
from ann4brains.nets import BrainNetCNN

###############################################################################
def read_config_file(path):
    if not os.path.exists(path):
        raise IOError('ini_loc: %r does not exisit'%path)
    #load the ini file
    config = configparser.ConfigParser()
    config.read(path)
    return config

def format_dict(dict):
    """
    Takes a dictionary and attempts to make the dictionary values their native python class
    EG {'a':'[1,2,3]','b':'5','c':'foo'} = {'a':[1,2,3],'b':5,'c':'foo'} where 5 is an int and [1,2,3] a list
    :param dct: The dict to change
    :return: The formatted dct, note this is done inplace!!!
    """
    for key, value in dict.items():  # Attempt to turn the type of the values into something other than a string
        try:
            dict[key] = ast.literal_eval(value)
        except ValueError:
            dict[key] = value
    return dict

# def metrics_from_string(metrics):
#     """
#     Take a list of stings or a string of a list returns a list of the the metric(s) so they can be passed into the
#         compile method in a keras model.
#     Some metrics can be passed to keras as a string, these metrics are found in listStrMetrics, if one is missing please
#         add it.
#     :param metric_string:
#     :return: a list of metrics for keras.
#     """
#     listStrMetrics = ['acc', 'accuracy', 'mse', 'mae', 'mape', 'cosine']
#     strMetrics = str(metrics)
#     listMetricStrings = re.findall(r'[A-Za-z_]+',strMetrics)
#     listMetrics = []
#     for strMetric in listMetricStrings:
#         if strMetric in listStrMetrics:
#             kerasMetric = strMetric
#         else:
#             try:
#                 kerasMetric = getattr(keras.metrics, strMetric)
#             except:
#                 raise ValueError("%s is not a valid keras metric"%strMetrics)
#         listMetrics.append(kerasMetric)
#     return listMetrics

# def get_optimizer_from_config(config):
#     if 'optimizer' in config.keys():
#         #if there's an optimizer
#         optimizer_dct = dict(config['optimizer'])
#         format_dict(optimizer_dct)
#         optimizer_class = getattr(keras.optimizers, optimizer_dct['class'])
#         optimizer_dct.pop('class')
#         optimizer = optimizer_class(**optimizer_dct)
#         return optimizer
#     else:
#         return None

# def add_layer_from_config(model, config, config_key, aInputShape, api='functional'):
#     """ By Alex Treacher
#     This will add a layer to a model from a config file. This is used to build/edit models from an ini file
#     :param model: the keras model to add the layer to
#     :param config: the config that has the ini for the layer
#     :param config_key: the key for the config
#     :param api: which keras API (functional or sequential) is being used to build the model
#     :return: Sequential API: the model with the layer added
#         Functional APU: the last layer that was added
#     """
#     ini_dict = dict(config[config_key])
#     format_dict(ini_dict)
#     class_to_add = getattr(keras.layers, ini_dict['class'])
#
#     sActivation=None
#     flAlpha=None
#
#     if 'activation' in ini_dict.keys():
#         if ini_dict['activation']=='relu':
#             sActivation = ini_dict['activation']
#             flAlpha = ini_dict['alpha']
#             ini_dict.pop('activation')
#             ini_dict.pop('alpha')
#
#     #Pull out the regularization magnitude
#     if 'regularizer' in ini_dict.keys():
#         flL2Alpha=ini_dict['regularizer']
#         ini_dict.pop('regularizer')
#         ini_dict['kernel_regularizer']=regularizers.l2(flL2Alpha)
#
#     ini_dict.pop('class')
#     if aInputShape is not None:
#         ini_dict['input_shape']=aInputShape
#     if api.lower() == 'sequential': #sequential return the model with added layer
#         model.add(class_to_add(**ini_dict))
#         if sActivation is not None:
#             if sActivation == 'relu':
#                 model.add(keras.layers.advanced_activations.LeakyReLU(alpha=flAlpha))
#         return model
#     if api.lower() == 'functional':  #functional model return the new layer
#         output_layer = class_to_add(**ini_dict)(model)
#         return output_layer
#
# def network_from_ini_2(ini_path, aInputShape=None, compiled=True):
#     """ By Alex Treacher
#     This is an updated version of network_from_ini that needs more testing, but should be more stable, include more
#         features and more automated in the long run
#     This is designed to take a ini file and create a neural network from it
#     Specific format is needed, the format can be seen from outputs of the above functions that create the ini files
#     In order to be consistent a sequental model should start with a dedicated InputLayer
#     :param ini_path: The location of the ini file
#     :param api: The api that the model will be made from. Currently supports functional and sequential
#     :return: keras model based on the ini. If api=='functional' and compiled==False:
#          returns [input_layer, output_layer] so the network can be manipulated or compiled
#     """
#     # First, we load the .ini file as 'config'
#     config = read_config_file(ini_path)
#
#     #make sure the first layer is an input
#     layers = [s for s in config.sections() if s.startswith('layer/')]
#     strAPI = config['model']['class'].lower()
#
#     #Initiate the model with the input layer
#     if strAPI=='sequential':
#         kerasModel = keras.models.Sequential()
#         #add the input layer
#         add_layer_from_config(kerasModel, config, layers[0], aInputShape, api=strAPI)
#
#     if strAPI=='functional':
#         input_config = dict(config[layers[0]])
#         input_class = getattr(keras.layers, input_config['class'])
#         input_config.pop('class')
#         format_dict(input_config)
#         kerasModel = input_class(**input_config)
#         layerFunctionalInput = kerasModel
#     #remove the input layer so as to not try to add it again later
#     layers = layers[1:]
#
#     # add the layers
#     for intI, strLayer in enumerate(layers):
#         if intI + 1 == layers.__len__(): #last layer
#             kerasModel.add(keras.layers.Flatten())
#         kerasModel = add_layer_from_config(kerasModel, config, strLayer, aInputShape, api=strAPI)
#         if intI + 1 == layers.__len__(): #last layer
#                 layerFunctionalOutput = kerasModel
#
#     #compile if compiled==True.
#     if compiled == True:
#         #get the optimzier
#         optimizer = get_optimizer_from_config(config)
#         if isinstance(optimizer, type(None)):
#             raise KeyError("Opimizer not found in %s. Please include one or set compiled = False."%ini_path)
#         #get the metrics
#         try:
#             strMetrics = config['model']['metrics']
#         except KeyError:
#             strMetrics = 'accuracy'
#         metrics = metrics_from_string(strMetrics)
#         if strAPI == 'functional':
#             kerasModel = keras.models.Model(inputs=layerFunctionalInput, outputs=layerFunctionalOutput)
#         kerasModel.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=metrics)
#         return kerasModel
#     #return non-compiled model
#     else:
#         if strAPI == 'functional':
#             return [layerFunctionalInput, layerFunctionalOutput]
#         else:
#             return kerasModel


################################################################################
def fAddLayerFromConfig(config, sLayer, aInputShape):
    """
    This will add a lay to BrainNetCNN's architexture list. This is used to build/edit models from an ini file
    :param config: the config that has the ini for the layer
    :param sLayer: the layer name
    :param aInputShape: the shape of the original input
    :return: a list in the format required for incorporating into a BrainNet architexture
    """
    ini_dict = dict(config[sLayer])
    format_dict(ini_dict)

    sClass=ini_dict['class']

    if sClass=='e2e':
        lsLayer = [sClass, {'n_filters': ini_dict['n_filters'],
                            'kernel_h': aInputShape[0],
                            'kernel_w': aInputShape[1]
                            }]

    elif sClass=='dropout':
        lsLayer = [sClass, {'dropout_ratio': ini_dict['dropout_ratio']}]

    elif sClass=='activation':
        lsLayer = [ini_dict['activation'], {'negative_slope': ini_dict['negative_slope']}]

    elif sClass=='e2n':
        lsLayer = [sClass, {'n_filters': ini_dict['n_filters'],
                            'kernel_h': aInputShape[0],
                            'kernel_w': aInputShape[1]
                            }]

    elif sClass=='fc':
        lsLayer = [sClass, {'n_filters': ini_dict['n_filters']}]

    elif sClass=='out':
        lsLayer=[sClass, {'n_filters': ini_dict['n_filters']}]

    else:
        if sClass is not None:
            raise ValueError('The Layer Name is specified as: ' + str(sClass) + ' The only allowed '
                             'layer names are: e2e, dropout, activation, e2n, fc, and out')
        elif sClass is None:
            raise ValueError('The Layer Name is NOT specified. (name is None) The only allowed '
                             'layer names are: e2e, dropout, activation, e2n, fc, and out')
    return lsLayer


def fModelArchFromIni(sIniPath, sIni, aInputShape, sSavePath, iCrossVal):
    """This is designed to take a ini file and create a Brain Net Network from it
    Specific format is needed, the format can be seen from outputs of the functions that create the ini files
    :param ini_path: The location of the ini file
    :param aInputShape: The shape (HxW) of the graph over which the network will scan
    :return: caffe architextrue (list of dictionaries)
    """
    # First, we load the .ini file as 'config'
    config = read_config_file(sIniPath)

    # Then, we initialize the network architexture list
    lsArch = list()

    #make sure the first layer is an input
    lsLayers = [s for s in config.sections() if s.startswith('layer/')]

    # add the layers
    for iModel, sLayer in enumerate(lsLayers):
        lsArch.extend([fAddLayerFromConfig(config, sLayer, aInputShape)])

    BrainNet=BrainNetCNN(sIni+str(iCrossVal), lsArch, hardware='gpu', dir_data=sSavePath)

    return BrainNet


################################################################################
def fReshapeUpperTriangVectToSquare(aXVect):
    # Reshape the vectorized upper triangular matrix into a square matrix

    # First, get the dimensions of the square
    iDim = int((-1 + np.sqrt(1 + 8 * aXVect.shape[0])) / 2)

    # Next, initialize variables to loop through vector and save results
    iVectLocation = 0
    aXSquare=np.zeros((iDim, iDim))

    # From the first row, take terms 1 to the end, from the second, 2 to the end, etc
    for iRow in range(iDim):
        iRowLen = iDim - iRow
        aXSquare[iRow, iRow:] = aXVect[iVectLocation:(iVectLocation+iRowLen)]
        iVectLocation = iVectLocation + iRowLen

    #Now mirror the other direction
    for iRow in range(iDim):
        for iColumn in range(iDim):
            aXSquare[iColumn, iRow] = aXSquare[iRow, iColumn]


    return aXSquare

def fReshapeInputData(aXData):
    # Reshapes the input data from an array with many upper triangular vectors
    # (N X L) to an array of shape N X H X W, N is the number of samples, L is
    # the length of the upper-triagular matrix taht has been vecotrized and, H
    # and W are the spatial dimensions for each sample.
    iXSamples=aXData.shape[0]
    iXVectLen=aXData.shape[1]
    iSquareDim=int((-1 + np.sqrt(1 + 8 * iXVectLen)) / 2)

    aXNew = np.zeros((iXSamples, iSquareDim, iSquareDim))
    for iSample in range(iXSamples):
        aXNew[iSample, :, :] = fReshapeUpperTriangVectToSquare(aXData[iSample, :])

    return aXNew

def fRunBrainNetCNNOnInput(sInputName, iModelNum, sSubInputName='', iEpochs=1, bEarlyStopping=True):
    sIni = 'BrainNetCNN_' + str(iModelNum)
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/BrainNetCNN'
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

    [dXData, dXTest, aYData, aYTest] = pickle.load(open(sDataPath, 'rb'))

    if sInputName =='anatomy':
        aXData = fReshapeInputData(dXData[sInputName])
        aXTest = fReshapeInputData(dXTest[sInputName])
    else:
        aXData = fReshapeInputData(dXData[sInputName][sSubInputName])
        aXTest = fReshapeInputData(dXTest[sInputName][sSubInputName])


    # The required dimensions for the BrainNetCNN network is size
    # N x C X H x W, where N is the number of samples, C is
    # the number of channels in each sample, and, H and W are the
    # spatial dimensions for each sample. So, we expand the Channels
    # Dimension to 1
    aXData = np.expand_dims(aXData, axis=1)

    aXTest = np.expand_dims(aXTest, axis=1)

    aXData = np.float32(aXData)
    aXTest = np.float32(aXTest)
    aYData = np.float32(aYData)
    aYTest = np.float32(aYTest)

    aDataShape = [aXData.shape[2], aXData.shape[3]]

    iSplitSize = int(aXData.shape[0]/3)

    # Split the Data for 3x cross validation
    lsXDataSplit = [[aXData[iSplitSize:,:,:,:]], # skip over beginning
                    [np.append(aXData[:iSplitSize,:], aXData[2*iSplitSize:,:], axis=0)], #split over middle
                    [aXData[:2*iSplitSize,:,:,:]] # skip over end
                    ]
    lsYDataSplit = [[aYData[iSplitSize:,:]], # skip over beginning
                    [np.append(aYData[:iSplitSize,:], aYData[2*iSplitSize:,:], axis=0)], #split over middle
                    [aYData[:2*iSplitSize,:]] # skip over end
                    ]
    lsXVal =[[aXData[:iSplitSize,:,:,:]], # include only beginning
             [aXData[iSplitSize:2*iSplitSize,:,:,:]], # include only middle
             [aXData[2*iSplitSize:,:,:,:]] # include only end
             ]
    lsYVal =[[aYData[:iSplitSize,:]], # include only beginning
             [aYData[iSplitSize:2*iSplitSize,:]], # include only middle
             [aYData[2*iSplitSize:,:]] # include only end
             ]

    for iCrossVal in range(3):
        BrainNetCNNModel = fModelArchFromIni(sIniPath, sIni, aDataShape, sSavePath, (iCrossVal+1))

        BrainNetCNNModel.fit(lsXDataSplit[iCrossVal][0], lsYDataSplit[iCrossVal][0], lsXVal[iCrossVal][0], lsYVal[iCrossVal][0])
        aPredicted=BrainNetCNNModel.predict(aXTest)

        pickle.dump(aPredicted, open(os.path.join(sSavePath, sIni) + sInputName + sSubInputName +
                                                                    'PredictedResultsCrossVal'+str(iCrossVal+1)+'.p', 'wb'))




if '__main__' == __name__:
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
    [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))

    iModel = sys.argv[1]
    iModel = iModel.split('_')[1]
    iModel = iModel.split('.')[0]

    sInputName='connectivity'

    for sAtlas in dXData[sInputName]:
        fRunBrainNetCNNOnInput(sInputName, iModel, sSubInputName=sAtlas)


















# def fReproduceModel(sInputName, iModelNum, sWeightsPath, sSubInputName=''):
#
#
#     sIni = 'BrainNetCNN_' + str(iModelNum)
#     sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
#     sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels'
#     sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
#
#     [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))
#
#     if sInputName =='anatomy':
#         aXData = dXData[sInputName]
#         aXTest = dXTest[sInputName]
#     else:
#         aXData = dXData[sInputName][sSubInputName]
#         aXTest = dXTest[sInputName][sSubInputName]
#
#     # The required dimensions for the BrainNetCNN network is size
#     # N x H x W x C, where N is the number of samples, C is
#     # the number of channels in each sample, and, H and W are the
#     # spatial dimensions for each sample.
#     aXData = np.expand_dims(aXData, axis=1)
#     aXData = np.expand_dims(aXData, axis=3)
#
#     aXTest = np.expand_dims(aXTest, axis=1)
#     aXTest = np.expand_dims(aXTest, axis=3)
#
#     aDataShape=[aSData.shape[1], aXData.shape[2]]
#
#     caffeModelArch=fModelArchFromIni(sIniPath, aInputShape=aDataShape)
#
#     kmModel.load_weights(sWeightsPath)
#
#     return kmModel
#
#
# if '__main__' == __name__:
#     sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
#     [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))
#
#     iModel=sys.argv[1]
#     iModel=iModel.split('_')[1]
#     iModel=iModel.split('.')[0]
#
#     fRunBrainNetCNNOnInput('anatomy', iModel, iEpochs=500)
#     for keys in dXData['connectivity']:
#        fRunBrainNetCNNOnInput('connectivity', iModel, sSubInputName=keys, iEpochs=500)
#     for keys in dXData['combined']:
#         fRunBrainNetCNNOnInput('combined', iModel, sSubInputName=keys, iEpochs=500)
#
#
#
#
#
#
# ###########################################################################################################
#
#     sTargetDirectory = fInitialize(sIni)
#
#
#     # Here we expand the dimensions for the neural net to work properly
#     # the added 1st dimension is the number of channels from the
#     # patient, in this case, all measurements are of one connectivity
#     # 'channel', not multiple as RGB images would be.
#
#     # The required dimensions for BrainNetCNN is size
#     # N x C x H x W, where N is the number of samples, C is
#     # the number of channels in each sample, and, H and W are the
#     # spatial dimensions for each sample.
#     xData = np.expand_dims(xData, axis=1)
#     xTest = np.expand_dims(xTest, axis=1)
#     xVal = np.expand_dims(xVal, axis=1)
#
#     # initializing the architexture
#     BrainNetArch = [ # load from ini file
#         ['e2n', {'n_filters': 16,
#                  'kernel_h': xData.shape[2],
#                  'kernel_w': xData.shape[3]}],
#         ['dropout', {'dropout_ratio': 0.5}],
#         ['relu', {'negative_slope': 0.33}],
#         ['fc', {'n_filters': 30}],
#         ['relu', {'negative_slope': 0.33}],
#         ['out', {'n_filters': 1}]
#     ]
#
#     BrainNetFullNetwork = BrainNetCNN(sIni, BrainNetArch, hardware='gpu', dir_data=sTargetDirectory)
#     BrainNetFullNetwork.fit(xData, yData, xVal, yVal)
#     yPredicted = BrainNetFullNetwork.predict(xTest)
#
#     # Now we print out the performance by several metrics
#     BrainNetROCAUCScore = roc_auc_score(yTest, yPredicted)
#     precisions, recalls, thresholds = precision_recall_curve(yTest, yPredicted)
#     BrainNetPRAUCScore = auc(recalls, precisions)
#     BrainNetF1Score = f1_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))))
#     BrainNetAccuracyScore = accuracy_score(yTest, np.rint((yPredicted-min(yPredicted))/(max(yPredicted)-min(yPredicted))), normalize=True)
#
#     #Then we save the performace metrics
#     pickle.dump(BrainNetFullNetwork, open(sTargetDirectory + sIni + "FullNetwork.p", 'wb'))
#
#     pdResults['BrainNetCNN'] = [BrainNetAccuracyScore, BrainNetPRAUCScore, BrainNetF1Score, BrainNetROCAUCScore]
#     pdResults.topickle