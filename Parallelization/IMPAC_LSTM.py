""" This code takes the Paris IMPAC Autism study data,and
trains an LSTM network as a classifier

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
import re
import sys
import ast
import keras
import configparser
from keras import regularizers
import pandas as pd
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, auc, f1_score
import pickle


###############################################################################
def read_config_file(path):
    if not os.path.exists(path):
        raise IOError('ini_loc: %r does not exisit' % path)
    # load the ini file
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


def metrics_from_string(metrics):
    """
    Take a list of stings or a string of a list returns a list of the the metric(s) so they can be passed into the
        compile method in a keras model.
    Some metrics can be passed to keras as a string, these metrics are found in listStrMetrics, if one is missing please
        add it.
    :param metric_string:
    :return: a list of metrics for keras.
    """
    listStrMetrics = ['acc', 'accuracy', 'mse', 'mae', 'mape', 'cosine']
    strMetrics = str(metrics)
    listMetricStrings = re.findall(r'[A-Za-z_]+', strMetrics)
    listMetrics = []
    for strMetric in listMetricStrings:
        if strMetric in listStrMetrics:
            kerasMetric = strMetric
        else:
            try:
                kerasMetric = getattr(keras.metrics, strMetric)
            except:
                raise ValueError("%s is not a valid keras metric" % strMetrics)
        listMetrics.append(kerasMetric)
    return listMetrics


def get_optimizer_from_config(config):
    if 'optimizer' in config.keys():
        # if there's an optimizer
        optimizer_dct = dict(config['optimizer'])
        format_dict(optimizer_dct)
        optimizer_class = getattr(keras.optimizers, optimizer_dct['class'])
        optimizer_dct.pop('class')
        optimizer = optimizer_class(**optimizer_dct)
        return optimizer
    else:
        return None


def add_layer_from_config(model, config, config_key, aInputShape=None, api='functional'):
    """ By Alex Treacher
    This will add a layer to a model from a config file. This is used to build/edit models from an ini file
    :param model: the keras model to add the layer to
    :param config: the config that has the ini for the layer
    :param config_key: the key for the config
    :param api: which keras API (functional or sequential) is being used to build the model
    :return: Sequential API: the model with the layer added
        Functional APU: the last layer that was added
    """
    ini_dict = dict(config[config_key])
    format_dict(ini_dict)
    class_to_add = getattr(keras.layers, ini_dict['class'])

    sActivation = None
    flAlpha = None

    if 'activation' in ini_dict.keys():
        if ini_dict['activation'] == 'relu':
            sActivation = ini_dict['activation']
            flAlpha = ini_dict['alpha']
            ini_dict.pop('activation')
            ini_dict.pop('alpha')

    # Pull out the regularization magnitude
    if 'regularizer' in ini_dict.keys():
        flL2Alpha = ini_dict['regularizer']
        ini_dict.pop('regularizer')
        ini_dict['kernel_regularizer'] = regularizers.l2(flL2Alpha)

    ini_dict.pop('class')
    if aInputShape is not None:
        ini_dict['input_shape'] = aInputShape
    if api.lower() == 'sequential':  # sequential return the model with added layer
        model.add(class_to_add(**ini_dict))
        if sActivation is not None:
            if sActivation == 'relu':
                model.add(keras.layers.advanced_activations.LeakyReLU(alpha=flAlpha))
        return model
    if api.lower() == 'functional':  # functional model return the new layer
        output_layer = class_to_add(**ini_dict)(model)
        return output_layer


def network_from_ini_2(ini_path, aInputShape=None, compiled=True):
    """ By Alex Treacher
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

    # make sure the first layer is an input
    layers = [s for s in config.sections() if s.startswith('layer/')]
    strAPI = config['model']['class'].lower()

    # Initiate the model with the input layer
    if strAPI == 'sequential':
        kerasModel = keras.models.Sequential()
        # add the input layer
        add_layer_from_config(kerasModel, config, layers[0], aInputShape, api=strAPI)

    if strAPI == 'functional':
        input_config = dict(config[layers[0]])
        input_class = getattr(keras.layers, input_config['class'])
        input_config.pop('class')
        format_dict(input_config)
        kerasModel = input_class(**input_config)
        layerFunctionalInput = kerasModel
    # remove the input layer so as to not try to add it again later
    layers = layers[1:]

    # add the layers
    for intI, strLayer in enumerate(layers):
        if intI + 1 == layers.__len__():  # last layer
            kerasModel.add(keras.layers.Flatten())
        kerasModel = add_layer_from_config(kerasModel, config, strLayer, api=strAPI)
        if intI + 1 == layers.__len__():  # last layer
            layerFunctionalOutput = kerasModel

    # compile if compiled==True.
    if compiled == True:
        # get the optimzier
        optimizer = get_optimizer_from_config(config)
        if isinstance(optimizer, type(None)):
            raise KeyError("Opimizer not found in %s. Please include one or set compiled = False." % ini_path)
        # get the metrics
        try:
            strMetrics = config['model']['metrics']
        except KeyError:
            strMetrics = 'accuracy'
        metrics = metrics_from_string(strMetrics)
        if strAPI == 'functional':
            kerasModel = keras.models.Model(inputs=layerFunctionalInput, outputs=layerFunctionalOutput)
        kerasModel.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=metrics)
        return kerasModel
    # return non-compiled model
    else:
        if strAPI == 'functional':
            return [layerFunctionalInput, layerFunctionalOutput]
        else:
            return kerasModel


################################################################################

def fRunLSTMNetOnInput(sInputName, iModelNum, sSubInputName='', iEpochs=1, bEarlyStopping=True):
    sIni = 'LSTM_' + str(iModelNum)
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels'
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

    [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))

    if sInputName == 'anatomy':
        aXData = dXData[sInputName]
        aXTest = dXTest[sInputName]
    else:
        aXData = dXData[sInputName][sSubInputName]
        aXTest = dXTest[sInputName][sSubInputName]

    # The required dimensions for the LSTM network is size
    # N x I x D, where N is the number of samples,
    # I is the number of inputs and D is the
    # spatial dimensions for each sample.
    aXData = np.expand_dims(aXData, axis=1)

    aXTest = np.expand_dims(aXTest, axis=1)

    aDataShape = [aXData.shape[1], aXData.shape[2]]

    kmModel = network_from_ini_2(sIniPath, aInputShape=aDataShape)

    if bEarlyStopping == True:
        kerStopping = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.01, patience=10,
                                                    restore_best_weights=True)
        kmModel.fit(aXData, aYData, epochs=iEpochs, callbacks=[kerStopping])
    else:
        kmModel.fit(aXData, aYData, epochs=iEpochs)
    aPredicted = kmModel.predict(aXTest)

    kmModel.save_weights(sSavePath + '/' + sIni + '_' + sInputName + sSubInputName + '.h5')
    # kmModel.save(sSavePath + '/' + sIni + sInputName + sSubInputName + 'TestEarlyStop.h5', include_optimizer=False, overwrite=True)
    pickle.dump(aPredicted,
                open(os.path.join(sSavePath, sIni) + sInputName + sSubInputName + 'PredictedResults.p', 'wb'))


def fReproduceModel(sInputName, iModelNum, sWeightsPath, sSubInputName=''):
    sIni = 'Dense_' + str(iModelNum)
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels'
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'

    [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))

    if sInputName == 'anatomy':
        aXData = dXData[sInputName]
        aXTest = dXTest[sInputName]
    else:
        aXData = dXData[sInputName][sSubInputName]
        aXTest = dXTest[sInputName][sSubInputName]

    # The required dimensions for the LSTM network is size
    # N x I x D, where N is the number of samples,
    # I is the number of inputs and D is the
    # spatial dimensions for each sample.
    aXData = np.expand_dims(aXData, axis=2)

    aXTest = np.expand_dims(aXTest, axis=2)

    iDataShape = aXData[0, :].shape[2]

    kmModel = network_from_ini_2(sIniPath, aInputShape=iDataShape)

    kmModel.load_weights(sWeightsPath)

    return kmModel


if '__main__' == __name__:
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestData.p'
    [dXData, dXTest, aYData, aYtest] = pickle.load(open(sDataPath, 'rb'))

    iModel = sys.argv[1]
    iModel = iModel.split('_')[1]
    iModel = iModel.split('.')[0]

    #fRunLSTMNetOnInput('anatomy', 0, iEpochs=500)

    fRunLSTMNetOnInput('anatomy', iModel, iEpochs=500)
    for keys in dXData['connectivity']:
        fRunLSTMNetOnInput('connectivity', iModel, sSubInputName=keys, iEpochs=500)
    for keys in dXData['combined']:
        fRunLSTMNetOnInput('combined', iModel, sSubInputName=keys, iEpochs=500)
