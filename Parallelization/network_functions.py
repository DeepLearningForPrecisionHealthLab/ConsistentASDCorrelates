"""
Author: Alex Treacher

This file contains functions that allow for the reading and creation of network architecture ini files that can then
later be loaded as a model in keras.
"""

import os, sys
sys.path.extend('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization')
import keras
import ast
import configparser
import pickle
import numpy as np
import pandas as pd
import random
import re

def read_config_file(path):
    if not os.path.exists(path):
        raise IOError('ini_loc: %r does not exisit'%path)
    #load the ini file
    config = configparser.ConfigParser()
    config.read(path)
    return config

def add_layer_from_config(model, config, config_key, api='functional'):
    """
    This will add a layer to a model from a config file. This is used to build/edit models from an ini file
    :param model: the keras model to add the layer to
    :param config: the config that has the ini for the layer
    :param config_key: the key for the config
    :param api: which keras API (functional or sequential) is being used to build the model
    :return: Sequential API: the model with the layer added
        Functional APU: the last layer that was added
    """
    ini_dct = dict(config[config_key])
    format_dct(ini_dct)
    class_to_add = getattr(keras.layers, ini_dct['class'])
    ini_dct.pop('class')
    if api.lower() == 'sequential': #sequential return the model with added layer
        model.add(class_to_add(**ini_dct))
        return model
    if api.lower() == 'functional':  #functional model return the new layer
        output_layer = class_to_add(**ini_dct)(model)
        return output_layer

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
    listMetricStrings = re.findall(r'[A-Za-z_]+',strMetrics)
    listMetrics = []
    for strMetric in listMetricStrings:
        if strMetric in listStrMetrics:
            kerasMetric = strMetric
        else:
            try:
                kerasMetric = getattr(keras.metrics, strMetric)
            except:
                raise ValueError("%s is not a valid keras metric"%strMetrics)
        listMetrics.append(kerasMetric)
    return listMetrics

def get_optimizer_from_config(config):
    if 'optimizer' in config.keys():
        #if there's an optimizer
        optimizer_dct = dict(config['optimizer'])
        format_dct(optimizer_dct)
        optimizer_class = getattr(keras.optimizers, optimizer_dct['class'])
        optimizer_dct.pop('class')
        optimizer = optimizer_class(**optimizer_dct)
        return optimizer
    else:
        return None

def cnn_random_binary_classifier_builder(output_path):
    """
    Makes semi random config file for a keras network, designed around the liber fibrosis project.
    Creates an 2 class network architecture with a hyperparameter set in the space defined by the '#options for randon selection'
    section in the function
    See also cnn_random_n_classifier_builder. (these two functions could easily be combined)
    :param output_path: where to save the file
    :return: None
    """
    #options for randon selection
    conv_layers_options = list(range(1,6,1))
    conv_filter_options = list(range(16,128,1))
    conv_kernal_size_options = [(2,2),(3,3),(4,4)]

    dense_layers_options = list(range(1,6,1))
    dense_unit_options = list(range(5,256,1))

    #set the values
    conv_no_layers = random.choice(conv_layers_options)
    dense_no_layers = random.choice(dense_layers_options)

    config = configparser.ConfigParser()
    config['model'] = {'class':'Sequential','loss': 'categorical_crossentropy'}
    config['layer/input'] = {'class': 'InputLayer', 'input_shape': '[56,28,1]'}
    for i in range(conv_no_layers):
        config['layer/conv%i'%(i)] = {'class':'Conv2D',
                                      'filters':random.choice(conv_filter_options),
                                      'kernel_size':random.choice(conv_kernal_size_options),
                                      'padding':'valid',
                                      'kernel_initializer':'he_normal'}
        config['layer/conv%i_activation'%(i)] = {'class': 'PReLU'}
    config['layer/flatten'] = {'class': 'Flatten'}
    for i in range(dense_no_layers-1):
        config['layer/dense%i'%(i)] = {'class':'Dense',
                                       'units':random.choice(dense_unit_options),
                                       'kernel_initializer':'he_normal'}
        config['layer/dense%i_activation' % (i)] = {'class': 'PReLU'}
    config['layer/dense%i' % (i+1)] = {'class': 'Dense',
                                     'units': 2,
                                     'kernel_initializer': 'he_normal',
                                     'activation': 'softmax'}
    config['optimizer'] = {'class': 'SGD', 'lr': .01}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def cnn_random_n_classifier_builder(output_path, input, n=3):
    """
    Makes semi random config file for a keras network, designed around the liber fibrosis project.
    Creates an n class network architecture with a hyperparameter set in the space defined by the '#options for randon selection'
    section in the function
    :param output_path: where to save the file
    :return: None
    """
    #options for randon selection
    conv_layers_options = list(range(1,6,1))
    conv_filter_options = list(range(16,128,1))
    conv_kernal_size_options = [(2,2),(3,3),(4,4)]

    dense_layers_options = list(range(1,6,1))
    dense_unit_options = list(range(5,256,1))

    #set the values
    conv_no_layers = random.choice(conv_layers_options)
    dense_no_layers = random.choice(dense_layers_options)

    config = configparser.ConfigParser()
    config['model'] = {'class':'Sequential','loss': 'categorical_crossentropy'}
    config['layer/input'] = {'class': 'InputLayer', 'input_shape': str(input)}
    for i in range(conv_no_layers):
        config['layer/conv%i'%(i)] = {'class':'Conv2D',
                                      'filters':random.choice(conv_filter_options),
                                      'kernel_size':random.choice(conv_kernal_size_options),
                                      'padding':'valid',
                                      'kernel_initializer':'he_normal'}
        config['layer/conv%i_activation'%(i)] = {'class': 'PReLU'}

    config['layer/flatten'] = {'class': 'Flatten'}

    for i in range(dense_no_layers-1):
        config['layer/dense%i'%(i)] = {'class':'Dense',
                                       'units':random.choice(dense_unit_options),
                                       'kernel_initializer':'he_normal'}
        config['layer/dense%i_activation' % (i)] = {'class': 'PReLU'}
    config['layer/dense%i' % (i+1)] = {'class': 'Dense',
                                     'units': n,
                                     'kernel_initializer': 'he_normal',
                                     'activation': 'softmax'}
    config['optimizer'] = {'class': 'Adam',
                            'beta_1':.7,
                            'beta_2':.9999,
                            'epsilon':10**-8}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def cnn_random_regressor_ini(output_path):
    """
    Makes semi random config file for a keras network, designed around the liber fibrosis project, but could easilly be
    used for other projects
    Creates regressor network architecture with a hyperparameter set in the space defined by the '#options for randon selection'
    section in the function
    :param output_path: where to save the file
    :return: None
    """
    #options for randon selection
    conv_layers_options = list(range(1,6,1))
    conv_filter_options = list(range(16,128,1))
    conv_kernal_size_options = [(2,2),(3,3),(4,4)]

    dense_layers_options = list(range(1,6,1))
    dense_unit_options = list(range(5,256,1))

    #set the values
    conv_no_layers = random.choice(conv_layers_options)
    dense_no_layers = random.choice(dense_layers_options)

    config = configparser.ConfigParser()
    config['model'] = {'class':'Sequential','loss': 'mean_squared_error'}
    config['layer/input'] = {'class': 'InputLayer', 'input_shape': '[91, 233,1]'}
    for i in range(conv_no_layers):
        config['layer/conv%i'%(i)] = {'class':'Conv2D',
                                      'filters':random.choice(conv_filter_options),
                                      'kernel_size':random.choice(conv_kernal_size_options),
                                      'padding':'valid',
                                      'kernel_initializer':'he_normal'}
        config['layer/conv%i_activation'%(i)] = {'class': 'PReLU'}
    config['layer/flatten'] = {'class': 'Flatten'}
    for i in range(dense_no_layers-1):
        config['layer/dense%i'%(i)] = {'class':'Dense',
                                       'units':random.choice(dense_unit_options),
                                       'kernel_initializer':'he_normal'}
        config['layer/dense%i_activation' % (i)] = {'class': 'PReLU'}
    config['layer/dense%i' % (i+1)] = {'class': 'Dense',
                                     'units': 1,
                                     'kernel_initializer': 'he_normal',
                                    }
    config['layer/dense%i_activation' % (i+1)] = {'class': 'PReLU'}
    config['optimizer'] = {'class': 'Adam',
                            'beta_1':.7,
                            'beta_2':.9999,
                            'epsilon':10**-8}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def MEG_spatial_test(output_path):
    """
    This was made as a positive control to test the MEG code
    This is the spatial section of model 312 from Prabhats code.
    :param output_path: where to save the file
    :return: None
    """
    config = configparser.ConfigParser()
    config['model'] = {'class':'Functional','loss': 'mean_squared_error'}
    config['layer/input'] = {'class': 'Input', 'shape': '[100,100,3]'}
    config['layer/bn0'] = {'class': 'BatchNormalization', 'axis': -1}
    #1st block
    config['layer/conv2d1'] = {'class':'Conv2D', 'filters':16, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn1'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu1'] = {'class':'Activation', 'activation':'relu'}
    config['layer/conv2d2'] = {'class':'Conv2D', 'filters':16, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn2'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu2'] = {'class':'Activation', 'activation':'relu'}
    config['layer/maxpool1'] = {'class':'MaxPooling2D', 'pool_size':(2,2)}

    #2nd block
    config['layer/conv2d3'] = {'class':'Conv2D', 'filters':32, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn3'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu3'] = {'class':'Activation', 'activation':'relu'}
    config['layer/conv2d4'] = {'class':'Conv2D', 'filters':32, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn4'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu4'] = {'class':'Activation', 'activation':'relu'}

    config['layer/maxpool2'] = {'class':'MaxPooling2D', 'pool_size':(2,2)}

    #3rd block
    config['layer/conv2d5'] = {'class':'Conv2D', 'filters':64, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn5'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu5'] = {'class':'Activation', 'activation':'relu'}
    config['layer/conv2d6'] = {'class':'Conv2D', 'filters':64, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn6'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu6'] = {'class':'Activation', 'activation':'relu'}
    config['layer/conv2d7'] = {'class':'Conv2D', 'filters':64, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn7'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu7'] = {'class':'Activation', 'activation':'relu'}
    config['layer/conv2d8'] = {'class':'Conv2D', 'filters':64, 'kernel_size':(9,9), 'border_mode':'same'}
    config['layer/bn8'] = {'class':'BatchNormalization', 'axis':-1}
    config['layer/relu8'] = {'class':'Activation', 'activation':'relu'}
    config['layer/maxpool3'] = {'class':'MaxPooling2D', 'pool_size':(2,2)}
    config['layer/flatten'] = {'class': 'Flatten'}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def MEG_spatial_ini(output_path, input_shape=[100,100,3]):
    """
    Creates a randomized network ini for the MEG project for the spatial section of the network.
    The random hyper parameters space that is sampled from can be seen in the function.
    :param output_path: Where to save the ini
    :return: None
    """
    # the hpyer parameter space to select from
    number_of_layers_options = range(4,15)
    number_of_filters_options = range(8,129)
    kernal_size_options = range(3,16)
    batch_norm = [True, False]
    max_pool = [True, False]    #set up the config and input layer

    #set up the config and add input layer
    config = configparser.ConfigParser()
    config['model'] = {'class': 'Functional'}
    config['layer/spatial_input'] = {'class': 'Input', 'shape': str(input_shape)}

    #make random network
    number_of_layers = random.choice(number_of_layers_options)
    for i in range(number_of_layers):
        kernal_size = random.choice(kernal_size_options)
        config['layer/spatial_conv2d%i'%i] = {'class':'Conv2D',
                                      'filters':random.choice(number_of_filters_options),
                                      'kernel_size':[kernal_size, kernal_size],
                                      'kernel_initializer':'he_normal',
                                      'border_mode':'same'}
        if random.choice(batch_norm):
            config['layer/spatial_batch_norm%i'%i] = {'class': 'BatchNormalization', 'axis': -1}
        config['layer/spatial_prelu%i'%i] = {'class': 'PReLU'}
        if random.choice(max_pool):
            config['layer/spatial_MaxPooling%i'%i] = {'class': 'MaxPooling2D'}
    config['layer/flatten'] = {'class': 'Flatten'}
    with open(output_path, 'w') as configfile:
        config.write(configfile)


def MEG_temporal_test(output_path):
    """
    This was made as a positive control to test the MEG code
    This is the temporal section of model 312 from Prabhats code.
    :param output_path: where to save the file
    :return: None
    """
    config = configparser.ConfigParser()
    config['model'] = {'class':'Functional','loss': 'mean_squared_error'}
    config['layer/input'] = {'class': 'Input', 'shape': '[10000,1]'}
    config['layer/conv2d1'] = {'class':'Conv1D', 'filters':64, 'kernel_size':(13), 'subsample_length':1, 'activation':'relu'}
    config['layer/MaxPooling1'] = {'class':'MaxPooling1D'}
    config['layer/conv2d2'] = {'class':'Conv1D', 'filters':64, 'kernel_size':(13), 'subsample_length':1,  'activation':'relu'}
    config['layer/MaxPooling2'] = {'class': 'MaxPooling1D'}
    config['layer/conv2d3'] = {'class':'Conv1D', 'filters':64, 'kernel_size':(13), 'subsample_length':1, 'activation':'relu'}
    config['layer/MaxPooling3'] = {'class': 'MaxPooling1D'}
    config['layer/conv2d4'] = {'class':'Conv1D', 'filters':64, 'kernel_size':(13), 'subsample_length':1, 'activation':'relu'}
    config['layer/MaxPooling4'] = {'class': 'MaxPooling1D'}
    config['layer/conv2d5'] = {'class':'Conv1D', 'filters':64, 'kernel_size':(13), 'subsample_length':1, 'activation':'relu'}
    config['layer/MaxPooling5'] = {'class': 'MaxPooling1D'}
    config['layer/flatten'] = {'class': 'Flatten'}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def MEG_temporal_ini(output_path, input_shape=[10000,1]):
    """
    Creates a randomized network ini for the MEG project for the temporal section of the network.
    The random hyper parameters space that is sampled from can be seen in the function.
    :param output_path: Where to save the ini
    :return: None
    """
    # the hpyer parameter space to select from
    number_of_layers_options = range(2,9)
    kernal_size_options = range(4,19)
    number_of_filters_options = range(8,64)
    max_pool = [True, False]

    #set up the config and input layer
    config = configparser.ConfigParser()
    config['model'] = {'class': 'Functional'}
    config['layer/temporal_input'] = {'class': 'Input', 'shape': str(input_shape)}
    # make the other layers
    number_of_layers = random.choice(number_of_layers_options)
    for i in range(number_of_layers):
        config['layer/temporal_conv2d%i'%i] = {'class':'Conv1D',
                                      'filters':random.choice(number_of_filters_options),
                                      'kernel_size':random.choice(kernal_size_options),
                                      'kernel_initializer':'he_normal',
                                      'subsample_length':1}
        config['layer/temporal_prelu%i'%i] = {'class': 'PReLU'}
        if random.choice(max_pool):
            config['layer/temporal_MaxPooling%i'%i] = {'class': 'MaxPooling1D'}
    config['layer/temporal_flatten'] = {'class': 'Flatten'}
    #save the file
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def MEG_dense_test(output_path):
    """
    This was made as a positive control to test the MEG code
    This is the dense section of model 312 from Prabhats code.
    :param output_path: where to save the file
    :return: None
    """
    config = configparser.ConfigParser()
    config['model'] = {'class':'Functional','loss':'categorical_crossentropy'}
    config['layer/merge'] = {'class':'merge','mode':'concat','concat_axis':'-1'}
    config['layer/BatchNomalization'] = {'class':'BatchNormalization'}
    config['layer/Dense1'] = {'class':'Dense','units':256,'activation':'relu','kernel_initializer':'he_normal'}
    config['layer/Dense2'] = {'class':'Dense','units':128,'activation':'relu','kernel_initializer':'he_normal'}
    config['layer/Dense3'] = {'class':'Dense','units':64,'activation':'relu','kernel_initializer':'he_normal'}
    config['layer/Dropout'] = {'class':'Dropout','rate':'0.5'}
    config['layer/Dense'] = {'class':'Dense','units':3,'activation':'softmax'}
    config['optimizer'] = {'class':'Adam','lr':'.0001'}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def MEG_dense_ini(output_path):
    """
    Creates a randomized network ini for the MEG project for the dense (aka merge) section of the network.
    The random hyper parameters space that is sampled from can be seen in the function.
    :param output_path: Where to save the ini
    :return: None
    """
    # the hpyer parameter space to select from
    number_of_layers_options = range(2,6)
    number_of_neurons_options = range(4,256)
    batch_norm_input_option = [True, False]
    dropout_option = [True, False]
    config = configparser.ConfigParser()
    config['model'] = {'class':'Functional','loss':'categorical_crossentropy'}
    config['layer/merge'] = {'class':'merge','mode':'concat','concat_axis':'-1'}
    if random.choice(batch_norm_input_option):
        config['layer/merge_BatchNomalization'] = {'class': 'BatchNormalization'}
    number_of_layers = random.choice(number_of_layers_options)
    for i in range(number_of_layers-1):
        config['layer/merge_Dense%i'%i] = {'class': 'Dense',
                                           'units': random.choice(number_of_neurons_options),
                                           'kernel_initializer': 'he_normal',
                                     }
        config['layer/merge_prelu%i'%i] = {'class': 'PReLU'}
    config['layer/merge_Dense_output'] = {'class':'Dense','units':3,'activation':'softmax'}
    config['optimizer'] = {'class': 'Adam', 'lr': '.0001'}
    with open(output_path, 'w') as configfile:
        config.write(configfile)

def make_MEG_ini_folder(output_path):
    """
    Make a single random network for the MEG project using the functions in this script
    :param output_path: Where to save the output
    :return: None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    MEG_temporal_ini(os.path.join(output_path,'temporal.ini'))
    MEG_spatial_ini(os.path.join(output_path,'spatial.ini'))
    MEG_dense_ini(os.path.join(output_path,'merge.ini'))

def network_from_ini(ini_loc, api='functional', compiled=True):
    """ By Alex Treacher
    This is designed to take a ini file and create a neural network from it
    Specific format is needed, the format can be seen from outputs of the above functions that create the ini files
    :param ini_loc: The location of the ini file
    :param api: The api that the model will be made from. Currently supports functional and sequential
    :return: nn based on the ini, if api=='functional' and compiled=Flase this returns [input_layer, output_layer] so the network can be munipulated or compiled
    """
    config = read_config_file(ini_loc)
    layers = [s for s in config.sections() if s.startswith('layer/')]
    #make sure the first layer is an input
    if not (config[layers[0]]['class'] == "InputLayer" or config[layers[0]]['class'] == "Input"):
        raise ValueError('First layer of %s should be an input layer of class Input')
    #Initiate the model
    if api.lower()=='sequential':
        nn = getattr(keras.models, config['model']['class'])()
    if api.lower()=='functional':
        input_config = dict(config[layers[0]])
        input_class = getattr(keras.layers, input_config['class'])
        input_config.pop('class')
        format_dct(input_config)
        input_layer = input_class(**input_config)
        #remove the input layer so as to not try to add it again later
        layers = layers[1:]

    # add the layers
    for i, layer_ini in enumerate(layers):
        ini_dct = dict(config[layer_ini])
        format_dct(ini_dct)
        class_to_add = getattr(keras.layers, ini_dct['class'])
        ini_dct.pop('class')
        # sort out the activation for the layer
        try:
            activation = ini_dct['activation']  # see if the activation is set in the ini
        except KeyError:
            activation = None  # the activation is not set in the ini
        advanced_activator = False
        if not isinstance(activation, type(None)):  # if the activation is set see if its an advanced activator
            try:
                getattr(keras.layers.advanced_activations, activation)()
                advanced_activator = True
                ini_dct.pop('activation') #this will have to be added after the layer
            except AttributeError:
                advanced_activator = False
        if api.lower() == 'sequential':
            nn.add(class_to_add(**ini_dct))
        if api.lower() == 'functional':
            if i == 0:
                nn = class_to_add(**ini_dct)(input_layer)
            # if its the last layer
            elif i + 1 == layers.__len__():
                output_layer = class_to_add(**ini_dct)(nn)
            # one of the middle layers
            else:
                nn=class_to_add(**ini_dct)(nn)
        if advanced_activator:
            if api.lower() == 'sequential':
                nn.add(getattr(keras.layers.advanced_activations, activation)())
            if api.lower() == 'functional':
                #adding to the input_layer
                if i==0:
                    nn=getattr(keras.layers.advanced_activations, activation)(input_layer)
                #if its the last layer
                elif i+1 == layers.__len__():
                    output_layer=getattr(keras.layers.advanced_activations, activation)(nn)
                #one of the middle layers
                else:
                    nn=getattr(keras.layers.advanced_activations, activation)(nn)

    # figure out the optimizer
    if 'optimizer' in config.keys():
        #if there's an optimizer
        optimizer_dct = dict(config['optimizer'])
        format_dct(optimizer_dct)
        optimizer_class = getattr(keras.optimizers, optimizer_dct['class'])
        optimizer_dct.pop('class')
        optimizer = optimizer_class(**optimizer_dct)
    elif compiled:
        # Error if there is no optimizer
        raise ValueError('No optimizer found in %s. Please include one or set compiled = False.'%ini_loc)
    else:
        pass

    #compile if needed.
    if compiled:
        if api.lower() == 'functional':
            nn = keras.models.Model(inputs=input_layer, outputs=output_layer)
        nn.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=['accuracy'])
        return nn
    else:
        if api.lower() == 'functional':
            return [input_layer, output_layer]
        else:
            return nn

def network_from_ini_2(ini_path, compiled=True):
    """
    This is an updated version of network_from_ini that needs more testing, but should be more stable, include more
        features and more automated in the long run
    This is designed to take a ini file and create a neural network from it
    Specific format is needed, the format can be seen from outputs of the above functions that create the ini files
    In order to be consistent a sequental model should start with a dedicated InputLayer
    Perhpas the ini functions should be moved to their own file
    :param ini_path: The location of the ini file
    :param api: The api that the model will be made from. Currently supports functional and sequential
    :return: keras model based on the ini. If api=='functional' and compiled==False:
         returns [input_layer, output_layer] so the network can be manipulated or compiled
    """
    config = read_config_file(ini_path)
    layers = [s for s in config.sections() if s.startswith('layer/')]
    #make sure the first layer is an input
    if not (config[layers[0]]['class'] == "InputLayer" or config[layers[0]]['class'] == "Input"):
        raise ValueError('First layer of %s should be an input layer of class Input')
    strAPI = config['model']['class'].lower()

    #Initiate the model with the input layer
    if strAPI=='sequential':
        kerasModel = keras.models.Sequential()
        #add the input layer
        add_layer_from_config(kerasModel, config, layers[0], api=strAPI)
    if strAPI=='functional':
        input_config = dict(config[layers[0]])
        input_class = getattr(keras.layers, input_config['class'])
        input_config.pop('class')
        format_dct(input_config)
        kerasModel = input_class(**input_config)
        layerFunctionalInput = kerasModel
    #remove the input layer so as to not try to add it again later
    layers = layers[1:]

    # add the layers
    for intI, strLayer in enumerate(layers):
        kerasModel = add_layer_from_config(kerasModel, config, strLayer, api=strAPI)
        if intI + 1 == layers.__len__(): #last layer
                layerFunctionalOutput = kerasModel

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
            kerasModel = keras.models.Model(inputs=layerFunctionalInput, outputs=layerFunctionalOutput)
        kerasModel.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=metrics)
        return kerasModel
    #return non-compiled model
    else:
        if strAPI == 'functional':
            return [layerFunctionalInput, layerFunctionalOutput]
        else:
            return kerasModel

def add_layers_from_init(model, ini_path, compiled=True):
    """
    This adds layers to the end of a model from an ini file
    Note this will be done inplace!
    See also network_from_ini
    :param model: The model you'd like to add the layers to, if the keras api is functional then it should be a list [input_layer, output_layer]
    :param ini_loc: The location of the ini file
    :param api: functional or sequential
    :return: the new UNCOMPILED model
    """
    config = read_config_file(ini_path)
    layers = [s for s in config.sections() if s.startswith('layer/')]
    #make sure the first layer is an input
    strAPI = config['model']['class'].lower()

    if strAPI == 'sequential':
        kerasModel = model
    elif strAPI == 'functional':
        kerasModel = model[1]
        layerFunctionalInput = model[0]
    else:
        raise ValueError("%s is not a valid API, set in %s"%(strAPI, ini_path))

    # add the layers
    for intI, strLayer in enumerate(layers):
        kerasModel = add_layer_from_config(kerasModel, config, strLayer, api=strAPI)
        if intI + 1 == layers.__len__(): #last layer
                layerFunctionalOutput = kerasModel

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
            layerFunctionalInput = model.input
            kerasModel = keras.models.Model(inputs=layerFunctionalInput, outputs=layerFunctionalOutput)
        kerasModel.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=metrics)
        return kerasModel
    #return non-compiled model
    else:
        if strAPI == 'functional':
            return [layerFunctionalInput, layerFunctionalOutput]
        else:
            return kerasModel

def merged_network_from_file(ini_loc, network_inputs, merge_inputs, compiled=True):
    """
    This takes in two or more network and combines them into one with a dense network that is specified by the inputted
    ini file.
    :param ini_loc:(str) The location of the ini file that merges the inputs
    :param network_inputs:(list of tensor vectors) the initial inputs to the network
    :param merge_inputs:(list of tensor vectors) the inputs into the merge
    :param compiled:(bool) Return a compiled network.
    :return: a Keras neural network
    """
    if not os.path.exists(ini_loc):
        raise IOError('ini_loc: %r does not exisit'%ini_loc)
    #load the ini file
    config = configparser.ConfigParser()
    config.read(ini_loc)
    layers = [s for s in config.sections() if s.startswith('layer/')]
    #make sure the first layer is an input
    if (not config[layers[0]]['class'] == "merge") and (not config[layers[0]]['class'] == "concatenate"):
        raise ValueError('First layer of %s should be a Merge or Concatenate class')
    input_config = dict(config[layers[0]])
    try:
        input_class = getattr(keras.layers, input_config['class'])
    except AttributeError: #for some versions of keras merge is located in legacy
        input_class = getattr(keras.legacy.layers, input_config['class'])
    input_config.pop('class')
    format_dct(input_config)
    try:
        input_layer = input_class(inputs=merge_inputs, **input_config)
    except TypeError:
        input_layer = input_class(merge_inputs, **input_config)
    #remove the input layer so as to not try to add it again later
    layers = layers[1:]
    # add the layers
    for i, layer_ini in enumerate(layers):
        ini_dct = dict(config[layer_ini])
        format_dct(ini_dct)
        class_to_add = getattr(keras.layers, ini_dct['class'])
        ini_dct.pop('class')
        # sort out the activation for the layer
        try:
            activation = ini_dct['activation']  # see if the activation is set in the ini
        except KeyError:
            activation = None  # the activation is not set in the ini
        advanced_activator = False
        if not isinstance(activation, type(None)):  # if the activation is set see if its an advanced activator
            try:
                getattr(keras.layers.advanced_activations, activation)()
                advanced_activator = True
                ini_dct.pop('activation')
            except AttributeError:
                advanced_activator = False
        if i==0 and i + 1 == layers.__len__(): #if there is one layer and its the output layer
            output_layer = class_to_add(**ini_dct)(input_layer)
        elif i == 0: #if its the first layer that is not also the output
            nn = class_to_add(**ini_dct)(input_layer)
        # if its the last layer
        elif i + 1 == layers.__len__(): #if its the last layer, but not also the first
            output_layer = class_to_add(**ini_dct)(nn)
        # one of the middle layers
        else: #if its one of the middle layers
            nn=class_to_add(**ini_dct)(nn)
        if advanced_activator: #add the advanced activator if needed
            #adding to the input_layer
            if i==0:
                nn=getattr(keras.layers.advanced_activations, activation)(input_layer)
            #if its the last layer
            elif i+1 == layers.__len__():
                output_layer=getattr(keras.layers.advanced_activations, activation)(nn)
            #one of the middle layers
            else:
                nn=getattr(keras.layers.advanced_activations, activation)(nn)

    # figure out the optimizer
    if 'optimizer' in config.keys():
        #if there's an optimizer
        optimizer_dct = dict(config['optimizer'])
        format_dct(optimizer_dct)
        optimizer_class = getattr(keras.optimizers, optimizer_dct['class'])
        optimizer_dct.pop('class')
        optimizer = optimizer_class(**optimizer_dct)
    elif compiled:
        # Error if there is no optimizer
        raise ValueError('No optimizer found in %s. Please include one or set compiled = False.'%ini_loc)
    else:
        pass

    model = keras.models.Model(input=network_inputs, output=output_layer)
    if compiled:
        model.compile(loss=config['model']['loss'], optimizer=optimizer, metrics=['accuracy'])
    return model

def MEG_network_from_ini_folder(folder_location, compiled=True):
    """
    This takes in a folder location with three ini file inside: "spatial.ini", "temporal.ini" and "merge.ini".
    It then creates a network based on these three files where the temporal network and spatial network are merged as
    described by temporal.ini, spatial.ini and merge.ini respectively.

    :param folder_location:(str) The folder containing the 3 files
    :param compiled:(bool) if the model should be compiled before returning it.
    :return: A keras model as defined by the ini files in the folder.
    """
    spatial_ini = os.path.join(folder_location, 'spatial.ini')
    temporal_ini = os.path.join(folder_location, 'temporal.ini')
    dense_ini = os.path.join(folder_location, 'merge.ini')
    network_input_spatial, network_output_spatial = network_from_ini(spatial_ini, compiled=False)
    network_input_temporal, network_output_temporal = network_from_ini(temporal_ini, compiled=False)
    network = merged_network_from_file(ini_loc=dense_ini,
                                       network_inputs=[network_input_spatial, network_input_temporal],
                                       merge_inputs=[network_output_spatial, network_output_temporal],
                                       compiled=compiled)
    return network

def make_MEG_networks(output_folder=None,number_to_make=100):
    """
    Makes random networks for the MEG project using the functions in this script
    :param output_folder:(str) This is where the networks will be saved. None is default, but will be changed to
    project/bioinformatics/DLLab/Alex/Projects/MEG_artifact/MEG_AAM/McGill_data/model3/random_search/models
    :param number_to_make:(int) The number of networks to make
    :return: None
    """
    if output_folder == None:
        output_folder = r'/project/bioinformatics/DLLab/Alex/Projects/MEG_artifact/MEG_AAM/McGill_data/model3/random_search/models'
    for i in range(2,number_to_make+1):
        nn_folder = os.path.join(output_folder, 'model%i'%i)
        make_MEG_ini_folder(nn_folder)

def format_dct(dct):
    """
    Takes a dictionary and attempts to make the dictionary values their native python class
    EG {'a':'[1,2,3]','b':'5','c':'foo'} = {'a':[1,2,3],'b':5,'c':'foo'} where 5 is an int and [1,2,3] a list
    :param dct: The dict to change
    :return: The formatted dct, note this is done inplace!!!
    """
    for key, value in dct.items():  # Attempt to turn the type of the values into something other than a string
        try:
            dct[key] = ast.literal_eval(value)
        except ValueError:
            dct[key] = value
    return dct

def model_memory_requirement(model, float_bit=32, training=True, momentum=True, batch_size=1, unit='MB'):
    """
    This calculates an estimate of the size (in MB) of model of the inputted network.
    A batch size can be given to estimate the memory needed whn running with that batch_size.
    If not training the model requires a significantly smaller amount of memory.
    Fitting a model using momentum requires more memory
    :param model:(keras model) The model
    :param float_bit:(int) the number of bits used for the floats in your model
    :param training:(bool) True if estimate for training, False if used for prediction
    :param momentum:(bool) If the optimizer for the network uses momentum
    :param batch_size:(int) The number of inputs in your mini-batch
    :param unit:(str) The unit of memory to return it size in, acceptable inputs: 'bit', 'byte', 'KB', 'MB', 'GB'
    :return: The estimated size in MB of the network
    """
    # www.youtube.com/watch?v=Hqtg7fznlnM
    # slides => http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
    """
    Notes on calculating this
    Get the number of parameters using keras network.count_params(). This will return the number of weights + number of
        biases.
    This could be an underestimate if you are using a CPU as the way the convolutions are calculated are turned into matrix
    multiplication and so take more memory. This is called convolution lowering

    Also currently not included is the overhead (miscilanious memory tensorflow, keras, python, etc),
        this could be added later if we can determine what it is.
    """
    model_size = model.count_params() * float_bit
    sum_output_size = np.sum(layer_blob_size(l) for l in model.layers)
    if not training:
        model_memory = model_size+sum_output_size
    else:
        model_mul = 3
        if momentum == False:
            model_mul = 2
        model_memory = (model_size * model_mul) + (sum_output_size * 2)
    bit_memory = model_memory*batch_size
    if unit=='bit':
        return bit_memory
    elif unit=='byte':
        return bit_memory/8.0
    elif unit=='KB':
        return (bit_memory/8.0)/(1024.0)
    elif unit=='MB':
        return (bit_memory/8.0)/(1024.0**2)
    elif unit=='GB':
        return (bit_memory/8.0)/(1024.0**3)

def flops(network):
    """

    :param network:
    :return:
    """
    """
    FLOPS are a measure of model complexity and indecat how much computation time is needed to train 
    """
    return np.sum([layer_flops(x) for x in network])

def layer_flops(layer):
    #this still needs vast testing, currently tested with Dense and Cond2D layers
    if layer.count_params() == 0:
        # if there are no params then there are no FLOPS, this is atleast true for input layers, flatten layers and max pooling layers
        return 0
    to_mult=[]
    to_mult.extend(layer.output_shape)
    to_mult.extend(layer.input_shape)

    try: #if it is a convolutional layer, this needs to be taken into account
        to_mult.extend(layer.kernel_size)
    except AttributeError:
        pass
    return np.prod([x for x in to_mult if x != None])

def layer_blob_size(layer):
    """
    This calculates the blob size for a layer, that is the amount of memory that the layer takes up in memory
    Note: adding the blob size for each layer does not give the memory size of that model, due to the input and output
        weights that are also saved.
    :param layer: the layer to calculate the blob size from
    :return: the blob size
    """
    return np.prod([x for x in layer.output_shape if x != None])

def network_summary_from_ini(ini_loc):
    #set up some important settings:
    functional_layers = {'Dense':'units',
                         'Conv1D':'kernel_size',
                         'Conv2D': 'kernel_size',
                         'Conv3D': 'kernel_size',
                         }
    if not os.path.exists(ini_loc):
        raise IOError('ini_loc: %r does not exisit'%ini_loc)
    #load the ini file
    config = configparser.ConfigParser()
    config.read(ini_loc)
    layers = [s for s in config.sections() if s.startswith('layer/')]
    functional_layer_count=0
    functional_shapes=[]
    for layer in layers:
        layer_dct = format_dct(dict(config[layer]))
        layer_class = layer_dct['class']
        if layer_class in functional_layers.keys():
            functional_layer_count+=1
            functional_shapes.append(layer_dct[functional_layers[layer_class]])
    return functional_layer_count, functional_shapes

def network_summary_from_ini(ini_loc):
    #set up some important settings:
    report_layers = {'Dense':'units',
                     'Conv1D':['filters','kernel_size'],
                     'Conv2D': ['filters','kernel_size'],
                     'Conv3D': ['filters','kernel_size'],
                     'MaxPooling':[]
                     }
    layers_to_ignore = ['PReLU']
    if not os.path.exists(ini_loc):
        raise IOError('ini_loc: %r does not exisit'%ini_loc)
    text = ""


#used for testing
if __name__ == "__main__":
    if False:
        ini_loc = r'/project/bioinformatics/DLLab/Alex/Projects/utilities/1.ini'
        ini_loc = r'/project/bioinformatics/DLLab/Alex/Projects/Liver_Fibrosis/Larger_ROI/random_search2/networks/2.ini'
        nn = network_from_ini(ini_loc, api='sequential')
    if False:
        output_file = r'/project/bioinformatics/DLLab/Alex/Projects/MEG_artifact/MEG_AAM/McGill_data/model3/random_search/models/model1/spatial.ini'
        MEG_spatial_test(output_file)
        network_input_spatial, network_output_spatial = network_from_ini(output_file, compiled=False)
        output_file2 = r'/project/bioinformatics/DLLab/Alex/Projects/MEG_artifact/MEG_AAM/McGill_data/model3/random_search/models/model1/temporal.ini'
        MEG_temporal_test(output_file2)
        network_input_temporal, network_output_temporal = network_from_ini(output_file2, compiled=False)
        output_file3 = r'/project/bioinformatics/DLLab/Alex/Projects/MEG_artifact/MEG_AAM/McGill_data/model3/random_search/models/model1/merge.ini'
        MEG_dense_test(output_file3)
        network = merged_network_from_file(ini_loc=output_file3,
                                           network_inputs=[network_input_spatial,network_input_temporal],
                                           merge_inputs=[network_output_spatial,network_output_temporal])
        print(network.summary())
    if False:
        #chech the MEG networks
        output_path = r'/project/bioinformatics/DLLab/Alex/Projects/utilities/temp/modelx'
        make_MEG_ini_folder(output_path)
        nn = MEG_network_from_ini_folder(output_path)
    if False:
        nn=keras.models.Sequential()
        nn.add(keras.layers.InputLayer((28,28,1)))
        nn.add(keras.layers.Conv2D(32, (3,3), padding='same'))
        nn.add(keras.layers.Conv2D(64, (3,3), padding='same'))
        nn.add(keras.layers.Conv2D(32, (3,3), padding='same'))
        nn.add(keras.layers.MaxPool2D(2,2))
        nn.add(keras.layers.Flatten())
        nn.add(keras.layers.Dense(128))
        nn.add(keras.layers.Dense(10))
        nn.summary()
