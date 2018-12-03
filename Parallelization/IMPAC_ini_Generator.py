""" This file generates ini files containing network architextures
for neural networks to be trained

Sept 2018: By Cooper Mellema in the Montillo Lab
"""

import os
import numpy as np
from configparser import ConfigParser
import random

np.random.seed(42)

bNewDense=False
bNewLSTM=False
bNewBrainNet=False
bNewStack=True

# Initialize the directories where the ini files will be stored
def fInitialize():
    # Set up figure saving
    sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
    sProjectIdentification = "AutismProject/Parallelization"

    sTargetDirectory = os.path.join(sProjectRootDirectory, sProjectIdentification, 'IniFiles/StackedWithRegularization')

    if not os.path.exists(sTargetDirectory):
        os.makedirs(sTargetDirectory)

    return sTargetDirectory

sTargetDirectory = fInitialize()


##############################################################################
# Initialize the ini files for the Dense Neural network

# Create a dictionary of hyperparameters to randomly select from

dHyperParam = {
    'hidden_layers': [1, 2, 3],
    'bottom_layer_width': [16, 32, 64, 128, 256],
    'Dropout': np.random.uniform(0.1, 0.6, 10),
    'regularization': 10 ** np.random.uniform(-4, -2, 10)
}


def fGenerateDenseNetworkINIs(sTargetDirectory, dHyperParam): #Input shape?
    """ Adapted from Alex Treacher's network_functions code by Cooper Mellema
    Makes semi random config file for a keras network, designed around the IMPAC Autism Project.
    Creates a 2 class network architecture with a hyperparameter set in the space defined by the
    dictionary dHyperparameters, which is passed in

    See also cnn_random_n_classifier_builder. (these two functions could easily be combined)
    :param sTargetDirectory: where to save the file
    :return: None
    """

    # Create 50 .ini files
    for iRandomInitialization in range(50):
        random.seed(iRandomInitialization)
        sIni = 'Dense_' + str(iRandomInitialization)

        iLayers = random.choice(dHyperParam['hidden_layers'])
        iBottomLayerWidth = random.choice(dHyperParam['bottom_layer_width'])
        flDropout = random.choice(dHyperParam['Dropout'])
        flRegularizer = random.choice(dHyperParam['regularization'])

        #Initialize the ini file with ConfigParser
        config = ConfigParser()
        config['model'] = {'class': 'Sequential', 'loss': 'binary_crossentropy'}

        # Create the input layer
        config['layer/input'] = {'class': 'Dense',
                                 # 'input_shape': str(aInputShape),#[len of vect, 1, 1 channel]
                                 'units': str(iBottomLayerWidth),
                                 'regularizer': str(flRegularizer),
                                 'activation': 'relu',
                                 'alpha': str(0.3)
                                 }

        # Create iLayers number of hidden layers with a Dropout layer before each
        for i in range(iLayers):
            config['layer/Dropout{}'.format(i)] ={'class': 'Dropout',
                                              'rate': flDropout
                                              }

            config['layer/Dense{}'.format(i)] = {'class': 'Dense',
                                             'units': int(iBottomLayerWidth/(i+1)),
                                             'activation': 'relu',
                                             'regularizer': str(flRegularizer),
                                             'alpha': str(0.3)
                                             }

        # Create the decision layer
        config['layer/Dense{}'.format(i+1)] = {'class': 'Dense',
                                           'units': 1,
                                           'regularizer': str(flRegularizer),
                                           'activation': 'sigmoid'
                                           }

        # set the optimizer for the training
        config['optimizer'] = {'class': 'nadam'}

        # set the batch size
        config['batch_size'] = {'class': '128'}

        with open(os.path.join(sTargetDirectory, sIni) + '.ini', 'w') as configfile:
            config.write(configfile)

if bNewDense:
    fGenerateDenseNetworkINIs(sTargetDirectory, dHyperParam)

##############################################################################
# Initialize the ini files for the LSTM Neural network

# Create a dictionary of hyperparameters to randomly select from
dHyperParam = {
    'hidden_layers': [1, 2, 3],
    'bottom_layer_width': [32, 64, 128, 256],
    'Dropout': np.random.uniform(0.1, 0.6, 10),
    #'gradient_clipping': 10**np.random.uniform(-1, 1, 10),
    'regularization': 10**np.random.uniform(-4, -2, 10)
}

def fGenerateLSTMNetworkINIs(sTargetDirectory, dHyperParam): #Input shape?
    """ Adapted from Alex Treacher's network_functions code by Cooper Mellema
    Makes semi random config file for a keras network, designed around the IMPAC Autism Project.
    Creates a 2 class network architecture with a hyperparameter set in the space defined by the
    dictionary dHyperparameters, which is passed in

    See also cnn_random_n_classifier_builder. (these two functions could easily be combined)
    :param sTargetDirectory: where to save the file
    :return: None
    """

    # Create 50 .ini files
    for iRandomInitialization in range(50):
        random.seed(iRandomInitialization)
        sIni = 'LSTM_' + str(iRandomInitialization)

        iLayers = random.choice(dHyperParam['hidden_layers'])
        iBottomLayerWidth = random.choice(dHyperParam['bottom_layer_width'])
        flDropout = random.choice(dHyperParam['Dropout'])
        #flGradientClipping = random.choice(dHyperParam['gradient_clipping'])
        flRegularizer = random.choice(dHyperParam['regularization'])

        #Initialize the ini file with ConfigParser
        config = ConfigParser()
        config['model'] = {'class': 'Sequential', 'loss': 'binary_crossentropy'}

        # Create the input layer
        config['layer/input'] = {'class': 'LSTM',
                                 # 'input_shape': str(aInputShape),#[len of vect, 1, 1 channel]
                                 'units': str(iBottomLayerWidth),
                                 'activation': 'relu',
                                 'regularizer': str(flRegularizer),
                                 'alpha': str(0.3),
                                 'return_sequences': 'True'
                                 }

        # Create iLayers number of hidden layers with a Dropout layer before each
        for i in range(iLayers):
            config['layer/Dropout{}'.format(i)] ={'class': 'Dropout',
                                              'rate': flDropout
                                              }

            config['layer/LSTM{}'.format(i)] = {'class': 'LSTM',
                                            'units': int(iBottomLayerWidth/(i+1)),
                                            #'gradient_clipping': str(flGradientClipping),
                                            'regularizer': str(flRegularizer),
                                            'activation': 'relu',
                                            'alpha': str(0.3),
                                            'return_sequences': 'True'
                                            }

        # Create the decision layer
        config['layer/Dense{}'.format(i+1)] = {'class': 'Dense',
                                           'units': 1,
                                           'regularizer': str(flRegularizer),
                                           'activation': 'sigmoid'
                                           }

        # set the optimizer for the training
        config['optimizer'] = {'class': 'nadam'}

        # set the batch size
        config['batch_size'] = {'class': '128'}

        with open(os.path.join(sTargetDirectory, sIni) + '.ini', 'w') as configfile:
            config.write(configfile)

if bNewLSTM:
    fGenerateLSTMNetworkINIs(sTargetDirectory, dHyperParam)


##############################################################################
# Initialize the ini files for the BrainNet Convolutional Neural network

# Create a dictionary of hyperparameters to randomly select from
dHyperParam = {
    'hidden_layers': [0, 1, 2],
    'bottom_layer_width': [16, 32, 64],
    'Dropout': np.random.uniform(0.1, 0.6, 10),
    'relu_slope': np.random.uniform(0.1, 0.5, 10)
}

def fGenerateBrainNetCNNINIs(sTargetDirectory, dHyperParam):  # Input shape?
    """ Adapted from Alex Treacher's network_functions code by Cooper Mellema
    Makes semi random config file for BrainNetCNN, impletmented in caffe,
    designed around the IMPAC Autism Project.
    Creates a multi-class network architecture with a hyperparameter set in the space defined by the
    dictionary dHyperparameters, which is passed in

    See also cnn_random_n_classifier_builder. (these two functions could easily be combined)
    :param sTargetDirectory: where to save the file
    :return: None
    """

    # Create 50 .ini files
    for iRandomInitialization in range(50):
        random.seed(iRandomInitialization)
        sIni = 'BrainNetCNN_' + str(iRandomInitialization)

        iLayers = random.choice(dHyperParam['hidden_layers'])
        iBottomLayerWidth = random.choice(dHyperParam['bottom_layer_width'])
        flDropout = random.choice(dHyperParam['Dropout'])
        flReluSlope = random.choice(dHyperParam['relu_slope'])

        # Initialize the ini file with ConfigParser
        config = ConfigParser()
        config['model'] = {'class': 'Caffe_sequential'}

        # Create the input layer
        config['layer/input0'] = {'class': 'e2e',
                                  'n_filters': str(iBottomLayerWidth),
                                  }


        # Create iLayers number of hidden layers with a Dropout layer after each
        for i in range(iLayers):

            config['layer/edge2edge{}'.format(i+1)] = {'class': 'e2e',
                                                   'n_filters': int(iBottomLayerWidth), #/somthing?
                                                   }

            # config['layer/Dropout{}'.format(i+1)] = {'class': 'dropout',
            #                                      'dropout_ratio': str(flDropout)
            #                                      }
            #
            # config['layer/Activation{}'.format(i+1)] = {'class': 'activation',
            #                                         'activation': 'relu',
            #                                         'negative_slope': str(flReluSlope)
            #                                         }

        # Create the edge to node layer with Dropout after
        config['layer/edege2node%i' % (i + 2)] = {'class': 'e2n',
                                                  'n_filters': str(int(2*iBottomLayerWidth)),
                                                  }

        config['layer/Dropout%i' % (i + 2)] = {'class': 'dropout',
                                               'dropout_ratio': flDropout
                                               }

        config['layer/Activation%i' % (i + 2)] = {'class': 'activation',
                                                  'activation': 'relu',
                                                  'negative_slope': str(flReluSlope)
                                                  }

        # Create the decision layer with Dropout before
        config['layer/FullyConnected%i' % (i + 3)] = {'class': 'fc',
                                                      'n_filters': str(int(2 * iBottomLayerWidth)),
                                                      }

        config['layer/Activation%i' % (i + 3)] = {'class': 'activation',
                                                  'activation': 'relu',
                                                  'negative_slope': str(flReluSlope)
                                                  }

        config['layer/output%i' % (i+3)] = {'class': 'out',
                                            'n_filters': str(1)
                                            }

        # set the optimizer for the training
        config['optimizer'] = {'class': 'nadam'}

        # set the batch size
        config['batch_size'] = {'class': '128'}

        with open(os.path.join(sTargetDirectory, sIni) + '.ini', 'w') as configfile:
            config.write(configfile)

if bNewBrainNet:
    fGenerateBrainNetCNNINIs(sTargetDirectory, dHyperParam)

##########################################################################
# Initialize the ini files for the Stacked Dense Network

# Create a dictionary of hyperparameters to randomly select from

dHyperParam = {
    'regularization': ['l1', 'l2', 'l1_l2'],
    'hidden_layers': [1, 2, 3, 4],
    'bottom_layer_width': [8, 16, 32, 64, 128],
    'Dropout': np.random.uniform(0.1, 0.6, 10),
    'l1': 10 ** np.random.uniform(-4, -2, 10),
    'l2': 10 ** np.random.uniform(-4, -2, 10),
    'batch_normalization': [True, False]
}

def fGenerateStackNetworkINIs(sTargetDirectory, dHyperParam): #Input shape?
    """ Adapted from Alex Treacher's network_functions code by Cooper Mellema
    Makes semi random config file for a keras network, designed around the IMPAC Autism Project.
    Creates a 2 class network architecture with a hyperparameter set in the space defined by the
    dictionary dHyperparameters, which is passed in

    See also cnn_random_n_classifier_builder. (these two functions could easily be combined)
    :param sTargetDirectory: where to save the file
    :return: None
    """

    # Create 50 .ini files
    for iRandomInitialization in range(50):
        random.seed(iRandomInitialization)
        if iRandomInitialization<10:
            sIni = 'Stack_0' + str(iRandomInitialization)
        else:
            sIni = 'Stack_' + str(iRandomInitialization)

        iLayers = random.choice(dHyperParam['hidden_layers'])
        iBottomLayerWidth = random.choice(dHyperParam['bottom_layer_width'])
        flDropout = random.choice(dHyperParam['Dropout'])
        flL1 = random.choice(dHyperParam['l1'])
        flL2 = random.choice(dHyperParam['l2'])
        bBatchNormalization = random.choice(dHyperParam['batch_normalization'])

        #Initialize the ini file with ConfigParser
        config = ConfigParser()
        config['model'] = {'class': 'Functional', 'loss': 'binary_crossentropy', 'regularization':
            np.random.choice(dHyperParam['regularization'])}

        if bBatchNormalization:
            config['layer/inputNormalize'] ={'class': 'BatchNormalization',
                                             'momentum': 0.99,
                                             'epsilon': 0.0000001
                                            }

        # Create the input layer
        config['layer/input'] = {'class': 'Dense',
                                 'units': str(iBottomLayerWidth),
                                 'l1': str(flL1),
                                 'l2': str(flL2),
                                 'activation': 'relu',
                                 'alpha': str(0.3)
                                 }

        # if there is no l1 regularization, get rid of the l1 param, and same for l2
        if config['model']['regularization']=='l1':
            config['layer/input'].pop('l2')
        elif config['model']['regularization']=='l2':
            config['layer/input'].pop('l1')

        # Create iLayers number of hidden layers with a Dropout layer before each
        for i in range(iLayers):
            if bBatchNormalization:
                config['layer/inputNormalize{}'.format(i)] = {'class': 'BatchNormalization',
                                                          'momentum': 0.99,
                                                          'epsilon': 0.0000001
                                                          }

            config['layer/Dropout{}'.format(i)] ={'class': 'Dropout',
                                              'rate': flDropout
                                              }

            config['layer/Dense{}'.format(i)] = {'class': 'Dense',
                                             'units': int(iBottomLayerWidth/(i+1)),
                                             'activation': 'relu',
                                             'l1': str(flL1),
                                             'l2': str(flL2),
                                             'activation': 'relu',
                                             'alpha': str(0.3)
                                             }

            # if there is no l1 regularization, get rid of the l1 param, and same for l2
            if config['model']['regularization'] == 'l1':
                config['layer/Dense{}'.format(i)].pop('l2')
            elif config['model']['regularization'] == 'l2':
                config['layer/Dense{}'.format(i)].pop('l1')


        # Create the decision layer
        config['layer/Dense{}'.format(i+1)] = {'class': 'Dense',
                                           'units': 1,
                                           'l1': str(flL1),
                                           'l2': str(flL2),
                                           'activation': 'sigmoid',
                                           'alpha': str(0.3)
                                           }

        # if there is no l1 regularization, get rid of the l1 param, and same for l2
        if config['model']['regularization'] == 'l1':
            config['layer/Dense{}'.format(i+1)].pop('l2')
        elif config['model']['regularization'] == 'l2':
            config['layer/Dense{}'.format(i+1)].pop('l1')
        # set the optimizer for the training
        config['optimizer'] = {'class': 'nadam'}

        # set the batch size
        config['batch_size'] = {'class': '128'}

        with open(os.path.join(sTargetDirectory, sIni) + '.ini', 'w') as configfile:
            config.write(configfile)



if bNewStack:
    config=fGenerateStackNetworkINIs(sTargetDirectory, dHyperParam)
