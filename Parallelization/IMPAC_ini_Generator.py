""" This file generates ini files containing network architextures
for neural networks to be trained

Sept 2018: By Cooper Mellema in the Montillo Lab
"""

import os
import numpy as np
from configparser import ConfigParser
import random

np.random.seed(42)

# Initialize the directories where the ini files will be stored
def fInitialize():
    # Set up figure saving
    sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
    sProjectIdentification = "AutismProject/Parallelization"

    sTargetDirectory = os.path.join(sProjectRootDirectory, sProjectIdentification, 'IniFiles')

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
            config['layer/Dropout%i' % (i)] ={'class': 'Dropout',
                                              'rate': flDropout
                                              }

            config['layer/Dense%i' % (i)] = {'class': 'Dense',
                                             'units': int(iBottomLayerWidth/(i+1)),
                                             'activation': 'relu',
                                             'regularizer': str(flRegularizer),
                                             'alpha': str(0.3)
                                             }

        # Create the decision layer
        config['layer/Dense%i' % (i+1)] = {'class': 'Dense',
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

#fGenerateDenseNetworkINIs(sTargetDirectory, dHyperParam)

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
            config['layer/Dropout%i' % (i)] ={'class': 'Dropout',
                                              'rate': flDropout
                                              }

            config['layer/LSTM%i' % (i)] = {'class': 'LSTM',
                                            'units': int(iBottomLayerWidth/(i+1)),
                                            #'gradient_clipping': str(flGradientClipping),
                                            'regularizer': str(flRegularizer),
                                            'activation': 'relu',
                                            'alpha': str(0.3),
                                            'return_sequences': 'True'
                                            }

        # Create the decision layer
        config['layer/Dense%i' % (i+1)] = {'class': 'Dense',
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

#fGenerateLSTMNetworkINIs(sTargetDirectory, dHyperParam)


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

            config['layer/edge2edge%i' % (i+1)] = {'class': 'e2e',
                                                   'n_filters': int(iBottomLayerWidth), #/somthing?
                                                   }

            # config['layer/Dropout%i' % (i+1)] = {'class': 'dropout',
            #                                      'dropout_ratio': str(flDropout)
            #                                      }
            #
            # config['layer/Activation%i' % (i+1)] = {'class': 'activation',
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

fGenerateBrainNetCNNINIs(sTargetDirectory, dHyperParam)

##########################################################################

