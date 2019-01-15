""" This code takes the Paris IMPAC Autism study data,and
trains a Brain Net CNN as a classifier

The performance metrics being used are:

    - accuracy
    - Precision-Recall area under curve
    - F1 score
    - ROC area under curve

Written by Cooper Mellema in the Montillo Deep Learning Lab at UT Southwesten Medical Center
Sept 2018
"""
import os
import sys
sBrainNetCNNCode = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master'
sBrainNetCNNCode2 = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master/ann4brains'
# sys.path.append(sCaffePath)

sys.path.append(sBrainNetCNNCode)
sys.path.append(sBrainNetCNNCode2)
#import BrainNetCNNCode
from ann4brains.nets import BrainNetCNN

import numpy as np

import ast
import configparser
import pickle
import sklearn.metrics as skm

# import subprocess
# subprocess.call(["export", "PYTHONPATH=/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/caffe"
#                            "-master/python:$PYTOHNPATH"], shell=True)
# sCaffePath="/project/bioinformatics/DLLab/Cooper/Libraries/caffe1.0.0_segnet/caffe/python"


sBrainNetCNNCode = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master'
sBrainNetCNNCode2 = '/project/bioinformatics/DLLab/Cooper/Libraries/ann4brains-master/ann4brains-master/ann4brains'
# sys.path.append(sCaffePath)

sys.path.append(sBrainNetCNNCode)
sys.path.append(sBrainNetCNNCode2)
#import BrainNetCNNCode
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
        lsLayer = ['e2e', {'n_filters': ini_dict['n_filters'],
                            'kernel_h': aInputShape[0],
                            'kernel_w': aInputShape[1]
                            }]

    elif sClass=='dropout':
        lsLayer = ['dropout', {'dropout_ratio': ini_dict['dropout_ratio']}]

    elif sClass=='activation':
        lsLayer = [ini_dict['activation'], {'negative_slope': ini_dict['negative_slope']}]

    elif sClass=='e2n':
        lsLayer = ['e2n', {'n_filters': ini_dict['n_filters'],
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
                             'layer names are: e2e, Dropout, activation, e2n, fc, and out')
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

def fRunBrainNetCNNOnInput(sInputName, iModelNum, dData, sSubInputName='', iEpochs=1, bEarlyStopping=True):

    sIni = 'BrainNetCNN_' + str(iModelNum)
    sIniPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/IniFiles/' + sIni + '.ini'
    sSavePath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/ISBIRerun' \
                '/BrainNetCNN'

    if not os.path.isfile(sSavePath + '/' + sIni + sInputName + "FullModelNetwork.p"):

        dXData = dData['dXData']
        dXTest = dData['dXTest']
        aYData = dData['aYData']
        aYTest = dData['aYtest']

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

        #First we fit the cross-validation models,
        for iCrossVal in range(3):
            if not os.path.isfile(sSavePath + sIni + "Cross Val"+str(iCrossVal)+"Network.p"):

                print 'running cross val ' + str(iCrossVal)
                BrainNetCNNModel = fModelArchFromIni(sIniPath, sIni, aDataShape, sSavePath, (iCrossVal+1))

                BrainNetCNNModel.pars['max_iter'] = 500 #500
                BrainNetCNNModel.pars['test_interval'] = 50
                BrainNetCNNModel.pars['snapshot'] = 100 #100

                BrainNetCNNModel.fit(lsXDataSplit[iCrossVal][0], lsYDataSplit[iCrossVal][0], lsXVal[iCrossVal][0], lsYVal[iCrossVal][0])
                aPredicted=BrainNetCNNModel.predict(lsXVal[iCrossVal][0])

                pickle.dump(aPredicted, open(os.path.join(sSavePath, sIni) + sSubInputName +
                                             'CrossVal' + str(
                    iCrossVal) + 'Predicted.p', 'wb'))

                # # Now we find the performance by several metrics
                # print type(lsYVal[iCrossVal])
                # print type(aPredicted)
                # aYVal=np.array(lsYVal[iCrossVal][0][:])
                # aYVal=np.rint(aYVal)
                #
                # print aYVal.shape
                # print aPredicted
                #
                # BrainNetROCAUCScore = skm.roc_auc_score(aYVal, aPredicted)
                # precisions, recalls, thresholds = skm.precision_recall_curve(lsYVal[iCrossVal], aPredicted)
                # BrainNetPRAUCScore = skm.auc(recalls, precisions)
                # BrainNetF1Score = skm.f1_score(lsYVal[iCrossVal], np.rint((aPredicted-min(aPredicted))/(max(aPredicted)-min(
                #     aPredicted))))
                # BrainNetAccuracyScore = skm.accuracy_score(lsYVal[iCrossVal], np.rint((aPredicted-min(aPredicted))/(max(
                #     aPredicted)-min(aPredicted))), normalize=True)
                #
                # #Then we save the performace metrics
                pickle.dump(BrainNetCNNModel, open(sSavePath + '/' + sIni + sSubInputName + "Cross Val"+str(
                    iCrossVal)+"Network.p",'wb'))
                #
                # dResults ={
                #     'acc': BrainNetAccuracyScore,
                #     'pr_auc': BrainNetPRAUCScore,
                #     'F1': BrainNetF1Score,
                #     'roc_auc': BrainNetROCAUCScore
                # }
                #
                # pickle.dump(dResults, open(sSavePath + sIni + "Cross Val"+str(iCrossVal)+"SummaryResults.p", 'wb'))

        #Then we fit the full model
        if not os.path.isfile(sSavePath + '/' + sIni + sInputName + "FullModelNetwork.p"):
            print 'running full model'
            BrainNetCNNModel = fModelArchFromIni(sIniPath, sIni, aDataShape, sSavePath, 0)

            BrainNetCNNModel.pars['max_iter'] = 500 #500
            BrainNetCNNModel.pars['test_interval'] = 50
            BrainNetCNNModel.pars['snapshot'] = 100 #100

            BrainNetCNNModel.fit(aXData, aYData, aXTest, aYTest)
            aPredicted = BrainNetCNNModel.predict(aXTest)

            pickle.dump(aPredicted, open(os.path.join(sSavePath, sIni) + sSubInputName + 'FullModelPredicted.p',
                        'wb'))

            # # Now we find the performance by several metrics
            # BrainNetROCAUCScore = skm.roc_auc_score(aYTest, aPredicted)
            # precisions, recalls, thresholds = skm.precision_recall_curve(aYTest, aPredicted)
            # BrainNetPRAUCScore = skm.auc(recalls, precisions)
            # BrainNetF1Score = skm.f1_score(aYTest, np.rint((aPredicted - min(aPredicted)) / (max(aPredicted) - min(
            #     aPredicted))))
            # BrainNetAccuracyScore = skm.accuracy_score(aYTest, np.rint((aPredicted - min(aPredicted)) / (max(
            #     aPredicted) - min(aPredicted))), normalize=True)
            #
            # # Then we save the performace metrics
            pickle.dump(BrainNetCNNModel, open(sSavePath + '/' + sIni + sSubInputName + "FullModelNetwork.p", 'wb'))
            #
            # dResults = {
            #     'acc': BrainNetAccuracyScore,
            #     'pr_auc': BrainNetPRAUCScore,
            #     'F1': BrainNetF1Score,
            #     'roc_auc': BrainNetROCAUCScore
            # }
            #
            # pickle.dump(dResults, open(sSavePath + sIni + "FullModelSummaryResults.p", 'wb'))





if '__main__' == __name__:
    sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/TrainTestDataPy2.pkl'
    dData = pickle.load(open(sDataPath, 'rb'))

    bTest=False

    if not bTest==True:
        iModel = sys.argv[1]
        iModel = iModel.split('_')[1]
        iModel = iModel.split('.')[0]

        sInputName = 'connectivity'

        for sAtlas in dData['dXData'][sInputName]:
            print 'running ' + sAtlas + ' atlas'


            fRunBrainNetCNNOnInput(sInputName, iModel, dData, sSubInputName=sAtlas)


    else:
        for iModel in range(50):

            if iModel<10:
                iModel='0'+str(iModel)
            else:
                iModel=str(iModel)

            sInputName='connectivity'

            for sAtlas in dData['dXData'][sInputName]:
                print 'running ' + sAtlas + ' atlas'

                fRunBrainNetCNNOnInput(sInputName, iModel, dData, sSubInputName=sAtlas)




