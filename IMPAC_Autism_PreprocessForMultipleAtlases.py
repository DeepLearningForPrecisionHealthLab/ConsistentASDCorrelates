"""This file sets up the XData for using two atlases at once as input features

The atlases used are as follows:
BASC atlas:
    -64 ROIs
    -122 ROIs
    -197 ROIs
Craddock atlas
Harvard-Oxford atlas
MSDL atlas
Power atlas

The goal will be to have a dictionary with each atlas combined pairwise,
as well as combined pairwise with anatomical data, eg
dXData.keys()=
Basc 64 ROI & Basc 122 ROI
...
Basc 64 ROI & Power
...
Craddock & Harv-Ox
...
Anatomical & Basc 64 ROI & Basc 122 ROI
...
Anatomical & Basc 64 ROI & Power
...
Anatomical & Craddock & Harv-Ox
...

The goal will to be have 7X7(-7 for self & self) combinations, half with
anatomical data, half without

"""
import numpy as np
import pickle
import os

# Initialize the directories where the data is stored
sProjectRootDirectory = "/project/bioinformatics/DLLab/Cooper/Code"
sProjectIdentification = "AutismProject"

# First, check if the data exists. If it does, just load it, if not,
# then fetch it and start appending 2 atlases at a time
if not os.path.isfile(os.path.join(sProjectRootDirectory, sProjectIdentification, 'TrainTestData2Atlas.p')):

    # Load the data
    [dXTrain, dXTest, aYTrain, aYTest] = pickle.load(open(os.path.join(sProjectRootDirectory,
                                                                           sProjectIdentification, 'TrainTestData.p'), 'rb'))

    # Set up the lists to loop through during preprocessing
    lsAtlases = list(dXTrain['connectivity'].keys())
    aAnatomicTrain = dXTrain['anatomy']
    aAnatomicTest = dXTest['anatomy']

    # Set up a new dictionary to store the data
    dXTrain2Atlas={}
    dXTest2Atlas={}

    # Loop through lists and append them to each other
    for iOuterIndex, sOuterAtlas in enumerate(lsAtlases):
        for iInnerIndex, sInnerAtlas in enumerate(lsAtlases):

            # If in the upper triangle, append the two atlases
            if iOuterIndex < iInnerIndex:
                sCombinedName = sOuterAtlas + '_' + sInnerAtlas
                aXNewTrain1 = dXTrain['connectivity'][sOuterAtlas]
                aXNewTrain2 = dXTrain['connectivity'][sInnerAtlas]
                aXNewTest1 = dXTest['connectivity'][sOuterAtlas]
                aXNewTest2 = dXTest['connectivity'][sInnerAtlas]

                dXTrain2Atlas[sCombinedName] = np.concatenate((aXNewTrain1, aXNewTrain2), axis=1)
                dXTest2Atlas[sCombinedName] = np.concatenate((aXNewTest1, aXNewTest2), axis=1)

            # If in the lower triangle, append the two atlases AND the anatomical data
            elif iOuterIndex > iInnerIndex:
                sCombinedName = 'anatomical' + '_' + sInnerAtlas + '_' + sOuterAtlas
                aXNewTrain1 = aAnatomicTrain
                aXNewTrain2 = dXTrain['connectivity'][sInnerAtlas]
                aXNewTrain3 = dXTrain['connectivity'][sOuterAtlas]
                aXNewTest1 = aAnatomicTest
                aXNewTest2 = dXTest['connectivity'][sInnerAtlas]
                aXNewTest3 = dXTest['connectivity'][sOuterAtlas]

                dXTrain2Atlas[sCombinedName] = np.concatenate((aXNewTrain1, aXNewTrain2, aXNewTrain3), axis=1)
                dXTest2Atlas[sCombinedName] = np.concatenate((aXNewTest1, aXNewTest2, aXNewTest3), axis=1)

            elif iOuterIndex == iInnerIndex:
                None

    # Save the data for use
    pickle.dump([dXTrain2Atlas, dXTest2Atlas, aYTrain, aYTest],
                open(os.path.join(sProjectRootDirectory, sProjectIdentification, 'TrainTestData2Atlas.p'), 'wb'))


else:
    [dXTrain2Atlas, dXTest2Atlas, aYTrain, aYTest] = pickle.load(open(os.path.join(sProjectRootDirectory,
                                                                       sProjectIdentification, 'TrainTestData2Atlas.p'), 'rb'))
