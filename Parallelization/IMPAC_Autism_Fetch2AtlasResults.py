import os
import pandas as pd
import numpy as np
import pickle

# Initialize the path to the data and the indices to loop over
sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/2Atlases/'

lsAtlases = ['basc064', 'basc122', 'basc197', 'craddock_scorr_mean',
             'harvard_oxford_cort_prob_2mm', 'msdl', 'power_2011']

# Create a list of atlas combinations
lsAtlasCombos=list()
for iOuterIndex, sAtlas1 in enumerate(lsAtlases):
    for iInnerIndex, sAtlas2 in enumerate(lsAtlases):
        if not iOuterIndex==iInnerIndex:
            lsAtlasCombos.extend([sAtlas1+'_'+sAtlas2])

# Loop through all Cross Validation Results and re-save the mean ROC AUC
for i in range(50):

    # Get model number tag
    if i<10:
        sModelNum='0'+ str(i)
    else:
        sModelNum=str(i)

    for sAtlasCombo in lsAtlasCombos:

        if os.path.isfile(sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal1.p') and os.path.isfile(sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal2.p') and os.path.isfile(sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal3.p'):
            CV1=pickle.load(open((sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal1.p'), 'rb'))
            CV2=pickle.load(open((sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal2.p'), 'rb'))
            CV3=pickle.load(open((sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasROCScoreCrossVal3.p'), 'rb'))

            CVMean=(CV1+CV2+CV3)/3.0
            pickle.dump(CVMean, open((sDataPath + 'Dense_' + sModelNum + sAtlasCombo + '2AtlasMeanROCScoreCrossVal.p'), 'wb'))

        if os.path.isfile(sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal1.p') and os.path.isfile(sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal2.p') and os.path.isfile(sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal3.p'):
            CV1=pickle.load(open((sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal1.p'), 'rb'))
            CV2=pickle.load(open((sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal2.p'), 'rb'))
            CV3=pickle.load(open((sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasROCScoreCrossVal3.p'), 'rb'))

            CVMean=(CV1+CV2+CV3)/3.0
            pickle.dump(CVMean, open((sDataPath + 'Dense_' + sModelNum + 'anatomical_' + sAtlasCombo + '2AtlasMeanROCScoreCrossVal.p'), 'wb'))


# Initialize a dataframe to hold results
pdResults=pd.DataFrame(index=lsAtlases, columns=lsAtlases)

for root, dirs, files in os.walk(sDataPath):
    files.sort()
    for iOuterIndex, sAtlas1 in enumerate(lsAtlases):
        for iInnerIndex, sAtlas2 in enumerate(lsAtlases):
            if not iOuterIndex == iInnerIndex:
                flMax = 0
                for file in files:
                    if file.endswith('anatomical_'+sAtlas1+'_'+sAtlas2+'2AtlasMeanROCScoreCrossVal.p'):
                        flMeanVal=pickle.load(open(os.path.join(sDataPath, file), 'rb'))
                        if flMeanVal>flMax:
                            flMax=flMeanVal
                            sBest=file[8:]
                if flMax>0:
                    pdResults.loc[sAtlas2, sAtlas1]=flMax
                    print(sBest, flMax)

#############Non-Anatomical in lower triangle########################
                flMax=0
                for file in files:
                    if (file.endswith(sAtlas1+'_'+sAtlas2+'2AtlasMeanROCScoreCrossVal.p')) and (file.find('anatomic')==-1):
                        flMeanVal=pickle.load(open(os.path.join(sDataPath, file), 'rb'))
                        if flMeanVal>flMax:
                            flMax=flMeanVal
                            sBest=file[8:]
                if flMax>0:
                    pdResults.loc[sAtlas1, sAtlas2]=flMax
                    print(sBest, flMax)

pdMultipleAtlases=pdResults
pdMultipleAtlases.to_csv('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/2Atlases/2AtlasBestModelSummaryAnatUpperTirang')
