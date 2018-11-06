import os
import pandas as pd
import numpy as np
import pickle

# Initialize the path to the data and the indices to loop over
sDataPath = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/TrainedModels/Stacked/'

# Loop through all Cross Validation Results and re-save the mean ROC AUC
for i in range(50):

    # Get model number tag
    if i<10:
        sModelNum='0'+ str(i)
    else:
        sModelNum=str(i)

    if os.path.isfile(sDataPath + 'Stack_' + sModelNum + 'CrossVal0ModelROCScore.p')\
            and os.path.isfile(sDataPath + 'Stack_' + sModelNum + 'CrossVal1ModelROCScore.p')\
            and os.path.isfile(sDataPath + 'Stack_' + sModelNum + 'CrossVal2ModelROCScore.p'):
        CV1=pickle.load(open((sDataPath + 'Stack_' + sModelNum + 'CrossVal0ModelROCScore.p'), 'rb'))
        CV2=pickle.load(open((sDataPath + 'Stack_' + sModelNum + 'CrossVal1ModelROCScore.p'), 'rb'))
        CV3=pickle.load(open((sDataPath + 'Stack_' + sModelNum + 'CrossVal2ModelROCScore.p'), 'rb'))

        CVMean=(CV1+CV2+CV3)/3.0
        pickle.dump(CVMean, open((sDataPath + 'Stack_' + sModelNum + 'MeanCVROCScore.p'), 'wb'))

# Walk through the directory and find the model with the highest ROC score
flMax = 0
sBest = 'run failed'

for root, dirs, files in os.walk(sDataPath):
    files.sort()

    for file in files:
        if file.endswith('MeanCVROCScore.p'):
            flMeanVal=pickle.load(open(os.path.join(sDataPath, file), 'rb'))
            if flMeanVal>flMax:
                flMax=flMeanVal
                sBest=file

print(sBest, flMax)
