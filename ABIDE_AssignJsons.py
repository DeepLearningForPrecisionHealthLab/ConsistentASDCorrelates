#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file assigns the correct jsons to the ABIDE data for further processing


Originally developed for use in the ASD comparison project
Created by Cooper Mellema
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

from os import replace
import shutil


def fDefineJsonMapping():
    import glob
    import os
    from pathlib import Path
    dJsonMapping={}
    sA2Raw='/project/bioinformatics/DLLab/STUDIES/ABIDE*/Source/GroupBySite/ABIDEII-*/*task*rest*bold.json'
    sA1Raw='/project/bioinformatics/DLLab/STUDIES/ABIDE*/RawDataBIDS/*/*task-rest*bold.json'
    sA1Raw2='/project/bioinformatics/DLLab/STUDIES/ABIDE*/RawDataBIDS/*/Source/*task-rest*bold.json'
    sA2Raw2='/project/bioinformatics/DLLab/STUDIES/ABIDE*/Source/GroupBySite/ABIDEII-*/Source/*task*rest*bold.json'

    for sA, sRaw in [('ABIDE1', sA1Raw),('ABIDE1', sA1Raw2), ('ABIDE2', sA2Raw), ('ABIDE2', sA2Raw2)]:
        if sA not in dJsonMapping.keys():
            dJsonMapping[sA]={}
        for sFile in glob.glob(sRaw):
            dJsonMapping[sA].update({x.split('sub-')[1]:sFile for x in glob.glob(os.path.join(Path(sFile).parent.as_posix(), 'sub-*'))})

    return dJsonMapping

def fMapJsonToSource():
    import glob
    import shutil
    import os
    sA1Source='/project/bioinformatics/DLLab/STUDIES/ABIDE*/source/sub-*/ses-*/func/*task-rest*bold.nii.gz'
    sA2Source='/project/bioinformatics/DLLab/STUDIES/ABIDE*/Source/ABIDE*-ALL/sub-*/ses-*/func/*task-rest*bold.nii.gz'
    dJsonMapping = fDefineJsonMapping()
    for sSource in [sA1Source, sA2Source]:
        for sFile in glob.glob(sSource):
            if 'ABIDE1' in sFile:
                sA='ABIDE1'
            else:
                sA='ABIDE2'
            sSub = sFile.split('sub-')[1].split('/')[0]
            if sSub in dJsonMapping[sA].keys():
                if not os.path.isfile(sFile.replace('.nii.gz', '.json')):
                    shutil.copy(dJsonMapping[sA][sSub], sFile.replace('.nii.gz', '.json'))
            else:
                print(f'no mapping found for {sA} subject: {sSub}')

def fRenameABIDE1():
    import os
    import glob
    from pathlib import Path
    sAnatNames='/project/bioinformatics/DLLab/STUDIES/ABIDE1/Source_unformatted/ABIDEI-ALL/*/session_*/anat*/mprage.nii.gz'
    sFuncNames='/project/bioinformatics/DLLab/STUDIES/ABIDE1/Source_unformatted/ABIDEI-ALL/*/session_*/rest*/rest.nii.gz'

    # symlink the anatomical files
    sAnatDest='/project/bioinformatics/DLLab/STUDIES/ABIDE1/source/sub-[[sub]]/ses-[[ses]]/anat/[[name]].nii.gz'
    for sAnat in glob.glob(sAnatNames):
        sSub = sAnat.split('ABIDEI-ALL/')[1].split('/')[0]
        sSes = sAnat.split('session_')[1].split('/')[0]
        sFile = sAnatDest.replace('[[sub]]', sSub).replace('[[ses]]', sSes).replace('[[name]]', f'sub-{sSub}_ses-{sSes}_run-1_T1w')
        os.makedirs(Path(sFile).parent.as_posix(), exist_ok=True)
        if not os.path.isfile(sFile):
            os.symlink(sAnat, sFile)

    # symlink the fMRI files
    sFuncDest='/project/bioinformatics/DLLab/STUDIES/ABIDE1/source/sub-[[sub]]/ses-[[ses]]/func/[[name]].nii.gz'
    for sFunc in glob.glob(sFuncNames):
        sSub = sFunc.split('ABIDEI-ALL/')[1].split('/')[0]
        sSes = sFunc.split('session_')[1].split('/')[0]
        sFile = sFuncDest.replace('[[sub]]', sSub).replace('[[ses]]', sSes).replace('[[name]]', f'sub-{sSub}_ses-{sSes}_task-rest_run-{sFunc.split("rest_")[1][0]}_bold')
        os.makedirs(Path(sFile).parent.as_posix(), exist_ok=True)
        if not os.path.isfile(sFile):
            os.symlink(sFunc, sFile)

fRenameABIDE1()
fMapJsonToSource()
