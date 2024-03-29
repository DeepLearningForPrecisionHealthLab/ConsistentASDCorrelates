#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This has functions to create glass brain connectivity images from lists of important features

This file, if run alone, will create images depicting the top features by permutation feature
 importance testing on the top 5 TSE models in the IMPAC study. The featues are depicted on a glass brain
 representation of the MNIST standard brain

Originally developed for use in the ASD comparison project
Created by Cooper Mellema on 17 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

import os
import nilearn
import numpy as np
import pandas as pd
import pickle as pkl
import math
import re
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiSpheresMasker

def fFindCentroids(sAtlas):
    # Find the centroids of each ROI for connectome plotting
    masker = NiftiLabelsMasker(labels_img=sAtlas, standardize=True)
    aCoordinates=nilearn.plotting.find_parcellation_cut_coords(labels_img=sAtlas)
    return aCoordinates

def fPlotROIs(sAtlas):
    # plot the ROIs of the BASC atlas
    plotting.plot_roi(sAtlas, cmap=plotting.cm.bwr)

def fSelectROIs(pdImportances, flThreshold, iTop):
    """
    Select all features above a threshold, if threshold is 0, select only the top 6
    :param pdImportances:
    :param flThreshold:
    :return:
    """
    #Select out ROI features
    pdROIImportances=pdImportances.drop((x for x in pdImportances.index if not x.__contains__('ROI')), axis=0)
    pdROIImportances=pdROIImportances.sort_values(['Importance'], ascending=False)

    # fill out an array with the importances above a threshold in the corresponding spots
    iLen=int((-1+(1+8*len(pdROIImportances.index))**(1/2))/2)
    aConnections=np.zeros((iLen, iLen))

    if flThreshold==0:
        # if threshold is 0, select only the top iTop
        pdROIImportances=pdROIImportances.head(iTop)

        # loop through importances and fill in matrix where importance > a threshold
        for sIndex in pdROIImportances.index:
            lsROIs=re.findall('\d+', sIndex)
            iRow=int(int(lsROIs[0])-1)
            iCol=int(int(lsROIs[1])-1)
            aConnections[iRow, iCol]=pdROIImportances.loc[sIndex].values[0]
    else:
        for iRow in range(iLen):
            for iCol in range(iLen):
                if iRow<=iCol:
                    if f'ROI{iRow+1:03}-ROI{iCol+1:03}' in pdROIImportances.index:
                        if pdROIImportances.loc[f'ROI{iRow+1:03}-ROI{iCol+1:03}'].values[0]>=flThreshold:
                            aConnections[iRow, iCol]=pdROIImportances.loc[f'ROI{iRow+1:03}-ROI{iCol+1:03}'].values[0]

    # Make the connections symmetric
    for iRow in range(iLen):
        for iCol in range(iLen):
            aConnections[iCol, iRow] = aConnections[iRow, iCol]

    return aConnections

def fMatchCoordinates(aConnections, aCoordinates):
    # Get coordinates in atlas space of the selected coordinates
    aUnique=np.unique(np.append(np.nonzero(aConnections)[0],np.nonzero(aConnections)[1]))
    aConnectionsNonzero=aConnections[aUnique,:]
    aConnectionsNonzero=aConnectionsNonzero[:,aUnique]

    lsNonzeroCoordinates=aCoordinates[aUnique]

    return lsNonzeroCoordinates, aConnectionsNonzero

def fPlotFinal(aConnections, lsNonzeroCoordinates, sSaveDir, iModel, iPermutations, flThresh):
    # Make the final plot
    cDisplay=plotting.plot_connectome(aConnections, lsNonzeroCoordinates,
                                      edge_vmin=5,
                                      edge_vmax=8,
                                      #node_color='k',
                                      colorbar=True,
                                      edge_cmap="rainbow",
                                      alpha=0.5,
                                      title='Plot')

    cInteractiveDisplay=plotting.view_connectome(aConnections, lsNonzeroCoordinates)
    dKey = {1: '64', 2: '122', 3: '197'}
    sDir = os.path.join(sSaveDir, f'{dKey[iModel]}ROI_Biomarkers{iPermutations}Permutations')
    print(sDir)
    cDisplay.savefig(f'{sDir}.png')
    plt.savefig(f'{sDir}.png')
    cInteractiveDisplay.save_as_html(f'{sDir}.html')
    plt.close()
    del(cDisplay)
    del(cInteractiveDisplay)

def fFormatImportances(dImportances):
    # make into pd dataframe
    pdImportances=pd.DataFrame.from_dict(dImportances, orient='index', columns=['Importance'])

    # Standard scale the importances
    pdImportances = pdImportances.subtract(pdImportances.mean(axis=0)[0]).divide(pdImportances.std(axis=0)[0])

    return pdImportances

def fBarPlotImportances(pdImportances, sDir, iPermutations, iModel, sType='Functional'):
    pdImportances=pdImportances.sort_values(by=['Importance'], axis=0, ascending=False)
    if sType=='Functional':
        pdImportances = pdImportances.loc[[idx for idx in pdImportances.index if 'ROI' in idx.upper()]]
    elif sType=='All':
        pass
    elif sType=='Structural':
        pdImportances = pdImportances.loc[[idx for idx in pdImportances.index if 'anatomy' in idx.lower()]]
    elif sType=='Confounds':
        pdImportances = pdImportances.loc[[idx for idx in pdImportances.index if not 'anatomy' in idx.lower()]]
        pdImportances = pdImportances.loc[[idx for idx in pdImportances.index if not 'ROI' in idx.upper()]]

    fig, ax=plt.subplots()
    pdImportances.head(15).plot(kind='barh', ax=ax)

    dKey = {1: '64', 2: '122', 3: '197'}
    ax.set_xlabel='Importance (\u03C3 from mean feature importance)'
    plt.tight_layout()
    plt.savefig(os.path.join(sDir, f'{dKey[iModel]}ROI_{sType}Biomarkers{iPermutations}PermutationsBarplot.png'))
    plt.close()

def fProcessModel(iModel, sAtlas, iPermutations, sDir, flThresh, iTop):
    # Load importances
    #dImportances = pkl.load(open(os.path.join(sDir, f'Model{iModel}Importances.p'), 'rb'))
    dKey={1:'64', 2:'122', 3:'197'}
    dImportances = pkl.load(open(os.path.join(sDir, f'{dKey[iModel]}ROI_MeanImportances.p'), 'rb'))
    pdImportances = fFormatImportances(dImportances)
    for sType in ['All', 'Functional', 'Structural', 'Confounds']:
        fBarPlotImportances(pdImportances, sDir, iPermutations, iModel, sType=sType)

    # make plots
    if not sAtlas=='anat':
        aCoordinates = fFindCentroids(sAtlas=sAtlas)
        lsNonzeroCoordinates=[]
        aConnectionsNonzero=[0]
        aConnections = fSelectROIs(pdImportances=pdImportances, flThreshold=flThresh, iTop=iTop)
        lsNonzeroCoordinates, aConnectionsNonzero = fMatchCoordinates(aConnections, aCoordinates)
        # while (len(np.array(np.nonzero(aConnectionsNonzero))[0])<iTop*2):
        #     aConnections = fSelectROIs(pdImportances=pdImportances, flThreshold=flThresh, iTop=iTop)
        #     lsNonzeroCoordinates, aConnectionsNonzero = fMatchCoordinates(aConnections, aCoordinates)
        #     flThresh=flThresh-1


        fPlotFinal(aConnectionsNonzero, lsNonzeroCoordinates, sDir, iModel, iPermutations, flThresh+1)

if __name__=='__main__':
    # Fetch the atlas that will be used in this case
    dBASC = datasets.fetch_atlas_basc_multiscale_2015(version='sym')

    # Set atlases per model
    dModelAtlases={
        1:'scale064',
        2:'scale122',
        3:'scale197'
        # 4:'anat'
    }

    for iModel, sAtlasKey in dModelAtlases.items():
        # set atlas
        if not sAtlasKey=='anat':
            sAtlas = dBASC[sAtlasKey]
        else:
            sAtlas='anat'

        # Process the model and make the glass brain
        iPermutations=8
        sDir = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison' \
            f'{iPermutations}Permutations'
        fProcessModel(iModel, sAtlas, iPermutations, sDir=sDir, flThresh=5, iTop=26)
