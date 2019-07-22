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

def fSelectROIs(pdImportances, flThreshold):
    #Select out ROI features
    pdROIImportances=pdImportances.drop((x for x in pdImportances.index if not x.__contains__('ROI')), axis=0)
    pdROIImportances=pdROIImportances.sort_values(['Importance'], ascending=False)

    # fill out an array with the importances above a threshold in the corresponding spots
    iLen=int((-1+(1+8*len(pdROIImportances.index))**(1/2))/2)
    aConnections=np.zeros((iLen, iLen))

    # loop through importances and fill in matrix where importance > a threshold
    for iRow in range(iLen):
        for iCol in range(iLen):
            if iRow<=iCol:
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
                                      edge_vmin=flThresh,
                                      edge_vmax=int(math.ceil(np.amax(aConnections))),
                                      colorbar=True,
                                      edge_cmap="jet",
                                      alpha=0.5,
                                      title='Plot')

    cInteractiveDisplay=plotting.view_connectome(aConnections, lsNonzeroCoordinates)

    cDisplay.savefig(os.path.join(sSaveDir, f'Model{iModel}Biomarkers{iPermutations}Permutations.png'))
    cInteractiveDisplay.save_as_html(os.path.join(sSaveDir, f'Model{iModel}Biomarkers{iPermutations}Permutations.html'))
    plt.close()
    del(cDisplay)
    del(cInteractiveDisplay)

def fFormatImportances(dImportances):
    # make into pd dataframe
    pdImportances=pd.DataFrame.from_dict(dImportances, orient='index', columns=['Importance'])

    # Standard scale the importances
    pdImportances = pdImportances.subtract(pdImportances.mean(axis=0)[0]).divide(pdImportances.std(axis=0)[0])

    return pdImportances

def fBarPlotImportances(pdImportances, sDir, iPermutations):
    pdImportances=pdImportances.sort_values(by=['Importance'], axis=0, ascending=False)
    fig, ax=plt.subplots()
    pdImportances.head(15).plot(kind='barh', ax=ax)
    ax.set_xlabel='Importance (\u03C3 from mean feature importance)'
    plt.tight_layout()
    plt.savefig(os.path.join(sDir, f'Model{iModel}Features{iPermutations}Permutations.png'))
    plt.close()

def fProcessModel(iModel, sAtlas, iPermutations):
    # Load importances
    sDir = f'/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AlternateMetrics/{iPermutations}Permutations'
    dImportances = pkl.load(open(os.path.join(sDir, f'Model{iModel}Importances.p'), 'rb'))
    pdImportances = fFormatImportances(dImportances)
    fBarPlotImportances(pdImportances, sDir, iPermutations)

    # make plots
    aCoordinates = fFindCentroids(sAtlas=sAtlas)
    lsNonzeroCoordinates=[]
    flThresh=6
    while (f'{lsNonzeroCoordinates}'==f'{[]}' or len(np.array(np.nonzero(aConnectionsNonzero)))==1):
        aConnections = fSelectROIs(pdImportances=pdImportances, flThreshold=flThresh)
        lsNonzeroCoordinates, aConnectionsNonzero = fMatchCoordinates(aConnections, aCoordinates)
        flThresh=flThresh-1

    fPlotFinal(aConnectionsNonzero, lsNonzeroCoordinates, sDir, iModel, 5, flThresh+1)

if __name__=='__main__':
    # Fetch the atlas that will be used in this case
    dBASC = datasets.fetch_atlas_basc_multiscale_2015(version='sym')

    # Set atlases per model
    dModelAtlases={
        1:'scale122',
        2:'scale197',
        3:'scale122',
        4:'scale122',
        5:'scale122'
    }

    for iModel, sAtlasKey in dModelAtlases.items():
        # set atlas
        sAtlas = dBASC[sAtlasKey]

        # Process the model and make the glass brain
        fProcessModel(iModel, sAtlas, 5)
