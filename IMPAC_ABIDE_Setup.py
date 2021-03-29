#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file prepares the ABIDE I and II data for use in the IMPAC modeling pipeline

This file Fetches the necessary atlases for using the IMPAC parcellations,
saves them, puts them in the right spot, and starts the processing pipeline

Originally developed for use in the Autism ML comparison project
Created by Cooper Mellema on 26 Jul, 2019
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"

__status__ = "Prototype"

import nibabel as nib
from nilearn import datasets as nildata

if __name__ == '__main__':
    # Fetch basc atlas, craddock atlas, harvard-oxford atlas, msdl atlas, power atlas
    cBasc=nildata.fetch_atlas_basc_multiscale_2015()
    cCraddock=nildata.fetch_atlas_craddock_2012()
    cHarvOx=nildata.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm')
    cMSDL=nildata.fetch_atlas_msdl()
    cPower=nildata.fetch_coords_power_2011()

    # Get the .nii.gz locations
    sBasc064=cBasc['scale064']
    sBasc122=cBasc['scale122']
    sBasc197=cBasc['scale197']
    sCraddock=cCraddock['tcorr_mean']
    sHarvOx=cHarvOx['maps']
    sMSDL=cMSDL['maps']
    aPowerPoints=cPower['rois']
