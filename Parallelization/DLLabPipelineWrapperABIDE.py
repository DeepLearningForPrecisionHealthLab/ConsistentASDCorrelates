"""
Script that does all preparation steps necessary to run DLLabPipeline

_______
04/08/2019
"""
__author__ = 'Vyom M Raval'
__email__ = 'Vyom.Raval@UTSouthwestern.edu'

import subprocess
import os
import pandas as pd
import glob
import json

def fRenameABIDEData(sDataDir, iABIDE=1):
    """
    Renames the native ABIDE I or II data to the correct BIDS format
    :param sDataDir: string, path to data directory
    :param iAbide: int, 1 or 2 for abide I or II respectively
    :return: none, modifies file names in place
    """
    # make list of subjects in data directory
    lsSubjects = glob.glob(sDataDir+'*')
    # exclude metadata
    lsSubjects.remove(os.path.join(sDataDir,'dataset_description.json'))
    # for each subject
    for sSubject in lsSubjects:
        sSubject=sSubject.split('/')[-1]
        # make bids compliant name and rename
        if not 'sub' in sSubject:
            sNewName=f'sub-{sSubject.split("/")[-1]}'
        else:
            sNewName=sSubject
            sSubject=sSubject.split("-")[-1]
        sNewPath=os.path.join(sDataDir, sNewName)
        if not os.path.isdir(sNewPath):
            os.rename(os.path.join(sDataDir, sSubject), sNewPath)
        lsSessions=glob.glob(sNewPath+'/*')
        sSubPath=sNewPath

        # for each session for a subject
        for sSession in lsSessions:
            sSession = sSession.split('/')[-1]
            # make bids compliant name and rename
            if 'ion' in sSession:
                sNewName=f'ses-{sSession.split("_")[-1]}'
            else:
                sNewName=sSession
                sSession=sSession.split("-")[-1]
            sNewPath=os.path.join(sSubPath, sNewName)
            if not os.path.isdir(sNewPath):
                os.rename(os.path.join(sSubPath, sSession), sNewPath)
            lsModalities=glob.glob(sNewPath+'/*')
            sSessPath=sNewPath

            # for each imaging modality for a session
            for sModality in lsModalities:
                sModality = sModality.split('/')[-1]
                #If one run of scan, label as 1
                if sModality.__contains__('_'):
                    sModality,iTrial=sModality.split('_')
                else:
                    iTrial=1
                # make bids compliant name and rename
                if 'anat' in sModality:
                    sNewName='anat'
                elif ('rest' in sModality or 'func' in sModality):
                    sNewName='func'
                sNewPath = os.path.join(sSessPath, sNewName)
                if not os.path.isdir(sNewPath):
                    os.rename(os.path.join(sSessPath, f'{sModality}_{iTrial}'), sNewPath)
                lsScans = glob.glob(sNewPath + '/*')
                sModePath=sNewPath

                # for each scan per imaging modality
                for sScan in lsScans:
                    sScan=sScan.split('/')[-1]
                    if '.nii.gz' in sScan:
                        # make bids compliant name
                        if 'mprage' in sScan:
                            sNewName=f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_T1w.nii.gz'
                        elif 'test' in sScan:
                            sNewName =f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_test.nii.gz'
                        elif 'rest' in sScan:
                            sNewName = f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_task-rest_run-{iTrial}_bold.nii.gz'
                            sJsonName=f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_task-rest_run-{iTrial}_bold.json'
                            # write Json file with metadata
                            fWriteJson(os.path.join(sModePath,sJsonName), sSubject, iTrial, ABIDE=iABIDE)
                        elif 'dwi' in sScan:
                            sNewName = f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_dwi.nii.gz'
                            #write json file with metadata
                            sJsonName = f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_dwi.json'
                            fWriteDWIJson(os.path.join(sModePath,sJsonName), sSubject, iTrial, ABIDE=iABIDE)

                        # take new file name and change the old name to new
                        sNewPath = os.path.join(sModePath, sNewName)
                        if not os.path.isfile(sNewPath):
                            os.rename(os.path.join(sModePath, sScan), sNewPath)

                    # repeat above for .bval and .bvec files
                    elif '.bval' in sScan:
                        sNewName = f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_dwi.bval'
                        sNewPath = os.path.join(sModePath, sNewName)
                        if not os.path.isfile(sNewPath):
                            os.rename(os.path.join(sModePath, sScan), sNewPath)
                    elif '.bvec' in sScan:
                        sNewName = f'sub-{sSubject}_ses-{sSession.split("_")[-1]}_run-{iTrial}_dwi.bvec'
                        sNewPath = os.path.join(sModePath, sNewName)
                        if not os.path.isfile(sNewPath):
                            os.rename(os.path.join(sModePath, sScan), sNewPath)

def fWriteJson(sJsonName, sSubject, iTrial, ABIDE=1):
    """
    Writes the BIDS formatted .json file
    :param sJsonName: filepath to .json file (with .json extension)
    :param ABIDE: 1 or 2 for ABIDE 1 or abide 2
    :return: none, writes file to sJsonName
    """
    sSubject=sSubject.lstrip('0')
    pdMetadata=fGetSubjectMetadata(ABIDE=ABIDE)
    pdMetadata=pdMetadata.loc[int(sSubject)]

    if len(pdMetadata['TE (ms)'])>0:
        flTE=float(pdMetadata['TE (ms)'])
    else:
        flTE="NA"

    dJson={
        "Manufacturer": pdMetadata['Manufacturer'],
        "ManufacturersModelName": pdMetadata['Scanner Type'],
        "AcquisitionNumber": iTrial,
        "InstitutionName": pdMetadata['site'],
        "DeviceSerialNumber": "NA",
        "SoftwareVersions": "",
        "ScanningSequence": 'Gradient Echo',
        "SequenceVariant": 'Reverse Spiral Acquisition',
        "SeriesDescription": f"ABIDE{ABIDE}",
        "ProtocolName": "ABIDE",
        "ImageType": ["ORIGINAL"],
        "AcquisitionTime": "NA",
        "MagneticFieldStrength": pdMetadata['Field strength'],
        "FlipAngle": pdMetadata['Flip Angle'],
        "EchoTime": flTE,
        "RepetitionTime": pdMetadata['TR (s)'],
        "PhaseEncodingLines": pdMetadata['No. Slices'],
        "ConversionSoftware": "NA",
        "ConversionSoftwareVersion": "NA"
    }

    with open(sJsonName, 'w') as outfile:
        json.dump(dJson, outfile)

def fWriteDWIJson(sJsonName, sSubject, iTrial, ABIDE=1):
    """
    Writes the BIDS formatted .json file for dwi MRI
    :param sJsonName: filepath to .json file (with .json extension)
    :param ABIDE: 1 or 2 for ABIDE 1 or abide 2
    :return: none, writes file to sJsonName
    """

    dJson = {
        "AcquisitionNumber": iTrial,
        "SeriesDescription": f"ABIDE{ABIDE}",
        "ProtocolName": "ABIDE",
    }

    with open(sJsonName, 'w') as outfile:
        json.dump(dJson, outfile)

def fGetSubjectMetadata(ABIDE=1):
    #Load list of subjects matched to imaging site
    if ABIDE==1:
        sSubjectSiteDir="/project/bioinformatics/DLLab/STUDIES/ABIDE1/PreprocessedABIDE1" \
           "/TheirPreprocessedResults/Phenotypic_V1_0b_preprocessed1.csv"
        pdSubjectSite=pd.DataFrame.from_csv(sSubjectSiteDir)
        pdSubjectSite=pdSubjectSite[['subject', 'SITE_ID']].set_index('subject')
        pdSubjectSite=pdSubjectSite.rename(columns={"SITE_ID": "site"})

        # Load list of parameters per site
        sSiteParamDir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDETest/SiteParameters.csv"

    elif ABIDE==2:
        # get list of subjects
        lsSubjects=glob.glob("/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/ABIDEII-ALL/*")
        for iSub in range(len(lsSubjects)):
            lsSubjects[iSub] = lsSubjects[iSub].split('-')[2]

        # get a list of sites
        lsSites=glob.glob("/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/GroupBySite/*")
        for iSite in range(len(lsSites)):
            lsSites[iSite] = lsSites[iSite].split('/')[-1].split('-')[-1]

        #make dataframe
        pdSubjectSite=pd.DataFrame(index=lsSubjects, columns=['site'])

        # match subject to site
        for sSub in lsSubjects:
            for sSite in lsSites:
                if os.path.isdir(f"/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/GroupBySite/ABIDEII-{sSite}/sub-{sSub}"):
                    pdSubjectSite.loc[sSub, 'site']=sSite

        # set site parameter directory
        sSiteParamDir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDEIITest" \
                      "/SiteParameters.csv"

    pdSiteParam=pd.DataFrame.from_csv(sSiteParamDir).transpose()

    # Add new columns
    for sColumn in pdSiteParam.columns:
        pdSubjectSite[sColumn]=""

    #Rename sites
    for sSub in pdSubjectSite.index:
        if '_' in pdSubjectSite.loc[sSub]['site']:
            if pdSubjectSite.loc[sSub]['site']=='MAX_MUN':
                pdSubjectSite.loc[sSub]['site']='MAXMUN'
            else:
                pdSubjectSite.loc[sSub]['site']=pdSubjectSite.loc[sSub]['site'].split('_')[0]

    #loop through and populate the columns based on the site
    for sSite in pdSiteParam.index:
        for sColumn in pdSiteParam.columns: \
    pdSubjectSite.loc[pdSubjectSite['site']==sSite.upper(),sColumn]=pdSiteParam.loc[sSite][sColumn]


    return pdSubjectSite

################################### user editing zone starts ##########################################

# Path to local clone of DLLabPipeline
strDllabPipelineDir = '/project/bioinformatics/DLLab/Cooper/Code/nipype-pre-processing-pipeline'

# Conda environment to use
strConda = '/project/bioinformatics/DLLab/shared/distribution_dev/DLLabPipelineV1.1'

# Optional path to csv file for reading in subject names and sessions, leave empty if you want to list subjects and sessions below
strSubjectInfoCsv = ''
strCsvSubjectCol = 'Subject ID'  # Name of column for subject ID
strCsvSessionCol = 'Session'  # Name of column for session

# Global variables common to CONN and DLLabPipeline
strRootDir = '/project/bioinformatics/DLLab/STUDIES/ABIDE2/'
strStudyName = ''
strDerivativeName = 'DLLabPipeline'
strJobName = 'ABIDE2_NYU2'
strDataDir = '/project/bioinformatics/DLLab/STUDIES/ABIDE2/Source/GroupBySite/ABIDEII-NYU_2'
bCopy = False  # if False, don't copy subjects at all
bAllSubjects = False  # if True, copy all subjects from strDataDir, else copy subjects listed in lsSubjects


####### Variables for Config Files #######
# Variables in ProcessingOptions.ini
lsSubjects = -1  # Can be a list of subjects or -1 for all subjects in the Data Directory
lsSessions = -1  # Can be a list of sessions, one for each subject, or -1 to process all sessions
bDicomConversion = False
bCat12 = True
bFreeSurfer = True
bfMRI = True
bfMRIpre = True
nFWHM = 6
strEPItemplate = os.path.join(strDllabPipelineDir, 'templates',
                              'tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref_brain_resamp_nn.nii.gz')
bT1Norm = True
bTaskBased = False
bRestingState = True
strDenoisingPipeline = 'all'
bGiftIca = False
nIcaDims = 20
bCONN = True
strConnInput = 'motion'
bConnDenoising = True
lsConnAtlasNames =[
    'MSDL',
    'BASC_064',
    'BASC_122',
    'BASC_197',

    'HarvardOxford',
    'Craddock'
]
lsConnAtlasPaths =[
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/msdl_atlas/MSDL_rois'
        '/msdl_rois.nii',
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/basc_multiscale_2015'
        '/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale064.nii.gz',
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/basc_multiscale_2015'
        '/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale122.nii.gz',
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/nilearn_data'
        '/basc_multiscale_2015/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale197.nii.gz',
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/nilearn_data/fsl/data'
        '/atlases/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz',
    '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_Atlases/nilearn_data/craddock_2012/scorr05_mean_all.nii.gz'
]

for iAtlasNum in range(len(lsConnAtlasNames)):
    if iAtlasNum==0:
        sConnAtlases=f'{lsConnAtlasNames[iAtlasNum]}'
    else:
        sConnAtlases=f'{sConnAtlases}\n 			  {lsConnAtlasNames[iAtlasNum]}'
    sConnAtlases=f'{sConnAtlases}\n			  {lsConnAtlasPaths[iAtlasNum]}'

bDMRI = False


# Variables in param_runner.yaml
nNodes = 10
nCpu = 4  # cpus_per_task in param_runner.yaml
strTimeLimit = '20-4:15:00'
nMaxSubjects = len(glob.glob(f'{strDataDir}/sub*'))  # Total number of subjects to be processed

#################################### user editing zone ends ###########################################

# Derived variables
strStudyDir = os.path.join(strRootDir, strStudyName)
strJobDir = os.path.join(strStudyDir, 'Derivatives', strDerivativeName, 'BatchRuns', strJobName)

# Create BIDS format directory
if not os.path.isdir(strJobDir):
    os.makedirs(strJobDir)
    os.makedirs(os.path.join(strJobDir, 'log'))

# if lsSubjects=-1, make list of all
if lsSubjects == -1:
    lsSubjects = []
    for sRoot, lsDirs, lsFiles in os.walk(strDataDir):
        for sDir in lsDirs:
            if 'sub-' in sDir:
                lsSubjects.append(os.path.join(sRoot, sDir))

# Create list of subject info
if not strSubjectInfoCsv:
    # if a csv file doesn't exist
    strSubjectList = ''
    # or those in prespecified list
    for subject in lsSubjects:
        if subject.__contains__('dataset_description'):
            pass
        else:
            strSubjectList += os.path.join(strStudyDir, 'Data', subject) + '\n\t\t  '

    # turn session list into string
    strSessionList = lsSessions
    if isinstance(lsSessions, list):
        strSessionList = ''
        for session in lsSessions:
            strSessionList += session + '\n\t   '


# Copy subjects into Data folder
if bCopy:
    # copy all if bAllSubjects
    if bAllSubjects:
        subprocess.call(['cp', '-r', strDataDir, os.path.join(strStudyDir, 'Data'), '--verbose'])
    # else copy only those in lsSubjects
    else:
        # else use only those in the subject list
        for subject in lsSubjects:
            subprocess.call(
                ['cp', '-r', os.path.join(strDataDir, subject), os.path.join(strStudyDir, 'Data'), '--verbose'])
# load csv if it exists
elif os.path.isfile(strSubjectInfoCsv):
    dfSubjectInfo = pd.read_csv(strSubjectInfoCsv, index_col=0)
    strSubjectList = ''
    # turn pd index into list
    for subject in dfSubjectInfo[strCsvSubjectCol]:
        strSubject = str(subject)
        if 'sub-' not in strSubject:
            strSubject = 'sub-' + strSubject
        strSubjectList += os.path.join(strStudyDir, 'Data', strSubject) + '\n\t\t  '
    # reformat to string
    strSessionList = ''
    for strSession in dfSubjectInfo[strCsvSessionCol]:
        strSessionList += strSession + '\n\t   '

# Create and edit processingOptions.ini file
with open(os.path.join(strJobDir, 'processingOptions_' + strJobName + '.ini'), 'w') as objFile:
    objFile.write(f'''; BIDs folders to use
[Directories]
Study_dir={strStudyDir}
Job_dir={strJobDir}

[LogFiles]
log_directory={os.path.join(strJobDir, 'log')}
csv_log={os.path.join(strJobDir, 'log', 'output_result.csv')}

;Subject sessions
[SubjectList]
Filepaths={strSubjectList}

; If each subject contains multiple sessions, which session to process. Can be a single session ID, a list, or -1 to
; run all sessions
[SessionList]
Sessions={strSessionList}

; Standard resource files: e.g. atlases, source paths
[Resources]
; dicom2bds requires the conda environment "DLLabPipelineEnv"
;nipype, cpac, fmriprep
conda_env={strConda}
; path to SPM12
spm12_root_dir=/project/bioinformatics/DLLab/softwares/spm_files/spm12
; Path to the MNI152 T1 Template
mni152T1= /project/bioinformatics/DLLab/shared/Atlases/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii
; Name of the directory of Job (Derivatives/BatchRuns/"JobName") used for the log files


;Dicom to nifti to Bids conversion Sub-Pipeline options
[DicomConversion]
Do_processing= {bDicomConversion} ;[True: Do DICOM conversion/ False: Skip DICOM conversion]
Method= dcm2bids
; heuristics for mapping nii to MRI contrast type
Conversion_config_file=/project/bioinformatics/DLLab/distribution/nipype-pre-processing-pipeline/config_files/dcm2bids_config.json ;[Required for dcm2bids]File path to the heuristic conversion config json file
; TBD a new name for the folder of uncategorized nii files. Should we rename temporary as uncategorised??
;Clean_up_folder_name=/project/radiology/ANSIR_lab/s167746/data/Volumes/Imaging/uncategorised

; Structural image preprocessing Sub-pipeline Cat12 options
[Cat12]
Do_cat12_processing= {bCat12} ; [True: Do Structural preprocessing/ False: Skip Structural image preprocessing]
;Folder name assigned to the directory where the output will be stored E.g. Derivatives/DLLabPipeline/Subject01/"Cat12"
Derivative_folder_name= CAT12

; Structural image preprocessing Sub-pipeline Freesurfer options
[FreeSurfer]
Do_freesurfer_processing= {bFreeSurfer} ; [True: Do Structural preprocessing/ False: Skip Structural image preprocessing]
;Folder name assigned to the directory where the output will be stored E.g. Derivatives/DLLabPipeline/Subject01/"FreesDerivatives/CONN/BatchRuns/testconn/urfer"
Derivative_folder_name= Freesurfer

; fMRI sub-pipeline options
[fMRI]
; True: Do functional image preprocessing/ False: Skip functional image preprocessing
Do_processing= {bfMRI}
; Number of parallel threads per subject. Increasing this can greatly speed up the ANTS normalization steps.
Parallel_threads=8
; True: Ignore cached intermediate output files from previous runs and restart all processing steps
Ignore_cached=False
; Folder name assigned to the directory where the output will be stored E.g. Derivatives/DLLabPipeline/Subject01/fmri
Derivative_folder_name= fmri

; FMRI PREPROCESSING SPECIFIC OPTIONS
; True: Do preprocessing steps, False: skip preprocessing (if was already done previously)
Do_preprocessing={bfMRIpre}
; FWHM in mm for smoothing filter
FWHM={nFWHM}
; Path to T1 template for spatial normalization. Set to "Default" to use the standard MNI152 template. Unused if
; Do_T1_norm is set to False
T1_template=Default
; Path to T1 template brain mask. Required if a custom T1 template is used. Otherwise, set to "Default"
T1_template_mask=Default
; Path to EPI template for direct EPI-based spatial normalization. Set to "Default" to use the standard MNI EPI
; template.
EPI_template={strEPItemplate}
; True: Do two-step spatial norm, involving EPI to T1 coregistration and T1-based normalization
; False: Do direct EPI-based spatial normalization, involving coregistration of the EPI image to the EPI template
Do_T1_norm={bT1Norm}
; True: Brain-mask the subject's EPI image before performing normalization. If used with EPI-based normalization,
; the template should already be brain-masked as well (this is done automatically if using the default EPI template).
Mask_EPI=True

; MOTION CORRECTION OPTIONS
; True: Do resting-state preprocessing (motion correction)
Motion_correction={bRestingState}
; Method for ICA-AROMA. Valid options are aggr, nonaggr, and both
AROMA_method=both
; Maximum intensity value for MinMax normalization
Norm_units=1000
; Use first temporal derivative terms in nuissance regression
Regressor_derivatives=True
; Linearly detrend the nuissance regressors
Regressor_detrend=True
; Use squared terms in nuissance regression
Regressor_squared=True

; TASK-BASED FMRI SPECIFIC OPTIONS
; True: Do task-based preprocessing (GLM analysis)
Task_based={bTaskBased}
; 'motion' will run on all motion denoided images while'preproc' will only run on the preprocessed image
Task_input_image={strConnInput}
; Mask to use for GLM analysis. 'Default' uses the EPI template brain mask, 'subject' uses the GM+WM segmentation
; generated from the anatomical image. Or give a full path to a mask file. 
Glm_mask=Default
; Highpass filter cutoff in seconds. Typically, this is set to 2x the longest inter-onset interval.
Highpass=128
; True: concatenate all runs in the scan session as if they were one continuous scan. Does not work if the trial
; types are different among the runs.
Concatenate_runs=False
; True: ignore contrast vectors specified below containinig regressors that don't match any of the trial types in the
; subject's task events file.
Remove_invalid_contrasts=True

; RESTING-STATE FMRI SPECIFIC OPTIONS
Resting_state={bRestingState}
; comma-separated list of tasks to process
Resting_task_names=rest
; Parcellations to use. List the parcellation name, then the path to the .nii file, then the path to a text file
; with ROI names
Parcellations={sConnAtlases}
Do_ica=True
; Number of ICA components
Ica_dims=20

; Diffusion image pre-processing options
[dMRI]
Do_processing={bDMRI}
Method= Camino ; Other options: DTI_TK
Atlas= MNI152 ; Other options: ICBM465/MoriAtlas(JHU)
''')

# Create and edit yaml file
with open(os.path.join(strJobDir, 'param_runner_' + strJobName + '.yaml'), 'w') as objFile:
    objFile.write(f'''command: >-
  bash {os.path.join(strJobDir, 'RunPipeline_' + strJobName + '.sh')} $B
work_dir: >-
  {strJobDir}
summary:
  - id: test
    regex: 'ra : ([-+]?[0-9]*\?[0-9])'
partition: super
nodes: {nNodes}
cpus_per_task: {nCpu}
time_limit: '{strTimeLimit}'
parameters:
  - id: subjects
    min: 0
    max: {nMaxSubjects - 1}
    step: 1
    substitution: $B
    type: int_range
                    ''')

# Create and edit shell file
with open(os.path.join(strJobDir, 'RunPipeline_' + strJobName + '.sh'), 'w') as objFile:
    objFile.write(f'''#!/bin/bash

# COMMAND GROUP 0
module load python            
# COMMAND GROUP 1
source {os.path.join(strDllabPipelineDir, 'config_files', 'exportpath.sh')}
# COMMAND GROUP 2
source activate {strConda}

# COMMAND GROUP 2
subject_index=$1

python {os.path.join(strDllabPipelineDir, 'processingPipeline.py')} -c {os.path.join(strJobDir, 'processingOptions_' + strJobName + '.ini')} -i $subject_index

# END OF SCRIPT
                    ''')
