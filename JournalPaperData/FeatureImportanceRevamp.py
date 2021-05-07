#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file creates the feature importance plot for the ASD manuscript


Originally developed for use in the ASD comparison project
Created by Cooper Mellema
in Dr. Montillo's Deep Learning Lab
University of Texas Southwestern Medical Center
Lyda Hill Dept. of Bioinformatics
"""

__author__ = "Cooper Mellema"
__email__ = "Cooper.Mellema@UTSouthwestern.edu"
__status__ = "Prototype"

import os
import pickle as pkl
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd
import copy
import numpy as np
import re
import json
import glob
from scipy.stats import ttest_ind as spttest_ind
import re
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ttest_ind as smttest_ind
import matplotlib.transforms as mtrans

def fLoadImportances(sDir = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison64Permutations'):
    """ Loads feature importance data
        importances are stored as Model 1 = best 64 ROI model, Model 2 = best 122 ROI, ... Model 4 = second best 64 ROI model, etc.
    Args:
        sDir (str, optional): where permutation results are stored.
            Defaults to '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison64Permutations'.

    Returns:
        pdImportances()
    """
    dImportances={}
    for i in range(15):
        dData = pkl.load(open(os.path.join(sDir, f'RawImportances/Model{i+1}Importances.p'), 'rb'))
        dImportances.update({i:dData})

    # sort into atlases
    pdData = pd.DataFrame.from_dict(dImportances).transpose()
    pdData.insert(0,'Atlas', np.tile([64,122,197],5))

    return pdData

def fPValuePermutation(series, df):
    """Calculates p-values for each importance

    Args:
        series (pandas series): dataframe column from importance testing
        df (dataframe): [description]

    Returns:
        p value (float)
    """
    _,p,_ = smttest_ind(fGenerateNullDistribution(df.drop(series.name, axis=1, inplace=False)),
        series.values, alternative='smaller')
    return p

def fPValueFill(pdData, lsCols, iAtlas, sPath='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/Pvalues'):
    """replaces raw importance scores with p-values

    Args:
        pdData: importance dataframe
        lsCols: list of columns to test (top 45)
        iAtlas: int: atlas designation

    Returns:
        pdPvals: p-value dataframe
    """
    if not os.path.isfile(f'{sPath}_{iAtlas}.csv'):
        # initialize holding variables
        pdPvals=copy.deepcopy(pdData)
        pdPvals.iloc[:,1:]=np.nan
        dPosition={
            64:0,
            122:1,
            197:2
        }
        # fill in p-values
        for iRow in range(5):
            for sCol in lsCols:
                pdPvals.loc[(dPosition[iAtlas]+3*iRow), sCol] =\
                    fPValuePermutation(pdData.iloc[iRow:iRow+1,:][sCol], pdData.iloc[iRow:iRow+1,1:])

        #FDR correct to 0.01
        pdPvals=pdPvals.dropna(axis=1)
        for i,_ in enumerate(pdPvals.index):
            pdPvals.iloc[i,1:]=fdrcorrection(pdPvals.iloc[i,1:].dropna().values.flatten(), alpha=0.01, method='negcorr')[1]

        #drop atlas column
        pdPvals = pdPvals.iloc[:,1:]
        pdPvals.to_csv(f'{sPath}_{iAtlas}.csv')

    else:
        pdPvals = pd.read_csv(f'{sPath}_{iAtlas}.csv', index_col=0)
    
    return pdPvals

def fGenerateNullDistribution(df):
    """ generates a null distribution of feature importances based on the work from:
        Janitza 2018: A Computationally Fast Variable Importance test for random forests for high-dimensional data
    """
    # flatten all non-nan importance values of 0 or less
    vec=df[df<=0].values[~(np.isnan(df[df<=0].values))].flatten()
    # mirror distribution around 0
    vec=np.concatenate([vec[vec!=0], vec[vec==0],-vec[vec!=0]])
    return vec

def fImportancesAndPvals(sRoot = '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison64Permutations'):
    """
    # NOTE not currently working
    """
    lsData = []
    for sFile in glob.glob(os.path.join(sRoot, '*.p')):
        sNewFile=sFile.replace('.p', '_Pvals.csv')
        if not os.path.isfile(sNewFile):
            # load raw data
            df=pd.DataFrame.from_dict(pkl.load(open(sFile,'rb')))
            # calculate pvals, median, and mean
            df=pd.concat([df.median(axis=0), df.mean(axis=0), df.apply(_fP, axis=0, args=(df,))], axis=1)
            # rename and label columns
            df.columns=['Median Importance', 'Mean Importance', 'One-sided t-test Pval']
            df.insert(0,'Model',int(re.search(r'\d+', sFile.split('/')[-1]).group()))
            # do FDR correction
            for flAlpha in [0.05, 0.01]:
                df.loc[~df['One-sided t-test Pval'].isna(), f'FDR {flAlpha} corrected Pvals'] = fdrcorrection(df['One-sided t-test Pval'].dropna().values, alpha=flAlpha, method='negcorr')[1]
            df.to_csv(sNewFile)
        else:
            df = pd.read_csv(sNewFile, index_col=0)
        lsData.append(df)

    #below is updated median feature importance across more permutations
    pdImportancesAndPvals = pd.concat(lsData, axis=1)
    pdData = pd.DataFrame.from_dict({i:pdImportancesAndPvals.loc[pdImportancesAndPvals['Model']==i, 'Median Importance'].to_dict() for i in sorted(pdImportancesAndPvals['Model'].unique())}).transpose()
    pdData.insert(0,'Atlas', np.tile([64,122,197],5))
    return pdData

def _fLookupIfExist(dDict, sKey1, sKey2):
    """returns ROI dict value or appropriately formatted key
    """
    try:
        return dDict[sKey1][sKey2]
    except:
        return f'ROI {int(sKey2)+1}'

def fRenameROIs(pdData, iAtlas):
    """ reformats ROI names in dataframe to functional regional names

    Args:
        pdData (dataframe): dataframe of importances
        iAtlas (int): atlas resolution (64, 122, or 197)

    Returns:
        pdData (dataframe): dataframe of importances, updated names
    """
    import re
    import json
    # takes ROIx-ROIy string and changes it to named regions
    dLookup = json.load(open('/project/bioinformatics/DLLab/s169682/Code/AutismProject/JournalPaperData/ROICorrespondence.json', 'r'))

    # loop through names and rename if in lookup table
    for sIndex in pdData.index:
        lsDigits=re.findall('\d+', sIndex)
        pdData=pdData.rename(index={sIndex: f'{_fLookupIfExist(dLookup,str(iAtlas),str(int(lsDigits[0])-1))} to {_fLookupIfExist(dLookup,str(iAtlas),str(int(lsDigits[1])-1))}'})
        pdData=pdData.rename(index={sIndex: sIndex.replace('-', ' to ')})
    return pdData

def fFetchColor(lsROIs):
    """ formats and fetches desinated colors for funtional mappings

    Args:
        lsROIs (list): roi-to-roi names for lookup

    Returns:
        list of colors for seaborn plotting
    """
    # fetch lookup table:
    dLookup = json.load(open('/project/bioinformatics/DLLab/s169682/Code/AutismProject/JournalPaperData/ROIFunctions.json', 'r'))

    # define deep palette colors
    dColors={
        'r':list(sns.color_palette("deep"))[3],
        'o':list(sns.color_palette("deep"))[1],
        'y':list(sns.color_palette("deep"))[8],
        'g':list(sns.color_palette("deep"))[2],
        'b':list(sns.color_palette("deep"))[0],
        'v':list(sns.color_palette("deep"))[4]
    }

    def _fColorFunc(sFunc1, sFunc2):
        if set([sFunc1, sFunc2])==set(['m','m']):
            return 'b'
        elif set([sFunc1, sFunc2])==set(['m','o']):
            return 'g'
        elif set([sFunc1, sFunc2])==set(['o','o']):
            return 'y'
        elif set([sFunc1, sFunc2])==set(['o','l']):
            return 'o'
        elif set([sFunc1, sFunc2])==set(['l','l']):
            return 'r'
        elif set([sFunc1, sFunc2])==set(['m','l']):
            return 'v'
        else:
            raise NotImplementedError
    
    # map ROIs to colors
    lsColors = []
    for sROIs in lsROIs:
        sROI1, sROI2 = sROIs.split(' to ')
        sFunc1 = dLookup[sROI1]
        sFunc2 = dLookup[sROI2]
        lsColors.append(_fColorFunc(sFunc1, sFunc2))

    # map color names to color palette colors
    return [dColors[x] for x in lsColors]

def fTTest(sCol, iAtlas, dXData, aYData):
    """ takes pandas dataframe and performs t-test between ASD and HC feature values

    Args:
        sCol (string): designates which feature is being analysed
        iAtlas (int): atlas resolution (64, 122, or 197)
        dXData, aYData (dict, array): formatted ASD data for testing between ASD and HC

    Returns:
        p, diff (float, float) p-value and average difference between HC and ASD
    """
    sID=f"basc{f'{iAtlas}'.zfill(3)}"
    pdASD=dXData[sID].loc[np.squeeze(aYData).astype(bool)]
    pdHC=dXData[sID].loc[~np.squeeze(aYData).astype(bool)]
    
    _, p = spttest_ind(pdASD[sCol].values, pdHC[sCol].values, equal_var=False)
    diff = pdASD[sCol].mean()-pdHC[sCol].mean()
    return p, diff

def fTTestIcon(sCol, iAtlas,dXData, aYData):
    """does t-testing for a feature [is there a diff between hc and asd?] and returns icon for graph
        (+ for asd>hc, 0 for =, - for asd<hc)

    Args:
        sCol (string): designates which feature is being analysed
        iAtlas (int): atlas resolution (64, 122, or 197)
        dXData, aYData (dict, array): formatted ASD data for testing between ASD and HC

    Returns:
        string, +, -, or o
    """
    p,diff = fTTest(sCol, iAtlas,dXData, aYData)
    if p>0.05:
        return 'o'
    if diff>0:
        return '+'
    else:
        return '-'

def fPlot(pdData, iAtlas, ax, dXData, aYData, sSort='Median'):
    """plots all the feature importances, their functions, and asd vs hc comparison

    Args:
        pdData (dataframe): of importances
        iAtlas (int): atlas resolution (64, 122, or 197)
        ax (axis): matplotlib axis object
        dXData, aYData (dict, array): formatted ASD data for testing between ASD and HC

    """
    import matplotlib as mpl
    mpl.rc('hatch', linewidth=6)
    sns.set(rc={'axes.facecolor':'gainsboro'})

    pdAtlas = pdData[pdData['Atlas']==iAtlas].transpose().dropna().iloc[1:]
    if sSort=='Mean':
        pdAtlas[sSort] = pdAtlas.mean(axis=1)
        fEstimator=np.mean
    elif sSort=='Median':
        pdAtlas[sSort] = pdAtlas.median(axis=1)
        fEstimator=np.median
    else:
        raise NotImplementedError
    pdAtlas=pdAtlas/pdAtlas[sSort].std()
    pdPlot=pdAtlas.sort_values(by=sSort, ascending=False).filter(like='ROI', axis=0).head(15).iloc[:,:-1].T

    # get ttest results
    lsIcons = [fTTestIcon(x, iAtlas,dXData, aYData) for x in pdPlot.columns]
    pdPval=fPValueFill(pdData[pdData['Atlas']==iAtlas], pdPlot.columns, iAtlas)
    # reorder p values
    pdPval=pdPval[list(pdPlot.columns)]

    # rename ROIs
    pdPlot_raw = fRenameROIs(pdPlot.transpose(), iAtlas).transpose()
    
    #fetch list of colors
    lsColors = fFetchColor(list(pdPlot_raw.columns))
    #lsColors = fFetchColor(iAtlas)

    # plot the horizontal bar graph for background
    pdPlot=pdPlot_raw.melt(var_name='cols', value_name='vals')
    pdPlot['cols']=[f'{x} ' for x in pdPlot['cols'].values]
    g0=sns.barplot(data=pdPlot, y='cols', x='vals', estimator=fEstimator, palette=lsColors, ci=None, ax=ax, zorder=0)
    g0.set(xlabel='\n')
    g0.set(ylabel=None)

    # # plot background p-values
    # aPvals = pdPval.values[(pdPlot_raw==pdPlot_raw.median()).values]
    # ax2=ax.twiny()
    # ax2.grid(False)
    # ax2.plot(aPvals, range(len(aPvals)), color='dimgray', marker='d', ms=6, linewidth=0, linestyle=None)
    # ax2.plot(aPvals, range(len(aPvals)), color='firebrick', marker='d', ms=4, linewidth=0, linestyle=None, alpha=1)
    # ax2.set_xscale("log")
    # ax2.set_xlim(ax2.get_xlim()[::-1])
    # ax2.hlines(xmin=np.ones_like(aPvals), xmax=aPvals, y=range(len(aPvals)), color='dimgray', linewidth=1)
    # ax2.set_zorder(3)
    # ax2.set_facecolor(None)

    # plot the horizontal bar graph again
    ax3=ax.twiny()
    ax3.set_zorder(5)
    g=sns.barplot(data=pdPlot, y='cols', x='vals', estimator=fEstimator, palette=lsColors, ci=None, ax=ax3)
    g.set(xlabel='\n')
    g.set(ylabel=None)
    g.set_facecolor(None)
    g.grid(False)

    # add in ttest result labels
    for i,p in enumerate(g.patches):
        if 15>i>=0:
            width = p.get_width()
            ax3.text(-pdAtlas[sSort].filter(like='ROI', axis=0).max()*0.0175,
                    p.get_y()+0.60*p.get_height()-0.075,
                    lsIcons[i],
                    ha='center', va='center', size=12, weight='bold')
                
    # plot light or dark bars:
    g_white=sns.barplot(data=pdPlot, y='cols', estimator=fEstimator, x='vals', edgecolor=(1,1,1,0.3), facecolor=(1,1,1,0), ci=None, ax=ax3, hatch='\\', lw=0, zorder=15)
    g_white.set(xlabel='\n')
    g_white.set(ylabel=None)
    g_white.set_facecolor(None)
    g_white.grid(False)

    g_black=sns.barplot(data=pdPlot, y='cols', estimator=fEstimator, x='vals', edgecolor=(0,0,0,0.25), facecolor=(1,1,1,0), ci=None, ax=ax3, hatch='/', lw=0, zorder=15)
    g_black.set(xlabel='\n')
    g_black.set(ylabel=None)
    g_black.set_facecolor(None)
    g_black.grid(False)

    for i,p in enumerate(g_white.patches):
        if 30>i>=15:
            if lsIcons[i-15]!='+':
                p.set_hatch(None)
    for i,p in enumerate(g_black.patches):
        if i>=30:
            if lsIcons[i-30]!='-':
                p.set_hatch(None)

    # plot outlines on bars
    g_outline=sns.barplot(data=pdPlot, y='cols', estimator=fEstimator, x='vals', edgecolor='dimgray', facecolor=(1,1,1,0), ci=None, ax=ax3, lw=1, zorder=20)
    g_outline.set(xlabel='\n')
    g_outline.set(ylabel=None)
    g_outline.set_facecolor(None)
    g_outline.grid(False)

    # hide extra labels
    ax3.axes.xaxis.set_visible(False)

    # plot corresponding p-values
    # ax4=ax.twiny()
    # ax4.set_zorder(100)
    # ax4.grid(False)
    # ax4.plot(aPvals, range(len(aPvals)), color='dimgray', marker='d', ms=6, linewidth=0, linestyle=None, zorder=50)
    # ax4.plot(aPvals, range(len(aPvals)), color='firebrick', marker='d', ms=4, linewidth=0, linestyle=None, alpha=1, zorder=60)
    # ax4.set_xscale("log")
    # ax4.set_xlim(ax4.get_xlim()[::-1])
    # ax4.set_facecolor(None)


def fPlotAll(pdData, sSavePath=None):
    """ plots all the feature importances, their functions, and asd vs hc comparison for 3 atlases

    Args:
        pdData (dataframe): of importances
        sSavePath (string, optional): save path. Defaults to None.
    """
    dims = (8, 10)
    fig, ax = plt.subplots(3, figsize=dims)
    
    sData='/project/bioinformatics/DLLab/Cooper/Code/AutismProject/AllDataWithConfounds.p'
    #Dictionary that containes the whole dataset (train and test) in pd dataframe
    [dXData, aYData] = pkl.load(open(sData, 'rb'))
    
    for i, iAtlas in enumerate([64,122,197]):
        fPlot(pdData, iAtlas, ax[i], dXData, aYData)
    ax[0].set_xlabel('P-value (FDR corrected)', labelpad=30)
    ax[0].xaxis.set_label_position('top')
    ax[2].set_xlabel('Importance: standard deviations from mean importance')
    plt.subplots_adjust(bottom=-0.1, top=1)

    if sSavePath is not None:
        plt.savefig(sSavePath, bbox_inches='tight', dpi=1000)
        plt.show()
    else:
        plt.show()

if '__main__' == __name__:
    pdData = fLoadImportances('/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/AtlasResolutionComparison8Permutations')
    fPlotAll(pdData, '/project/bioinformatics/DLLab/Cooper/Code/AutismProject/JournalPaperData/FeatureImportances.png')
