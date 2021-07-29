#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:11:10 2021

@author: claudia
@goal: plot the 2d histograms of reflectivity, rain rate, LWC and fall speed vs height
for the MRR-PRO entire dataset. 
"""


# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences



def hist_and_plot(data, title, yvar='DWRxk', xvar='DWRkw',
                  xlabel='DWR Ka W   [dB]', ylabel='DWR X Ka   [dB]',
                  xlim=[-5, 20], ylim=[-5, 20], lognorm=True, vminmax=None,
                  savename='auto3f', inverty=False, figax=None, stats=None,
                  bins=100, density=False, CFAD=False, cmap='viridis'):
  dataclean = data[[xvar, yvar]].dropna()
  hst, xedge, yedge = np.histogram2d(dataclean[xvar], dataclean[yvar], bins=bins)
  hst = hst.T
  hst[hst==0] = np.nan
  if density:
    #xBinW = xedge[1:]-xedge[:-1]
    #yBinW = yedge[1:]-yedge[:-1]
    #hst = hst/xBinW/yBinW[:,np.newaxis]
    hst = 100.*hst/np.nansum(hst)
  if CFAD:
    hst = 100.*hst/np.nansum(hst,axis=1)[:,np.newaxis]
  xcenter = (xedge[:-1] + xedge[1:])*0.5
  ycenter = (yedge[:-1] + yedge[1:])*0.5
  if figax is None:
    fig, ax = plt.subplots(1,1)
  else:
    fig, ax = figax
  if lognorm:
    norm = colors.LogNorm(vmin=np.nanmin(hst[np.nonzero(hst)]),
                                             vmax=np.nanmax(hst))
    if vminmax is not None:
      norm = colors.LogNorm(vmin=vminmax[0], vmax=vminmax[1])
      hst[hst<0.1*vminmax[0]] = np.nan
    mesh = ax.pcolormesh(xcenter, ycenter, hst[:], cmap=cmap,
                         norm=norm)
  else:
    if vminmax is None:
      mesh = ax.pcolormesh(xcenter, ycenter, hst[:], cmap='jet')
    else:
      mesh = ax.pcolormesh(xcenter, ycenter, hst[:], cmap='jet',
                           vmin=vminmax[0], vmax=vminmax[1])
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  clabel='counts'
  if CFAD or density:
    clabel='relative frequency [%]'
    if stats is not None:
      bins = pd.cut(data[yvar], yedge)
      groups = data.groupby(bins)[xvar]
      lines = []
      labels = []
      if 'median' in stats:
        l = ax.plot(groups.median(), ycenter, label='median',
                    lw=2, ls='-.', c='k')
        lines.append(l[0])
        labels.append('median')
      if 'mean' in stats:
        l = ax.plot(groups.mean(), ycenter, label='mean',
                    lw=2, ls='--', c='r' )
        lines.append(l[0])
        labels.append('mean')
      if 'quartile' in stats:
        l = ax.plot(groups.quantile(0.25), ycenter, label='.25 quantile', c='k')
        ax.plot(groups.quantile(0.75), ycenter, label='.75 quantile',
                ls='-', c='k')
        lines.append(l[0])
        labels.append('quartiles')
      if 'decile' in stats:
        l = ax.plot(groups.quantile(0.10), ycenter, label='.10 decile',
                    ls=':', c='k')
        ax.plot(groups.quantile(0.90), ycenter, label='.90 decile',
                ls=':', c='k')
        lines.append(l[0])
        labels.append('deciles')
      ax.legend(lines, labels, framealpha=0.95)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.1)
  cb = plt.colorbar(mesh, cax=cax, extend='max', label=clabel)
  #cb = plt.colorbar(mesh, ax=ax, extend='max', label=clabel)
  ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  ax.grid()
  if inverty:
    ax.invert_yaxis()
  if savename == 'auto3f':
    fig.savefig('triple_frequency/'+'_'.join(title.split())+addlabel+'.png')
  elif savename is None:
    pass
  else:
    fig.savefig(savename)
  return hst, xcenter, ycenter, cb, xedge, yedge
def add_one_contour(ax, data, Tminmax, color, levels=1):
  if ax is None:
    fig, ax = plt.subplots(1,1)
  Tmin, Tmax = Tminmax
  if (Tmin is None) and (Tmax is None):
    Tstr = 'whole dataset'
    h,x,y = hist_and_plot(data[:], Tstr)
  elif Tmin is None:
    Tstr = 'T < '+str(Tmax)
    h,x,y = hist_and_plot(data[(data['T'] < Tmax)], Tstr)
  elif Tmax is None:
    Tstr = 'T > '+str(Tmin)
    h,x,y = hist_and_plot(data[(data['T'] > Tmin)], Tstr)
  else:
    Tstr = str(Tmin)+' < T < '+str(Tmax)
    h,x,y = hist_and_plot(data[(data['T'] <= Tmax)*(data['T'] > Tmin)],
                          Tstr)
  CS = ax.contour(x, y, np.log10(h[:]), 1, colors=color)
  CS.collections[0].set_label(Tstr)
  return ax


#%%

fileListProcess = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/*.nc'))
# combining all data in one dataset
day_data = xr.open_mfdataset(fileListProcess,
                               concat_dim = 'time',
                               data_vars = 'minimal',
                              )

#%%

# creating a xvar and yvar
yvar_matrix = (np.ones((len(day_data.time.values),1))*np.array([day_data.height.values]))
xvar_matrix = day_data.Zea.values
varStringArr = [ "Zea"]#, "W",'RR', "LWC"]
for ivar,varSel in enumerate(varStringArr):

    print('producing 2d histogram of '+varStringArr[ivar])
    varString = varStringArr[ivar]
    # settings for reflectivity
    if varString == 'Zea':
        mincm = -30.
        maxcm = 60.
        xvar_matrix = day_data.Zea.values
        cbarstr = 'ZE attenuated [dBZ]'
        strTitle = 'MRR - Equivalent reflectivity attenuated'
        bins = [128,100]
        # settings for mean Doppler velocity
    elif varString == 'W':
        mincm = -15.
        maxcm = 0.
        xvar_matrix = -day_data.fall_speed.values
        cbarstr = 'Fall speed [$ms^{-1}$]'
        strTitle = 'MRR - Fall speed '
        bins = [128,100]
        # settings for Spectral width
    elif varString == 'RR':
        mincm = -4. #5.81561144e-05
        maxcm = 4. #1.11669189e+04
        #colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
        colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
        xvar_matrix = day_data.rain_rate.values
        cbarstr = 'Log$_{10}$(Rainfall rate) [$mmh^{-1}$]'
        strTitle = 'MRR - Rainfall rate '
        bins = [128,100]
        # settings for skewness
    elif varString == 'LWC':
        mincm = -5. #1.
        maxcm = 2.6 #500.
        colors = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
        xvar_matrix = day_data.liquid_water_content.values
        cbarstr = 'Log$_{10}$(LWC) [$gm^{-3}$]'
        strTitle = 'MRR - Liquid water content '
        bins = [128,100]
    dict_plot = {'path':"/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/plots/stats_plots/",
             "varname":varString, 
             "instr":"MRR_PRO"}

    # flattening the series 
    yvar = yvar_matrix.flatten()
    xvar = xvar_matrix.flatten()
    
    # plot 2d histogram figure 
    i_good = (~np.isnan(xvar) * ~np.isnan(yvar) * (xvar < 100.))
   
    
    hst, xedge, yedge = np.histogram2d(xvar[i_good], yvar[i_good], bins=bins)
    if varString == "LWC":
        hst, xedge, yedge = np.histogram2d(np.log10(xvar[i_good]), yvar[i_good], bins=bins)
    if varString == "RR":
        hst, xedge, yedge = np.histogram2d(np.log10(xvar[i_good]), yvar[i_good], bins=bins)
        
    hst = hst.T
    hst[hst==0] = np.nan
    
    hst = 100.*hst/np.nansum(hst,axis=1)[:,np.newaxis]
    xcenter = (xedge[:-1] + xedge[1:])*0.5
    ycenter = (yedge[:-1] + yedge[1:])*0.5
    
    print(xcenter)
    hmin         = 0.05#50.
    hmax         = 1.3# 1300.
    labelsizeaxes = 30
    fontSizeTitle = 30
    fontSizeX     = 30
    fontSizeY     = 30
    cbarAspect    = 50
    fontSizeCbar  = 30
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9))
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    ax = plt.subplot(1,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    cax = ax.pcolormesh(xcenter, ycenter*0.001, hst, cmap='viridis')
    #ax.set_xscale('log')
    ax.set_ylim(hmin,hmax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(mincm, maxcm)                                 # limits of the x-axes
    #ax.set_title(strTitle, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel(cbarstr, fontsize=fontSizeX)
    ax.set_ylabel("Height [km]", fontsize=fontSizeY)
    cbar = fig.colorbar(cax, orientation='vertical')
    cbar.set_label(label='Count', size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)
    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    
    fig.tight_layout()
    fig.savefig('{path}{varname}_{instr}_2d_CFAD_MRR.png'.format(**dict_plot), bbox_inches='tight')