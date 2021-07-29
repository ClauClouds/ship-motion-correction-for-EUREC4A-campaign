#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:36:20 2021
code for producing figure paper cfad wband and mrrpro
@author: claudia
"""

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

fileListProcess_MRR = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/*.nc'))
# combining all data in one dataset
day_data = xr.open_mfdataset(fileListProcess_MRR,
                               concat_dim = 'time',
                               data_vars = 'minimal',
                              )

# creating a xvar and yvar
yvar_matrix = (np.ones((len(day_data.time.values),1))*np.array([day_data.height.values]))
xvar_matrix = day_data.Zea.values

# settings for reflectivity mrr-pro
varString ="Zea"
mincm = -30.
maxcm = 60.
xvar_matrix = day_data.Zea.values
cbarstr = 'ZE attenuated [dBZ]'
strTitle = 'MRR - Equivalent reflectivity attenuated'
bins = [128,100]
# settings for mean Doppler velocity



# flattening the series 
yvar = yvar_matrix.flatten()
xvar = xvar_matrix.flatten()

# plot 2d histogram figure 
i_good = (~np.isnan(xvar) * ~np.isnan(yvar) * (xvar < 100.))
hst, xedge, yedge = np.histogram2d(xvar[i_good], yvar[i_good], bins=bins)
hst = hst.T
hst[hst==0] = np.nan
hst = 100.*hst/np.nansum(hst,axis=1)[:,np.newaxis]
xcenter = (xedge[:-1] + xedge[1:])*0.5
ycenter = (yedge[:-1] + yedge[1:])*0.5

hmin         = 0.05#50.
hmax         = 1.3# 1300.


#%% 
# reading w-band data
fileListProcess_WBAND = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/eurec4a_intake/cfad/*.nc'))

# combining all data in one dataset
day_data_W = xr.open_mfdataset(fileListProcess_WBAND,
                               concat_dim = 'time',
                               data_vars = 'minimal',
                              )
# creating a xvar and yvar
yvar_matrix_w = (np.ones((len(day_data_W.time.values),1))*np.array([day_data_W.height.values]))
xvar_matrix_w = 10*np.log10(day_data_W.radar_reflectivity.values)
#%%
# settings for reflectivity mrr-pro
varString_w ="Ze"
mincm_w = -65.
maxcm_w = 30.
cbarstr_w = 'Ze [dBZ]'
strTitle_w = 'Wband - reflectivity '
ystart = day_data_W.height.values[0] +  np.ediff1d(day_data_W.height.values)[0]/2

y_edges = []
for ind in range(len(day_data_W.height.values)-1):  
    h_val = day_data_W.height.values[ind]
    y_edges.append(h_val + float((day_data_W.height.values)[ind+1]-h_val)/2)
y_final = y_edges[0:-1:3]
x_edges = np.arange(start=mincm_w, stop=maxcm_w , step=1.)

bins_w = [x_edges, y_final]

# flattening the series 
yvar_w = yvar_matrix_w.flatten()
xvar_w = xvar_matrix_w.flatten()

# plot 2d histogram figure 
i_good = ((~np.isnan(xvar_w)) * (~np.isnan(yvar_w)))
hst_w, xedge_w, yedge_w = np.histogram2d(xvar_w[i_good], yvar_w[i_good], bins=bins_w)
hst_w = hst_w.T

#hst_w[hst_w < 1.] = np.nan
hst_w[hst_w==0.] = np.nan
hst_w = 100.*hst_w/np.nansum(hst_w,axis=1)[:,np.newaxis]
xcenter_w = (xedge_w[:-1] + xedge_w[1:])*0.5
ycenter_w = (yedge_w[:-1] + yedge_w[1:])*0.5

hmin_w         = 0.01#50.
hmax_w         = 10.0# 1300.
#%%
dict_plot = {'path':"/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/plots/stats_plots/",
         "varname":varString, 
         "instr":"Wband_MRR_PRO"}

labelsizeaxes = 32
fontSizeTitle = 32
fontSizeX     = 32
fontSizeY     = 32
cbarAspect    = 20
fontSizeCbar  = 32
fonts_numbers = 32
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

fig, axs = plt.subplots(1, 2, figsize=(25,12), constrained_layout=True, sharex=False)
# setting dates formatter 
matplotlib.rc('xtick', labelsize=fonts_numbers)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=fonts_numbers)  # sets dimension of ticks in the plots
for ax, l in zip(axs.flatten(), ['(a) W-band radar ','(b) MRR-PRO']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=fonts_numbers, transform=ax.transAxes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    
# build colorbar
mesh = axs[0].pcolormesh(xcenter_w, ycenter_w*0.001, hst_w, cmap='viridis', rasterized=True) 
#, norm=colors.LogNorm(vmin=1., vmax=10**(2))
axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=2))
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].set_xlim(mincm_w, maxcm_w)
cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=cbarAspect, use_gridspec=True)
cbar.set_label(label='Occurrences',  size=fontSizeY)
cbar.ax.tick_params(axis='y', direction='out',  length=10, width=2)
cbar.ax.tick_params(axis='y', which='minor', length=5, width=2)

cbar.outline.set_linewidth(2)
axs[0].set_xlabel('Radar reflectivity [dBZ]', fontsize=fontSizeX)
axs[0].set_ylabel('Height [km]', fontsize=fontSizeX)
axs[0].set_ylim(hmin_w, hmax_w)


mesh = axs[1].pcolormesh(xcenter, ycenter*0.001, hst, cmap='viridis', rasterized=True)
axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=2))
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].set_xlim(-30., 60.)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=cbarAspect, use_gridspec=True)
cbar.set_label(label='Occurrences',  size=fontSizeY)
cbar.outline.set_linewidth(2)
cbar.ax.tick_params(axis='y', direction='out', length=10, width=2)
axs[1].set_xlabel('Radar attenuated reflectivity [dBZ]', fontsize=fontSizeX)
axs[1].set_ylabel('Height [km]', fontsize=fontSizeX)
axs[1].set_ylim(hmin, hmax)

fig.savefig('{path}{varname}_{instr}_2d_CFAD.png'.format(**dict_plot), bbox_inches='tight')

#%%

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,12))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1,2,1)
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
ax.set_xlabel('Radar reflectivity [dBZ]', fontsize=fontSizeX)
ax.set_ylabel("Height [km]", fontsize=fontSizeY)
cbar = fig.colorbar(cax, orientation='vertical')
cbar.set_label(label='Count', size=fontSizeCbar)
cbar.ax.tick_params(labelsize=labelsizeaxes)
# Turn on the frame for the twin axis, but then hide all
# but the bottom spine

fig.tight_layout()
fig.savefig('{path}{varname}_{instr}_2d_CFAD_MRR.png'.format(**dict_plot), bbox_inches='tight')