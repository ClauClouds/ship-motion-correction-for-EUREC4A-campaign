"""
Plot ship position on a map.
"""

import sys
import numpy as np
import pandas as pd
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append('/media/nrisse/data/work/igmk/eurec4a/scripts')
from path_setter import path_plot, path_data


if __name__ == '__main__':
    
    # read ship position data
    ds_pos_raw = xr.load_dataset(path_data+'position/shipData_all2.nc')
    
    #%% prepare data
    # remove outliers
    ds_pos_raw = ds_pos_raw.sel(time=ds_pos_raw.lat != 0)
    
    # join on 1 minute time stamp
    t0_eurec4a = datetime.datetime(2020, 1, 19, 0, 0, 0)
    t1_eurec4a = datetime.datetime(2020, 2, 19, 23, 59, 59)
    
    ds_pos = xr.Dataset()
    ds_pos.coords['time'] = pd.date_range(t0_eurec4a, t1_eurec4a, freq='1min')
    ds_pos = xr.merge([ds_pos, ds_pos_raw[['lat', 'lon']]], 
                      compat='no_conflicts', join='left')    

    # create list of dates during eurec4a
    t_eurec4a_daily = pd.date_range(t0_eurec4a, t1_eurec4a, freq='1D')
    
    #%% plot settings
    # colors
    col_ocean = '#CAE9FB'
    col_land = '#8EC460'
    col_coastlines = '#1F7298'
    cmap_track = 'plasma'
    col_track = cm.get_cmap(cmap_track, len(t_eurec4a_daily)).colors
    
    # fontsizes
    fs_grid = 10
    fs_cbar_labels = 10
    fs_track_labels = 10
    
    # zorders
    zorder_land = 0
    zorder_coastlines = 1
    zorder_gridlines = 2
    zorder_day_marker = 4
    zorder_track = 3
    zorder_day_annotation = 5
    
    #%% plot minutely ship position
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    # set map extent
    ax.set_extent(extents=(-61, -50, 6, 15), crs=ccrs.PlateCarree())

    # set ocean color
    ax.background_patch.set_facecolor(col_ocean)
        
    # add land feature
    land_10m = cfeature.NaturalEarthFeature(category='physical', name='land', 
                                            scale='10m',
                                        edgecolor=col_coastlines,
                                        linewidth=0.5,
                                        facecolor=col_land)
    ax.add_feature(land_10m, zorder=zorder_land)
    
    # add lat lon grid
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color='#7AB9DC', 
                      draw_labels=True, zorder=zorder_gridlines,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-70, -40, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 16, 1))
    gl.xlabel_style = {'size': fs_grid, 'color': 'k'}
    gl.ylabel_style = {'size': fs_grid, 'color': 'k'}
    
    # plot ship track
    for i, day in enumerate(t_eurec4a_daily.date):
                
        # plot line for ship position
        ax.plot(ds_pos.lon.sel(time=str(day)), 
                ds_pos.lat.sel(time=str(day)), 
                color=col_track[i], 
                transform=ccrs.PlateCarree(),
                zorder=zorder_track,
                )
        
        # add point when date starts
        ax.scatter(ds_pos.lon.sel(time=str(day))[0], 
                   ds_pos.lat.sel(time=str(day))[0], 
                   color='None', 
                   edgecolor='white',
                   s=15,
                   marker='D',
                   linewidths=1.5,
                   zorder=zorder_day_marker,
                   transform=ccrs.PlateCarree())
        
        # annotate day
        txt = day.strftime('%-m/%-d')
        if txt in ['1/27', '2/11', '1/20', '1/21', '2/14', '2/7', '1/22', 
                   '1/28', '2/6', '2/8', '2/12', '2/16']:
            ha = 'left'
        else:
            ha = 'right'
        
        if txt in ['2/1', '1/21', '2/14', '2/7', '1/22', '1/28', '2/13', 
                   '2/18', '2/9', '2/16']:
            va = 'top'
        else:
            va = 'bottom'
        
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        ax.annotate(txt, 
                    xy=(ds_pos.lon.sel(time=str(day))[0].values,
                        ds_pos.lat.sel(time=str(day))[0].values), 
                    ha=ha, va=va, xycoords=transform, 
                    fontsize=fs_track_labels, zorder=zorder_day_annotation)
    
    # add colorbar for dates
    cmap = plt.get_cmap(cmap_track)
    boundaries = np.arange(0, len(t_eurec4a_daily)+1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        ticks=np.arange(0.5, len(t_eurec4a_daily)+0.5))
    
    # modify labels
    labels = [t.strftime('%-d') for t in t_eurec4a_daily]
    labels[0] += ' Jan'
    labels[13] += ' Feb'
    cbar.ax.set_yticklabels(labels, fontsize=fs_cbar_labels)
    
    plt.savefig(path_plot+'ship_track/figure_2_ship_position.svg', 
                bbox_inches='tight')
    plt.savefig(path_plot+'ship_track/figure_2_ship_position.png', 
                dpi=300, bbox_inches='tight')
    
    plt.close('all')
