import analysis
import paths
import xarray
import cProfile, pstats, io
from pstats import SortKey
from os.path import isfile
from multiprocessing import Process
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
import cartopy.crs as ccrs
import regionmask
import numpy as np
from analysis import bilinear_interpolation
from regionmask.defined_regions import ar6
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages


def gen_autocorrelation_maps():
    merra2_max = xarray.open_dataset(paths.merra2_download())["T2MMAX"]
    merra2_min = xarray.open_dataset(paths.merra2_download())["T2MMIN"]
    max_coeffs = merra2_max.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    min_coeffs = merra2_min.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    merra2_max_detrended = merra2_max - xarray.polyval(merra2_max.time, max_coeffs)
    merra2_min_detrended = merra2_min - xarray.polyval(merra2_min.time, min_coeffs)
    merra2_max_anomalies = merra2_max_detrended.groupby("time.dayofyear") - merra2_max_detrended.groupby("time.dayofyear").mean()
    merra2_min_anomalies = merra2_min_detrended.groupby("time.dayofyear") - merra2_min_detrended.groupby("time.dayofyear").mean()

    merra2_max_auto_anomalies = analysis.autocorrelation(merra2_max_anomalies)
    merra2_min_auto_anomalies = analysis.autocorrelation(merra2_min_anomalies)
    
    all_max_avg = xarray.open_mfdataset(paths.trefhtmx_members()[0], concat_dim="member", combine="nested")["TREFHTMX"].mean(dim="member").sel(time=slice("1980-01-01", "2015-01-01"))
    all_min_avg =xarray.open_mfdataset(paths.trefhtmn_members()[0], concat_dim="member", combine="nested")["TREFHTMN"].mean(dim="member").sel(time=slice("1980-01-01", "2015-01-01"))

    max_coeffs = all_max_avg.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    min_coeffs = all_min_avg.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    all_max_detrended = all_max_avg - xarray.polyval(all_max_avg.time, max_coeffs)
    all_min_detrended = all_min_avg - xarray.polyval(all_min_avg.time, min_coeffs)
    all_max_anomalies = all_max_detrended.groupby("time.dayofyear") - all_max_detrended.groupby("time.dayofyear").mean()
    all_min_anomalies = all_min_detrended.groupby("time.dayofyear") - all_min_detrended.groupby("time.dayofyear").mean()
    
    all_max_auto_anomalies = analysis.autocorrelation(all_max_anomalies.drop("dayofyear"))
    all_min_auto_anomalies = analysis.autocorrelation(all_min_anomalies.drop("dayofyear"))
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 15), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"MERRA2 vs ALL Anomalies Autocorrelation", fontsize=26)
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})

    cmap="hot"
    vmax=1
    vmin=0

    merra2_max_auto_anomalies.plot(ax=ax1, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title("MERRA2 Temp. Max. Autocorrelation 1980-2014")
    ax1.coastlines()
    merra2_min_auto_anomalies.plot(ax=ax2, cmap=cmap, vmax=vmax, vmin=vmin)
    ax2.set_title("MERRA2 Temp. Min. Autocorrelation 1980-2014")
    ax2.coastlines()
    all_max_auto_anomalies.plot(ax=ax3, cmap=cmap, vmax=vmax, vmin=vmin)
    ax3.set_title("ALL Temp. Max. Autocorrelation 1980-2014")
    ax3.coastlines()
    all_min_auto_anomalies.plot(ax=ax4, cmap=cmap, vmax=vmax, vmin=vmin)
    ax4.set_title("ALL Temp. Min. Autocorrelation 1980-2014")
    ax4.coastlines()

    f.tight_layout()
    return f


def gen_isolated_signal_map(time_begin: int, time_end: int, var: str, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{var} Isoalted Signal Maps EXP-{exp_num} from {time_begin} to {time_end}", fontsize=26)

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_min_tavg.plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (all_min_tavg - xghg_min_tavg).plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("GHG Min. Avg.")
    ax2.coastlines()
    (all_min_tavg - xaer_min_tavg).plot(ax=ax3, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax3.set_title("AER Min. Avg.")
    ax3.coastlines()

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_max_tavg.plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (all_max_tavg - xghg_max_tavg).plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("GHG Max. Avg.")
    ax5.coastlines()
    (all_max_tavg - xaer_max_tavg).plot(ax=ax6, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax6.set_title("AER Max. Avg.")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_global_spatial_avg(var: str, exp_num: str):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 25), facecolor='w')
    f.suptitle(f"{var} Exp. {exp_num} Global Spatial Average Over Time", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)

    lat_lon_avg = lambda ds : ds.mean(dim="lat").mean(dim="lon")

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days

    all_min_savgs.plot(ax=ax1, alpha=0.7, color="green", linestyle="dotted")
    (all_min_savgs.groupby("time").mean()).plot(ax=ax1, color="green", linewidth=4, label="ALL")
    xghg_min_savgs.plot(ax=ax1, alpha=0.7, color="red", linestyle="dotted")
    (xghg_min_savgs.groupby("time").mean()).plot(ax=ax1, color="red", linewidth=4, label="XGHG")
    xaer_min_savgs.plot(ax=ax1, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_min_savgs.groupby("time").mean()).plot(ax=ax1, color="blue", linewidth=4, label="XAER")

    ax1.grid()
    ax1.set_title("Using Min. Temperature")
    ax1.legend()

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days

    all_max_savgs.plot(ax=ax2, alpha=0.7, color="green", linestyle="dotted")
    (all_max_savgs.groupby("time").mean()).plot(ax=ax2, color="green", linewidth=4, label="ALL")
    xghg_max_savgs.plot(ax=ax2, alpha=0.7, color="red", linestyle="dotted")
    (xghg_max_savgs.groupby("time").mean()).plot(ax=ax2, color="red", linewidth=4, label="XGHG")
    xaer_max_savgs.plot(ax=ax2, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_max_savgs.groupby("time").mean()).plot(ax=ax2, color="blue", linewidth=4, label="XAER")

    ax2.grid()
    ax2.set_title("Using Max. Temperature")
    ax2.legend()

    f.tight_layout()
    return f


def gen_merra2_comparison(var: str, exp_num: str):
    time_range=[str(date) for date in list(range(1980, 2015 + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{var} ALL vs MERRA2 Comparinson EXP-{exp_num} from 1980 to 2015", fontsize=26)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    merra2_min = bilinear_interpolation(xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])["HWF_tn90pct"].mean(dim="time").dt.days, all_min_tavg)
    merra2_max = bilinear_interpolation(xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])["HWF_tx90pct"].mean(dim="time").dt.days, all_max_tavg)

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    perc_vmin=-200
    perc_vmax=300
    perc_norm = colors.TwoSlopeNorm(vmin=perc_vmin, vcenter=0, vmax=perc_vmax)

    all_min_tavg.plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    merra2_min.plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("MERRA2 Min. Avg.")
    ax2.coastlines()
    (((all_min_tavg - merra2_min) / merra2_min)*100).rename("%").plot(ax=ax3, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax3.set_title("MERRA2-ALL Min. Percent Differences")
    ax3.coastlines()

    all_max_tavg.plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    merra2_max.plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("MERRA2 Max. Avg.")
    ax5.coastlines()
    (((all_max_tavg - merra2_max) / merra2_max)*100).rename("%").plot(ax=ax6, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax6.set_title("MERRA2-ALL Max. Percent Differences")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_ar6_regional_avgs(time_begin: int, time_end: int, var: str, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    spatial_avg_time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 35), facecolor='w')
    f.suptitle(f"{var} Exp. {exp_num} AR6 Regional Averages from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 16}
    rc('font', **font)

    all_max_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_max_tavg.groupby(ar6.land.mask(all_max_tavg)).mean().values)]
    xghg_max_values = [value for value in xghg_max_tavg.groupby(ar6.land.mask(xghg_max_tavg)).mean().values]
    xaer_max_values = [value for value in xaer_max_tavg.groupby(ar6.land.mask(xaer_max_tavg)).mean().values]

    bar_height = 5
    y_labels = []
    legend_elements = [Patch(facecolor='blue', label='ALL'),
                      Patch(facecolor='green', label='XGHG'),
                      Patch(facecolor='red', label='XAER')]

    for region, all_value in all_max_regions:
        num = region.number
        y_labels.append(region.name)
        xghg_value = xghg_max_values[num]
        xaer_value = xaer_max_values[num]
        plot_all = lambda : ax1.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax1.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax1.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        plots.sort()
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax1.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax1.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax1.set_title("Using Max. Temperature", fontsize=30)
    ax1.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax1.set_yticklabels(y_labels)
    ax1.grid()
    ax1.legend(handles=legend_elements, loc='upper right')

    all_min_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_min_tavg.groupby(ar6.land.mask(all_min_tavg)).mean().values)]
    xghg_min_values = [value for value in xghg_min_tavg.groupby(ar6.land.mask(xghg_min_tavg)).mean().values]
    xaer_min_values = [value for value in xaer_min_tavg.groupby(ar6.land.mask(xaer_min_tavg)).mean().values]

    for region, all_value in all_min_regions:
        num = region.number
        xghg_value = xghg_min_values[num]
        xaer_value = xaer_min_values[num]
        plot_all = lambda : ax2.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax2.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax2.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax2.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax2.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax2.set_title("Using Min. Temperature", fontsize=30)
    ax2.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax2.set_yticklabels(y_labels)
    ax2.grid()
    ax2.legend(handles=legend_elements, loc='upper right')

    return f


def gen_fig_deck(variable: str, exp_num: str) -> None:
    with PdfPages(f'{variable}_{exp_num}_figure_deck.pdf') as pdf:
        print("Generating isolate signal maps for time series...")
        pdf.savefig(gen_isolated_signal_map(1930, 1960, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(1960, 1990, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(1990, 2020, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(2020, 2050, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(2050, 2080, variable, exp_num))
        print("Generating AR6 regional averages for time series...")
        pdf.savefig(gen_ar6_regional_avgs(1930, 1960, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(1960, 1990, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(1990, 2020, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(2020, 2050, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(2050, 2080, variable, exp_num))
        print("Generating global spatial average maps...")
        pdf.savefig(gen_global_spatial_avg(variable, exp_num))
        print("Generating MERRA2 comparison maps...")
        pdf.savefig(gen_merra2_comparison(variable, exp_num))
        

def gen_isolated_signal_map_ratio(numerator: str, denominator: str, time_begin: int, time_end: int, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / all_min_tavg
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / xghg_min_tavg
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / xaer_min_tavg
    all_min_tavg = all_min_tavg*100
    xghg_min_tavg = xghg_min_tavg*100
    xaer_min_tavg = xaer_min_tavg*100
    
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / all_max_tavg
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / xghg_max_tavg
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / xaer_max_tavg
    all_max_tavg = all_max_tavg*100
    xghg_max_tavg = xghg_max_tavg*100
    xaer_max_tavg = xaer_max_tavg*100
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{numerator}/{denominator} Isoalted Signal Maps EXP-{exp_num} from {time_begin} to {time_end}", fontsize=26)

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_min_tavg.rename("%").plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (all_min_tavg - xghg_min_tavg).rename("%").plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("GHG Min. Avg.")
    ax2.coastlines()
    (all_min_tavg - xaer_min_tavg).rename("%").plot(ax=ax3, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax3.set_title("AER Min. Avg.")
    ax3.coastlines()

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_max_tavg.rename("%").plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (all_max_tavg - xghg_max_tavg).rename("%").plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("GHG Max. Avg.")
    ax5.coastlines()
    (all_max_tavg - xaer_max_tavg).rename("%").plot(ax=ax6, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax6.set_title("AER Max. Avg.")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_global_spatial_avg_ratio(numerator: str, denominator: str, exp_num: str):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 25), facecolor='w')
    f.suptitle(f"{numerator}/{denominator} Exp. {exp_num} Global Spatial Average Over Time", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)

    lat_lon_avg = lambda ds : ds.mean(dim="lat").mean(dim="lon")

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days

    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / all_min_savgs.where(all_min_savgs != 0)
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / xghg_min_savgs.where(xghg_min_savgs != 0)
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / xaer_min_savgs.where(xaer_min_savgs != 0)

    all_min_savgs.plot(ax=ax1, alpha=0.7, color="green", linestyle="dotted")
    (all_min_savgs.groupby("time").mean()).plot(ax=ax1, color="green", linewidth=4, label="ALL")
    xghg_min_savgs.plot(ax=ax1, alpha=0.7, color="red", linestyle="dotted")
    (xghg_min_savgs.groupby("time").mean()).plot(ax=ax1, color="red", linewidth=4, label="XGHG")
    xaer_min_savgs.plot(ax=ax1, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_min_savgs.groupby("time").mean()).plot(ax=ax1, color="blue", linewidth=4, label="XAER")

    ax1.grid()
    ax1.set_title("Using Min. Temperature")
    ax1.legend()

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days

    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / all_max_savgs.where(all_max_savgs != 0)
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / xghg_max_savgs.where(xghg_max_savgs != 0)
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / xaer_max_savgs.where(xaer_max_savgs != 0)

    all_max_savgs.plot(ax=ax2, alpha=0.7, color="green", linestyle="dotted")
    (all_max_savgs.groupby("time").mean()).plot(ax=ax2, color="green", linewidth=4, label="ALL")
    xghg_max_savgs.plot(ax=ax2, alpha=0.7, color="red", linestyle="dotted")
    (xghg_max_savgs.groupby("time").mean()).plot(ax=ax2, color="red", linewidth=4, label="XGHG")
    xaer_max_savgs.plot(ax=ax2, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_max_savgs.groupby("time").mean()).plot(ax=ax2, color="blue", linewidth=4, label="XAER")

    ax2.grid()
    ax2.set_title("Using Max. Temperature")
    ax2.legend()

    f.tight_layout()
    return f
    
    
def gen_merra2_comparison_ratio(numerator: str, denominator: str, exp_num: str):
    time_range=[str(date) for date in list(range(1980, 2015 + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{numerator}/{denominator} ALL vs MERRA2 Comparinson EXP-{exp_num} from {1980} to {2015}", fontsize=26)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / all_min_tavg.where(all_min_tavg != 0)
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days / all_max_tavg.where(all_max_tavg != 0)
    merra2_min = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])[f"{denominator}_tn90pct"].mean(dim="time").dt.days
    merra2_min = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])[f"{numerator}_tn90pct"].mean(dim="time").dt.days / merra2_min.where(merra2_min != 0)
    merra2_min = bilinear_interpolation(merra2_min, all_min_tavg)
    merra2_max = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])[f"{denominator}_tx90pct"].mean(dim="time").dt.days
    merra2_max = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])[f"{numerator}_tx90pct"].mean(dim="time").dt.days / merra2_max.where(merra2_max != 0)
    merra2_max = bilinear_interpolation(merra2_max, all_max_tavg)

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    perc_vmin=-200
    perc_vmax=300
    perc_norm = colors.TwoSlopeNorm(vmin=perc_vmin, vcenter=0, vmax=perc_vmax)

    (all_min_tavg*100).plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (merra2_min*100).plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("MERRA2 Min. Avg.")
    ax2.coastlines()
    (((all_min_tavg - merra2_min) / merra2_min)*100).rename("%").plot(ax=ax3, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax3.set_title("MERRA2-ALL Min. Percent Differences")
    ax3.coastlines()

    (all_max_tavg*100).plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (merra2_max*100).plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("MERRA2 Max. Avg.")
    ax5.coastlines()
    (((all_max_tavg - merra2_max) / merra2_max)*100).rename("%").plot(ax=ax6, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax6.set_title("MERRA2-ALL Max. Percent Differences")
    ax6.coastlines()

    f.tight_layout()
    return f
    
    
def gen_ar6_regional_avgs_ratio(numerator: str, denominator: str, time_begin: int, time_end: int, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    spatial_avg_time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / all_min_tavg.where(all_min_tavg != 0)
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / xghg_min_tavg.where(xghg_min_tavg != 0)
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / xaer_min_tavg.where(xaer_min_tavg != 0)
    all_min_tavg = all_min_tavg*100
    xghg_min_tavg = xghg_min_tavg*100
    xaer_min_tavg = xaer_min_tavg*100

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / all_max_tavg.where(all_max_tavg != 0)
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / xghg_max_tavg.where(xghg_max_tavg != 0)
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / xaer_max_tavg.where(xaer_max_tavg != 0)
    all_max_tavg = all_max_tavg*100
    xghg_max_tavg = xghg_max_tavg*100
    xaer_max_tavg = xaer_max_tavg*100

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 35), facecolor='w')
    f.suptitle(f"{numerator}/{denominator} Exp. {exp_num} AR6 Regional Averages from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 16}
    rc('font', **font)

    all_max_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_max_tavg.groupby(ar6.land.mask(all_max_tavg)).mean().values)]
    xghg_max_values = [value for value in xghg_max_tavg.groupby(ar6.land.mask(xghg_max_tavg)).mean().values]
    xaer_max_values = [value for value in xaer_max_tavg.groupby(ar6.land.mask(xaer_max_tavg)).mean().values]

    bar_height = 5
    y_labels = []
    legend_elements = [Patch(facecolor='blue', label='ALL'),
                      Patch(facecolor='green', label='XGHG'),
                      Patch(facecolor='red', label='XAER')]

    for region, all_value in all_max_regions:
        num = region.number
        y_labels.append(region.name)
        xghg_value = xghg_max_values[num]
        xaer_value = xaer_max_values[num]
        plot_all = lambda : ax1.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax1.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax1.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax1.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax1.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax1.set_title("Using Max. Temperature (%)", fontsize=30)
    ax1.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax1.set_yticklabels(y_labels)
    ax1.grid()
    ax1.legend(handles=legend_elements, loc='upper right')

    all_min_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_min_tavg.groupby(ar6.land.mask(all_min_tavg)).mean().values)]
    xghg_min_values = [value for value in xghg_min_tavg.groupby(ar6.land.mask(xghg_min_tavg)).mean().values]
    xaer_min_values = [value for value in xaer_min_tavg.groupby(ar6.land.mask(xaer_min_tavg)).mean().values]

    for region, all_value in all_min_regions:
        num = region.number
        xghg_value = xghg_min_values[num]
        xaer_value = xaer_min_values[num]
        plot_all = lambda : ax2.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax2.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax2.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax2.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax2.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax2.set_title("Using Min. Temperature (%)", fontsize=30)
    ax2.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax2.set_yticklabels(y_labels)
    ax2.grid()
    ax2.legend(handles=legend_elements, loc='upper right')
    return f


def gen_ratio_fig_deck(numerator: str, denominator: str, exp_num: str) -> None:
    with PdfPages(f'{numerator}-{denominator}_{exp_num}_figure_deck.pdf') as pdf:
        print("Generating isolate signal maps for time series...")
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1930, 1960, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1960, 1990, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1990, 2020, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 2020, 2050, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 2050, 2080, exp_num))
        print("Generating AR6 regional averages for time series...")
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1930, 1960, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1960, 1990, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1990, 2020, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 2020, 2050, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 2050, 2080, exp_num))
        print("Generating global spatial average maps...")
        pdf.savefig(gen_global_spatial_avg_ratio(numerator, denominator, exp_num))
        print("Generating MERRA2 comparison maps...")
        pdf.savefig(gen_merra2_comparison_ratio(numerator, denominator, exp_num))

        
def gen_data_sim_1920_1950_thresholds() -> None:
    """
    Generates thresholds using 1920-1950 base period for ALL, XGHG, and XAER in appropriate directories
    This is deprecated at the momnent and will probably be removed (we use the pre-industrial control threshold now)
    """
    all_mx_datasets, xghg_mx_datasets, xaer_mx_datasets = paths.trefhtmx_members()
    all_mn_datasets, xghg_mn_datasets, xaer_mn_datasets = paths.trefhtmn_members()
    
    max_min_label = [(all_mx_datasets, all_mn_datasets, "ALL"), 
                    (xghg_mx_datasets, xghg_mn_datasets, "XGHG"), 
                    (xaer_mx_datasets, xaer_mn_datasets, "XAER")]
    
    processes = []
    num_processes = 0
    
    for max_sets, min_sets, label in max_min_label:
        for index, max_file in enumerate(max_sets):
            out_path = f"{paths.DIR_PATH}thresholds/{label}/1920_1950/{label}_{index}_90prctl_threshold.nc"
            if not isfile(out_path):
                proc = Process(target=analysis.calculate_thresholds, args=(out_path, max_file, "TREFHTMX", min_sets[index], "TREFHTMN",))
                proc.daemon = True
                proc.start()
                processes.append(proc)
                num_processes += 1
                if num_processes >= 9:
                    num_processes = 0
                    for process in processes:
                        process.join()
            else:
                print(f"Threshold already exists: {label} {index}")
    for process in processes:
        process.join()
        
        
def gen_data_control_1920_1950_threshold() -> None:
    """
    Generates control threshold in appropriate directory
    Set up for generating multiple thresholds later (will need to modify the base period)
    """
    trefht, trefhtmn, trefhtmx = paths.control_downloads()
    analysis.calculate_thresholds(paths.DIR_PATH + "thresholds/CONTROL/control_threshold.nc", trefhtmx, "TREFHTMX", trefhtmn, "TREFHTMN", bp="1920-1950", control_data=True)


def gen_data_heatwave_statistics() -> None:
    """
    Generates heatwave statistics for ALL, XGHG, XAER, and MERRA2 ensemble members in the approriate directories
    """
    all_mn, xghg_mn, xaer_mn = paths.trefhtmn_members()
    all_mx, xghg_mx, xaer_mx = paths.trefhtmx_members()
    threshold_path = paths.control_threshold()
    exp_nums = ["3114", "3336" ,"3314", "3236", "3214", "1112", "1212", "1111"]#["3136", "1312"]#["3114", "3336" ,"3314", "3236", "3214", "3136", "1112", "1212", "1312", "1111"]
    
    for num in exp_nums:
        for index, ds_mn in enumerate(all_mn):
            print(f"{num} ALL {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/ALL/1920_1950_control_base/ALL-{num}-{index}-", all_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xghg_mn):
            print(f"{num} XGHG {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/XGHG/1920_1950_control_base/XGHG-{num}-{index}-", xghg_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xaer_mn):
            print(f"{num} XAER {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/XAER/1920_1950_control_base/XAER-{num}-{index}-", xaer_mx[index], ds_mn, threshold_path, num)
            merra2_path = paths.merra2_download()
        #analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/MERRA2/heat_ouputs/MERRA2-{num}-", merra2_path, merra2_path, threshold_path, num, tmaxvname="T2MMAX", tminvname="T2MMIN", model="MERRA2")
        
        