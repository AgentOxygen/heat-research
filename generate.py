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


# def gen_fig_signal_maps(axis: matplotlib.axes._subplots.AxesSubplot, exp_num: str) -> None:
#     pass


# def gen_fig_spaghetti(ax_max, ax_min, exp_num: str) -> None:
#     max_all_datasets, max_xghg_datasets, max_xaer_datasets = paths.heat_out_trefht_tmax_members_1920_1950()
#     min_all_datasets, min_xghg_datasets, min_xaer_datasets = paths.heat_out_trefht_tmin_members_1920_1950()

#     print(f"{len(max_xaer_datasets)} {len(min_xaer_datasets)} \n {len(min_xghg_datasets)} "
#           f"{len(max_xghg_datasets)} \n {len(min_all_datasets)} {len(max_all_datasets)}")

#     datasets_labels = [(max_xaer_datasets, "max. AER"), (min_xaer_datasets, "min. AER"),
#                        (max_xghg_datasets, "max. GHG"), (min_xghg_datasets, "min. GHG")]

#     all_max_avg = None
#     for ds_name in max_all_datasets:
#         data = xarray.open_dataset(ds_name)
#         if variable == "AHW2F/AHWF":
#             data = data[f"AHW2F_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#             divisor = xarray.open_dataset(ds_name)[f"AHWF_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#             data = data.where(divisor != 0, drop=True) / divisor
#         else:
#             data = data[f"{variable}_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#         data.plot(ax=ax_max, color="red", alpha=0.2)
#         if all_max_avg is None:
#             all_max_avg = data
#         else:
#             all_max_avg += data
#     all_max_avg = all_max_avg / len(max_all_datasets)
#     all_max_avg.plot(ax=ax_max, color="red", label="max. ALL")

#     all_min_avg = None
#     for ds_name in min_all_datasets:
#         data = xarray.open_dataset(ds_name)
#         if variable == "AHW2F/AHWF":
#             data = data[f"AHW2F_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#             divisor = xarray.open_dataset(ds_name)[f"AHWF_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#             data = data.where(divisor != 0, drop=True) / divisor
#         else:
#             data = data[f"{variable}_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#         data.plot(ax=ax_min, color="red", alpha=0.2)
#         if all_min_avg is None:
#             all_min_avg = data
#         else:
#             all_min_avg += data
#     all_min_avg = all_min_avg / len(min_all_datasets)
#     all_min_avg.plot(ax=ax_min, color="red", label="min. ALL")

#     for ds_names, label in datasets_labels:
#         max_avg = None
#         min_avg = None
#         color = "red"
#         if "GHG" in label:
#             color = "green"
#         elif "AER" in label:
#             color = "blue"
#         for ds_name in ds_names:
#             data = xarray.open_dataset(ds_name)
#             if "max" in label:
#                 if variable == "AHW2F/AHWF":
#                     divisor = data[f"AHWF_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#                     data = all_max_avg - (data[f"AHW2F_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days.where(divisor != 0, drop=True) / divisor)
#                 else:
#                     data = all_max_avg - data[f"{variable}_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#                 if max_avg is None:
#                     max_avg = data
#                 else:
#                     max_avg += data
#                 data.plot(ax=ax_max, color=color, alpha=0.2)
#             else:
#                 if variable == "AHW2F/AHWF":
#                     divisor = data[f"AHWF_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#                     data = all_max_avg - (data[f"AHW2F_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days / divisor)
#                 else:
#                     data = all_max_avg - data[f"{variable}_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
#                 if min_avg is None:
#                     min_avg = data
#                 else:
#                     min_avg += data
#                 data.plot(ax=ax_min, color=color, alpha=0.2)

#         if max_avg is not None:
#             max_avg = max_avg / len(ds_names)
#             max_avg.plot(ax=ax_max, color=color, label=label)
#         if min_avg is not None:
#             min_avg = min_avg / len(ds_names)
#             min_avg.plot(ax=ax_min, color=color, label=label)
#     ax_max.set_title(f"{variable} MAX. EXP {exp}")
#     ax_max.legend(loc="upper left")
#     ax_min.set_title(f"{variable} MIN. EXP {exp}")
#     ax_min.legend(loc="upper left")


# def gen_fig_bar_graph(axis: matplotlib.axes._subplots.AxesSubplot, exp_num: str, region_set: regionmask.core.regions.Regions) -> None:
#     pass


# def gen_fig_merra2_comparison(axis: matplotlib.axes._subplots.AxesSubplot, exp_num: str) -> None:
#     merra_min_path = f"tn90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
#     merra_max_path = f"tx90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
#     if variable == "AHW2F/AHWF":
#         merra2_min = xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"AHW2F_tn9pct"].mean(dim="time").dt.days
#         merra2_min = merra2_min / xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"AHWF_tn9pct"].mean(dim="time").dt.days
#         merra2_max = xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"AHW2F_tx9pct"].mean(dim="time").dt.days
#         merra2_max = merra2_max / xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"AHWF_tx9pct"].mean(dim="time").dt.days
        
#         ensemble_min_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"AHW2F_tn90pct"].sel(time=("1980", "2010")).dt.days
#         ensemble_min_avg = ensemble_min_avg / xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"AHWF_tn90pct"].sel(time=("1980", "2010")).dt.days
#         ensemble_max_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"AHW2F_tx90pct"].sel(time=("1980", "2010")).dt.days
#         ensemble_max_avg = ensemble_max_avg / xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"AHWF_tx90pct"].sel(time=("1980", "2010")).dt.days
#         ensemble_min_avg = ensemble_min_avg.mean(dim="time").fillna(0)*100
#         ensemble_max_avg = ensemble_max_avg.mean(dim="time").fillna(0)*100
#         print(ensemble_max_avg)
#     else:
#         merra2_min = xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"{variable}_tn9pct"].mean(dim="time").dt.days
#         merra2_max = xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"{variable}_tx9pct"].mean(dim="time").dt.days
        
#         ensemble_min_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"{variable}_tn90pct"].sel(time=("1980", "2010"))
#         ensemble_max_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"{variable}_tx90pct"].sel(time=("1980", "2010"))
        
#         ensemble_min_avg = ensemble_min_avg.mean(dim="time").dt.days
#         ensemble_max_avg = ensemble_max_avg.mean(dim="time").dt.days

#     min_regridder = xe.Regridder(merra2_min, ensemble_min_avg, 'bilinear')
#     max_regridder = xe.Regridder(merra2_max, ensemble_max_avg, 'bilinear')

#     merra2_min = min_regridder(merra2_min)
#     merra2_max = max_regridder(merra2_max)
    
#     f, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
#     f.suptitle(f"{variable} Exp. {exp_num} MERRA2 Comparison (1980-2010 Avg)", fontsize=30)
#     font = {'family': 'normal',
#             'weight': 'bold',
#             'size': 22}
#     rc('font', **font)

#     if v1mx is not None and v1mn is not None:
#         vmax = v1mx
#         vmin = v1mn
#     else:
#         vmin = 0
#         vmax = 20
    
#     ensemble_min_avg.plot(ax=ax1, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
#     ax1.set_title(f"ALL {variable} Min. Top 90 Perc.")
#     ax1.coastlines()

#     merra2_min.plot(ax=ax2, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
#     ax2.set_title(f"MERRA2 {variable} Min. Top 90 Perc.")
#     ax2.coastlines()

#     ensemble_max_avg.plot(ax=ax3, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
#     ax3.set_title(f"ALL {variable} Max. Top 90 Perc.")
#     ax3.coastlines()

#     merra2_max.plot(ax=ax4, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
#     ax4.set_title(f"MERRA2 {variable} Max. Top 90 Perc.")
#     ax4.coastlines()

#     if v2mx is not None and v2mn is not None:
#         vmax = v2mx
#         vmin = v2mn
#     else:
#         vmin = -200
#         vmax = 200
#     norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

#     max_compare = ((ensemble_max_avg - merra2_max) / merra2_max) * 100
#     max_compare = max_compare.rename("Perc. Change %")
#     max_compare.where(max_compare < vmax).plot(ax=ax5, cmap='seismic', vmax=vmax, vmin=vmin, norm=norm, rasterized=True)
#     ax5.set_title(f"(Model - MERRA2) / MERRA2 {variable} Max. Top 90 Perc.")
#     ax5.coastlines()

#     min_compare = ((ensemble_min_avg - merra2_min) / merra2_min) * 100
#     min_compare = min_compare.rename("Perc. Change %")
#     min_compare.where(min_compare < vmax).plot(ax=ax6, cmap='seismic', vmax=vmax, vmin=vmin, norm=norm, rasterized=True)
#     ax6.set_title(f"(Model - MERRA2) / MERRA2 {variable} Min. Top 90 Perc.")
#     ax6.coastlines()

#     f.tight_layout()
#     if pdf is None:
#         variable = variable.replace("/", "-")
#         f.savefig(f"{FIGURE_IMAGE_OUTPUT}/exp-{exp_num}-{variable}-MERRA2-comparison.png")
#     else:
#         pdf.savefig(f)


# def gen_fig_deck() -> None:
#     pass


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
    exp_nums = ["3114", "3336" ,"3314", "3236", "3214", "3136", "1112", "1212", "1312", "1111"]
    
    for num in exp_nums:
        for index, ds_mn in enumerate(all_mn):
            print(f"{num} ALL {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/ALL/1920_1950_control_base/ALL-{num}-{index}-", all_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xghg_mn):
            print(f"{num} XGHG {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/ALL/1920_1950_control_base/XGHG-{num}-{index}-", xghg_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xaer_mn):
            print(f"{num} XAER {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/ALL/1920_1950_control_base/XAER-{num}-{index}-", xaer_mx[index], ds_mn, threshold_path, num)
            merra2_path = paths.merra2_download()
        analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/MERRA2/heat_ouputs/MERRA2-{num}-", merra2_path, merra2_path, threshold_path, num, tmaxvname="T2MMAX", tminvname="T2MMIN", model="MERRA2")

        
gen_data_heatwave_statistics()