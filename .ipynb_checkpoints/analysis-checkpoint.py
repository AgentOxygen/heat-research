import os
import cartopy.crs as ccrs
import paths
from paths import RESAMPLED_YEARLY_AVG, SAMPLE_NC, OUT_EM_AVGS_1920_1950, HEAT_OUTPUT_1920_1950_BASE, \
    MERRA2_DATA, FIGURE_IMAGE_OUTPUT, OUT_ALL_AVGS_1980_2000
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from os import listdir, remove, system
import imageio
from uuid import uuid4
from multiprocessing import Process, Queue
from matplotlib.backends.backend_pdf import PdfPages
import xesmf as xe


def avg_min_max_list_of_lists(lists: list) -> tuple:
    list_max = lists[0] * 1
    list_min = lists[0] * 1
    list_avg = lists[0] * 0
    for vals in lists:
        for index, val in enumerate(vals):
            if val > list_max[index]:
                list_max[index] = val
            if val < list_min[index]:
                list_min[index] = val
        list_avg += vals
    list_avg = list_avg / len(lists)
    return list_avg, list_min, list_max


def fig1_recreation(img_output_path: str, time_slice_begin=1920, time_slice_end=2020, baseline_begin=1920,
                    baseline_end=1970):
    all_mean_temps = []
    all_temp_lists = []
    trefht_all_datasets, trefht_xaer_datasets, trefht_xghg_datasets = paths.get_paths_annual_averages()

    for ds_name in trefht_all_datasets:
        ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
        ds = ds.sel(year=slice(time_slice_begin, time_slice_end))

        n_lat = ds.TREFHT.lat.size
        n_lon = ds.TREFHT.lon.size
        total_area = n_lat * n_lon

        ds_avg = ds.TREFHT.sum(dim="lat").sum(dim="lon").values / total_area
        all_temp_lists.append(ds_avg*1)
        if len(all_mean_temps) == 0:
            all_mean_temps = ds_avg
        else:
            all_mean_temps += ds_avg

    all_mean_temps = all_mean_temps / len(trefht_all_datasets)
    all_baseline = np.mean(all_mean_temps[0:baseline_end - baseline_begin])

    xaer_year_lists = []
    xaer_temp_lists = []
    for ds_name in trefht_xaer_datasets:
        ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
        ds = ds.sel(year=slice(time_slice_begin, time_slice_end))

        n_lat = ds.TREFHT.lat.size
        n_lon = ds.TREFHT.lon.size
        total_area = n_lat * n_lon

        xaer_year_lists.append(ds.year.values)
        xaer_temp_lists.append(ds.TREFHT.sum(dim="lat").sum(dim="lon").values / total_area)

    xghg_year_lists = []
    xghg_temp_lists = []
    for ds_name in trefht_xghg_datasets:
        ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
        ds = ds.sel(year=slice(time_slice_begin, time_slice_end))

        n_lat = ds.TREFHT.lat.size
        n_lon = ds.TREFHT.lon.size
        total_area = n_lat * n_lon

        xghg_year_lists.append(ds.year.values)
        xghg_temp_lists.append(ds.TREFHT.sum(dim="lat").sum(dim="lon").values / total_area)

    xaer_temp_avg, xaer_temp_min, xaer_temp_max = avg_min_max_list_of_lists(xaer_temp_lists)
    xaer_baseline = np.mean(xaer_temp_avg[0:50])

    xghg_temp_avg, xghg_temp_min, xghg_temp_max = avg_min_max_list_of_lists(xghg_temp_lists)
    xghg_baseline = np.mean(xghg_temp_avg[0:50])

    all_temp_avg, all_temp_min, all_temp_max = avg_min_max_list_of_lists(all_temp_lists)

    plt.clf()
    figure = plt.figure()
    figure_axis = figure.add_subplot()
    for index, years in enumerate(xaer_year_lists):
        figure_axis.plot(years, xaer_temp_lists[index], 'b', alpha=0)
    for index, years in enumerate(xghg_year_lists):
        figure_axis.plot(years, xghg_temp_lists[index], 'r', alpha=0)

    years = xghg_year_lists[0]
    figure_axis.fill_between(years, xghg_temp_min - xghg_baseline, xghg_temp_max - xghg_baseline, alpha=0.2, color='r')
    figure_axis.plot(years, xghg_temp_avg - xghg_baseline, 'r', label="XGHG")
    figure_axis.fill_between(years, xaer_temp_min - xaer_baseline, xaer_temp_max - xaer_baseline, alpha=0.2, color='b')
    figure_axis.plot(years, xaer_temp_avg - xaer_baseline, 'b', label="XAER")
    figure_axis.plot(years, all_mean_temps - all_baseline, color="black", label="ALL")
    figure_axis.fill_between(years, all_temp_min - all_baseline, all_temp_max - all_baseline,
                             alpha=0.2, color='black')
    figure_axis.yaxis.set_ticks(np.arange(-0.6, 1.6, 0.3))
    figure_axis.set_ylim(-0.8, 1.6)

    figure_axis.set_xlabel("Year")
    h = figure_axis.set_ylabel("Temp. ($^\circ$C)")
    h.set_rotation(0)
    figure_axis.legend()
    figure_axis.set_title("Recreation of Fig 1. (Rel to 1920-1970 average)")
    figure.savefig(img_output_path, bbox_inches='tight')


def output_animated_yrly_trefht_based(spec_dataset_names: list, img_output_path: str, label: str,
                                      time_slice_begin=1920, time_slice_end=2080,
                                      baseline_begin=1920, baseline_end=1970,
                                      color_bar_max=1, color_bar_min=-1) -> None:
    years = xarray.open_dataset(RESAMPLED_YEARLY_AVG + spec_dataset_names[0])\
        .sel(year=slice(time_slice_begin, time_slice_end)).year
    temp_lists = []
    avg_ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + spec_dataset_names[0])\
                 .sel(year=slice(time_slice_begin, time_slice_end)) * 0
    print("Initialized.")
    for ds_name in spec_dataset_names:
        ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
        ds = ds.sel(year=slice(time_slice_begin, time_slice_end))
        print(ds_name)
        years = ds.year.values
        avg_ds["TREFHT"] += ds.TREFHT
    avg_ds = avg_ds / len(spec_dataset_names)
    baseline = avg_ds.TREFHT.sel(year=slice(baseline_begin, baseline_end))
    baseline = baseline.sum(dim="year") / baseline.year.size
    avg_ds["TREFHT"] -= baseline
    print("Plotting...")
    filenames = Queue()

    def output_frame(ds_, year_, filenames_queue):
        print(year_)
        plt.clf()
        figure = plt.figure()
        figure_axis = figure.add_subplot()
        figure_axis.set_title(f'TREFHT Anomalies (Rel. 1920-1970) in {label}: {int(year_)}')
        fig_map = figure_axis.pcolor(ds_.TREFHT.sel(year=year_), cmap='seismic', vmax=color_bar_max, vmin=color_bar_min)
        figure.colorbar(fig_map, ax=figure_axis, format='%.0f')
        # create file name and append it to a list
        filename_ = f'{year_}-{uuid4()}.png'
        filenames_queue.put(filename_)

        # save frame
        figure.savefig(filename_)
        print(f"Initializing process {year_}")

    processes = []
    for year in years:
        proc = Process(target=output_frame, args=(avg_ds, year, filenames,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()

    filenames_list = []
    while not filenames.empty():
        name = filenames.get()
        filenames_list.append(name)
    filenames_list.sort()

    images = []
    for name in filenames_list:
        images.append(imageio.imread(name))
    kargs = {'duration': 0.3}
    imageio.mimsave(img_output_path, images, **kargs)

    # Remove files
    for filename in set(filenames_list):
        remove(filename)


def output_animated_sample() -> None:
    data = xarray.open_dataset(SAMPLE_NC).AHWN_tx9pct
    years = data.time.values
    img_output_path = "output.gif"
    label = "XGHG EM #4 Exp: 3336"
    color_bar_max = 14
    color_bar_min = 0

    print("Plotting...")
    filenames = Queue()

    def output_frame(data_slice, year_, filenames_queue):
        print(year_)
        plt.clf()
        figure = plt.figure()
        figure_axis = figure.add_subplot()
        figure_axis.set_title(f'AHWN (1920-2005) in {label}: {int(year_)}')
        fig_map = figure_axis.pcolor(data_slice, cmap='gist_heat', vmax=color_bar_max, vmin=color_bar_min)
        figure.colorbar(fig_map, ax=figure_axis, format='%.0f')
        # create file name and append it to a list
        filename_ = f'{year_}-{uuid4()}.png'
        filenames_queue.put(filename_)

        # save frame
        figure.savefig(filename_)
        print(f"Initializing process {year_}")

    processes = []
    for year in years:
        proc = Process(target=output_frame, args=(data.sel(time=year).astype('float64'), year, filenames,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()

    filenames_list = []
    while not filenames.empty():
        name = filenames.get()
        filenames_list.append(name)
    filenames_list.sort()

    images = []
    for name in filenames_list:
        images.append(imageio.imread(name))
    kargs = {'duration': 0.7}
    imageio.mimsave(img_output_path, images, **kargs)

    # Remove files
    for filename in set(filenames_list):
        remove(filename)


def output_all_animated() -> None:
    trefht_all_datasets, trefht_xaer_datasets, trefht_xghg_datasets = paths.get_paths_annual_averages()
    output_animated_yrly_trefht_based(trefht_xaer_datasets, "xaer-output.gif", "XAER", color_bar_max=3)
    output_animated_yrly_trefht_based(trefht_xghg_datasets, "xghg-output.gif", "XGHG")
    output_animated_yrly_trefht_based(trefht_all_datasets, "all-output.gif", "ALL", color_bar_max=3)


def output_heat_maps(variable: str, past_begin: str, past_end: str, fut_begin: str, fut_end: str,
                     out_dir: str, pdf=None) -> None:
    exp_nums = ["3336", "3314", "3236", "3214", "3136", "3114"]
    simulations = ["ALL", "XGHG", "XAER"]
    variants = [("min", "tn9pct"), ("max", "tx9pct")]
    for exp_num in exp_nums:
        f, axis = plt.subplots(2, 3, figsize=(35, 13), subplot_kw=dict(projection=ccrs.PlateCarree()))
        f.suptitle(f"{variable} EXP:{exp_num} | AVG:{fut_begin}-{fut_end} minus AVG:{past_begin}-{past_end} Abs. Diff.",
                   fontsize=30)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        rc('font', **font)
        index = 0
        for row in axis:
            variant, suffix = variants[index]
            iindex = 0
            for cell in row:
                simulation = simulations[iindex]
                title = f"{simulation} {variant}"
                print(f"{exp_num} {title}")
                ds = xarray.open_dataset(OUT_EM_AVGS_1920_1950 + f"{simulation}-{variant}-{exp_num}.nc")
                past = ds[f"{variable}_{suffix}"].sel(time=(past_begin, past_end)).mean(dim="time").dt.days
                future = ds[f"{variable}_{suffix}"].sel(time=(fut_begin, fut_end)).mean(dim="time").dt.days
                perc_change = (future - past)
                data_plot = perc_change.plot(ax=cell)
                cell.set_title(title)
                cell.coastlines()
                iindex += 1
            index += 1

        f.tight_layout()
        if pdf is None:
            f.savefig(f"{out_dir}/exp-{exp_num}-{past_begin}-{past_end}--{fut_begin}-{fut_end}.png")
        else:
            pdf.savefig(f)


def output_merra2_maps(exp_num: str, variable: str, out_dir: str, pdf=None) -> None:
    merra_min_path = f"tn90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
    merra_max_path = f"tx90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
    merra2_min = xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"{variable}_tn9pct"].mean(dim="time").dt.days
    merra2_max = xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"{variable}_tx9pct"].mean(dim="time").dt.days

    ensemble_min_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"{variable}_tn90pct"]\
        .sel(time=("1980", "2010"))
    ensemble_min_avg = ensemble_min_avg.sel(time=("1980", "2010")).mean(dim="time").dt.days
    ensemble_max_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"{variable}_tx90pct"]\
        .sel(time=("1980", "2010"))
    ensemble_max_avg = ensemble_max_avg.sel(time=("1980", "2010")).mean(dim="time").dt.days

    min_regridder = xe.Regridder(merra2_min, ensemble_min_avg, 'bilinear')
    max_regridder = xe.Regridder(merra2_max, ensemble_max_avg, 'bilinear')

    merra2_min = min_regridder(merra2_min)
    merra2_max = max_regridder(merra2_max)

    #merra2_min = merra2_min.assign_coords(lon=((merra2_min.lon + 180) % 360))
    #merra2_max = merra2_max.assign_coords(lon=((merra2_max.lon + 180) % 360))

    f, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(35, 13), subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{variable} Exp. {exp_num} MERRA2 Comparison (1980-2010 Avg)", fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)

    vmin = 0
    vmax = 25
    #norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    print("Plotting..")

    ensemble_min_avg.plot(ax=ax1, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax1.set_title(f"ALL {variable} Min. Top 90 Perc.")
    ax1.coastlines()

    merra2_min.plot(ax=ax2, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax2.set_title(f"MERRA2 {variable} Min. Top 90 Perc.")
    ax2.coastlines()

    ensemble_max_avg.plot(ax=ax3, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax3.set_title(f"ALL {variable} Max. Top 90 Perc.")
    ax3.coastlines()

    merra2_max.plot(ax=ax4, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax4.set_title(f"MERRA2 {variable} Max. Top 90 Perc.")
    ax4.coastlines()

    max_compare = ((ensemble_max_avg - merra2_max) / merra2_max).fillna(0)
    max_compare.plot(ax=ax5, cmap='seismic', rasterized=True)
    ax5.set_title(f"(Model - MERRA2) {variable} Max. Top 90 Perc.")
    ax5.coastlines()

    min_compare = ((ensemble_min_avg - merra2_min) / merra2_min).fillna(0)
    print(min_compare)
    min_compare.plot(ax=ax6, cmap='seismic', rasterized=True)
    ax6.set_title(f"(Model - MERRA2) {variable} Min. Top 90 Perc.")
    ax6.coastlines()

    print("Saving image...")

    f.tight_layout()
    if pdf is None:
        f.savefig(f"{out_dir}/exp-{exp_num}-{variable}-MERRA2-comparison.png")
    else:
        pdf.savefig(f)


def output_heat_signal_isolating_maps(variable: str, exp_num: str, time_begin: str, time_end: str,
                                      out_dir: str, pdf=None) -> None:
    variants = [("min", "tn9pct"), ("max", "tx9pct")]
    f, axis = plt.subplots(2, 3, figsize=(35, 13), subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{variable} EXP:{exp_num} | AVG:{time_begin}-{time_end} Abs. Value",
               fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)
    index = 0
    for row in axis:
        variant, suffix = variants[index]
        iindex = 0
        all_ds = xarray.open_dataset(OUT_EM_AVGS_1920_1950 + f"ALL-{variant}-{exp_num}.nc")
        all_ds = all_ds[f"{variable}_{suffix}"]
        all_ds = all_ds.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        vmin = -75
        vmax = 75
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        for cell in row:
            title = f"ALL {variant}"
            if iindex == 0:
                all_ds.plot(ax=cell, cmap="seismic", norm=norm, vmax=vmax, vmin=vmin, rasterized=True)
            elif iindex == 1:
                xghg_ds = xarray.open_dataset(OUT_EM_AVGS_1920_1950 + f"XGHG-{variant}-{exp_num}.nc")
                xghg_ds = xghg_ds[f"{variable}_{suffix}"]
                xghg_ds = xghg_ds.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
                data_plot = all_ds - xghg_ds
                data_plot.plot(ax=cell, cmap="seismic", norm=norm, vmax=vmax, vmin=vmin, rasterized=True)
                title = f"GHG {variant}"
            else:
                xaer_ds = xarray.open_dataset(OUT_EM_AVGS_1920_1950 + f"XAER-{variant}-{exp_num}.nc")
                xaer_ds = xaer_ds[f"{variable}_{suffix}"]
                xaer_ds = xaer_ds.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
                data_plot2 = all_ds - xaer_ds
                data_plot2.plot(ax=cell, cmap="seismic", norm=norm, vmax=vmax, vmin=vmin, rasterized=True)
                title = f"AER {variant}"
            print(f"{exp_num} {title}")
            cell.set_title(title)
            cell.coastlines()
            iindex += 1
        index += 1
    text = f"The mean number of heat wave days that occur annually, averaged over the time frame {time_begin} to" \
           f" {time_end}. The GHG and AER maps were calculated as ALL-XGHG and ALL-XAER respectively."
    f.text(0.5, 0.005, text, ha='center')
    f.tight_layout()
    if pdf is None:
        f.savefig(f"{out_dir}/exp-{exp_num}-signal_isolation-{time_begin}-{time_end}.png")
    else:
        pdf.savefig(f, dpi=70)


def spaghetti(variable: str, exp: str, out_dir: str, pdf=None) -> None:
    dataset_names = listdir(HEAT_OUTPUT_1920_1950_BASE)
    heat_out_max_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'tx' in name and exp in name]
    heat_out_max_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'tx' in name and exp in name]
    heat_out_max_all_datasets = [name for name in dataset_names if 'ALL' in name and 'tx' in name and exp in name]

    heat_out_min_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'tn' in name and exp in name]
    heat_out_min_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'tn' in name and exp in name]
    heat_out_min_all_datasets = [name for name in dataset_names if 'ALL' in name and 'tn' in name and exp in name]

    datasets_labels = [(heat_out_max_all_datasets, "max. ALL"), (heat_out_min_all_datasets, "min. ALL"),
                       (heat_out_max_xaer_datasets, "max. XAER"), (heat_out_min_xaer_datasets, "min. XAER"),
                       (heat_out_max_xghg_datasets, "max. XGHG"), (heat_out_min_xghg_datasets, "min. XGHG")]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)
    f, (ax_max, ax_min) = plt.subplots(2, 1, figsize=(15, 13))
    f.suptitle(f"Raw Value Time Series Data",
               fontsize=26)
    for ds_names, label in datasets_labels:
        max_avg = None
        min_avg = None
        color = "red"
        if "XGHG" in label:
            color = "green"
        elif "XAER" in label:
            color = "blue"
        for ds_name in ds_names:
            print(ds_name)
            data = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + ds_name)
            if "max" in label:
                data = data[f"{variable}_tx9pct"].mean(dim="lat").mean(dim="lon").dt.days
                if max_avg is None:
                    max_avg = data
                else:
                    max_avg += data
                data.plot(ax=ax_max, color=color, alpha=0.2)
            else:
                data = data[f"{variable}_tn9pct"].mean(dim="lat").mean(dim="lon").dt.days
                if min_avg is None:
                    min_avg = data
                else:
                    min_avg += data
                data.plot(ax=ax_min, color=color, alpha=0.2)
        if max_avg is not None:
            max_avg = max_avg / len(ds_names)
            max_avg.plot(ax=ax_max, color=color, label=label)
        if min_avg is not None:
            min_avg = min_avg / len(ds_names)
            min_avg.plot(ax=ax_min, color=color, label=label)
    ax_max.set_title(f"{variable} MAX. EXP {exp}")
    ax_max.legend(loc="upper left")
    ax_min.set_title(f"{variable} MIN. EXP {exp}")
    ax_min.legend(loc="upper left")

    text = f"The number of heat wave days each year, thick line represents the mean value of all the ensemble members." \
           f" Displayed are the raw values from the XGHG and XAER simulations."
    f.text(0.5, 0.005, text, wrap=True, horizontalalignment='center', fontsize=12)

    f.tight_layout()
    if pdf is None:
        f.savefig(f"{out_dir}/exp-{exp}-{variable}-spaghetti.png")
    else:
        pdf.savefig(f)


def aldente_spaghetti_differences(variable: str, exp: str, out_dir: str, pdf=None) -> None:
    dataset_names = listdir(HEAT_OUTPUT_1920_1950_BASE)
    heat_out_max_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'tx' in name and exp in name]
    heat_out_max_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'tx' in name and exp in name]
    heat_out_max_all_datasets = [name for name in dataset_names if 'ALL' in name and 'tx' in name and exp in name]

    heat_out_min_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'tn' in name and exp in name]
    heat_out_min_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'tn' in name and exp in name]
    heat_out_min_all_datasets = [name for name in dataset_names if 'ALL' in name and 'tn' in name and exp in name]

    print(f"{len(heat_out_max_xaer_datasets)} {len(heat_out_min_xaer_datasets)} \n {len(heat_out_min_xghg_datasets)} "
          f"{len(heat_out_max_xghg_datasets)} \n {len(heat_out_min_all_datasets)} {len(heat_out_max_all_datasets)}")

    datasets_labels = [(heat_out_max_xaer_datasets, "max. AER"), (heat_out_min_xaer_datasets, "min. AER"),
                       (heat_out_max_xghg_datasets, "max. GHG"), (heat_out_min_xghg_datasets, "min. GHG")]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)
    f, (ax_max, ax_min) = plt.subplots(2, 1, figsize=(15, 13))
    f.suptitle(f"Isolated Signal Time Series Data",
               fontsize=26)

    all_max_avg = None
    for ds_name in heat_out_max_all_datasets:
        print(ds_name)
        data = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + ds_name)
        data = data[f"{variable}_tx9pct"].mean(dim="lat").mean(dim="lon").dt.days
        data.plot(ax=ax_max, color="red", alpha=0.2)
        if all_max_avg is None:
            all_max_avg = data
        else:
            all_max_avg += data
    all_max_avg = all_max_avg / len(heat_out_max_all_datasets)
    all_max_avg.plot(ax=ax_max, color="red", label="max. ALL")

    all_min_avg = None
    for ds_name in heat_out_min_all_datasets:
        print(ds_name)
        data = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + ds_name)
        data = data[f"{variable}_tn9pct"].mean(dim="lat").mean(dim="lon").dt.days
        data.plot(ax=ax_min, color="red", alpha=0.2)
        if all_min_avg is None:
            all_min_avg = data
        else:
            all_min_avg += data
    all_min_avg = all_min_avg / len(heat_out_min_all_datasets)
    all_min_avg.plot(ax=ax_min, color="red", label="min. ALL")

    for ds_names, label in datasets_labels:
        max_avg = None
        min_avg = None
        color = "red"
        if "GHG" in label:
            color = "green"
        elif "AER" in label:
            color = "blue"
        for ds_name in ds_names:
            print(ds_name)
            data = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + ds_name)
            if "max" in label:
                data = all_max_avg - data[f"{variable}_tx9pct"].mean(dim="lat").mean(dim="lon").dt.days
                if max_avg is None:
                    max_avg = data
                else:
                    max_avg += data
                data.plot(ax=ax_max, color=color, alpha=0.2)
            else:
                data = all_min_avg - data[f"{variable}_tn9pct"].mean(dim="lat").mean(dim="lon").dt.days
                if min_avg is None:
                    min_avg = data
                else:
                    min_avg += data
                data.plot(ax=ax_min, color=color, alpha=0.2)

        if max_avg is not None:
            max_avg = max_avg / len(ds_names)
            max_avg.plot(ax=ax_max, color=color, label=label)
        if min_avg is not None:
            min_avg = min_avg / len(ds_names)
            min_avg.plot(ax=ax_min, color=color, label=label)
    ax_max.set_title(f"{variable} MAX. EXP {exp}")
    ax_max.legend(loc="upper left")
    ax_min.set_title(f"{variable} MIN. EXP {exp}")
    ax_min.legend(loc="upper left")

    text = f"The number of heat wave days each year, thick line represents the mean value of all the ensemble members."\
           f" GHG and AER were calculated from ALL-XGHG and ALL-XAER respectively. "

    f.text(0.5, 0.005, text, wrap=True, horizontalalignment='center', fontsize=12)

    f.tight_layout()
    if pdf is None:
        f.savefig(f"{out_dir}/exp-{exp}-{variable}-aldente-spaghetti-differences.png")
    else:
        pdf.savefig(f)


def generate_figure_images() -> None:
    var_exp_list = [("HWF", "3336"),
                    ("HWF", "3314"),
                    ("HWF", "3236"),
                    ("HWF", "3136"),
                    ("HWF", "3114")]

    time_ranges = [("1960", "1990"), ("1990", "2020"), ("2020", "2050"), ("2050", "2080")]

    processes = []
    for variable, exp_num in var_exp_list:
        proc = Process(target=spaghetti,
                       args=(variable, exp_num, FIGURE_IMAGE_OUTPUT))
        proc.daemon = True
        proc.start()
        processes.append(proc)
    for process in processes:
        process.join()

    for variable, exp_num in var_exp_list:
        proc = Process(target=aldente_spaghetti_differences,
                       args=(variable, exp_num, FIGURE_IMAGE_OUTPUT))
        proc.daemon = True
        proc.start()
        processes.append(proc)
    for process in processes:
        process.join()

    for t_b, t_e in time_ranges:
        processes = []
        for variable, exp_num in var_exp_list:
            proc = Process(target=output_heat_signal_isolating_maps,
                           args=(variable, exp_num, t_b, t_e, FIGURE_IMAGE_OUTPUT))
            proc.daemon = True
            proc.start()
            processes.append(proc)
        for process in processes:
            process.join()


def generate_figure_pdf() -> None:
    var_exp_list = [("HWF", "3336"),
                    ("HWF", "3314"),
                    ("HWF", "3236"),
                    ("HWF", "3136"),
                    ("HWF", "3114")]

    time_ranges = [("1960", "1990"), ("1990", "2020"), ("2020", "2050"), ("2050", "2080")]

    processes = []

    def spaghetti_output(var_exp_list_):
        with PdfPages(FIGURE_IMAGE_OUTPUT + 'spaghettis.pdf') as pdf:
            print("SPAGHETTI")
            for variable, exp_num in var_exp_list_:
                spaghetti(variable, exp_num, FIGURE_IMAGE_OUTPUT, pdf)

    def aldente_spaghetti_output(var_exp_list_):
        with PdfPages(FIGURE_IMAGE_OUTPUT + 'aldente_spaghettis.pdf') as pdf:
            print("ALDENTE SPAGHETTI")
            for variable, exp_num in var_exp_list_:
                aldente_spaghetti_differences(variable, exp_num, FIGURE_IMAGE_OUTPUT, pdf)

    def heat_signal_isolations_output(var_exp_list_, time_ranges_):
        with PdfPages(FIGURE_IMAGE_OUTPUT + 'heat_signal_isolations.pdf') as pdf:
            print("HEAT ISOLATION")
            for t_b, t_e in time_ranges_:
                for variable, exp_num in var_exp_list_:
                    output_heat_signal_isolating_maps(variable, exp_num, t_b, t_e, FIGURE_IMAGE_OUTPUT, pdf)

    proc = Process(target=spaghetti_output, args=(var_exp_list,))
    proc.daemon = True
    processes.append(proc)
    proc = Process(target=aldente_spaghetti_output, args=(var_exp_list,))
    proc.daemon = True
    processes.append(proc)
    proc = Process(target=heat_signal_isolations_output, args=(var_exp_list, time_ranges,))
    proc.daemon = True
    processes.append(proc)
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def bar_graphs() -> None:
    all_dataset = xarray.open_dataset(paths.get_paths_heat_output_avg()[0][0])


output_merra2_maps("3114", "HWF", FIGURE_IMAGE_OUTPUT)


