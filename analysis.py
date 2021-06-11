from settings import RESAMPLED_YEARLY_AVG, SAMPLE_NC, POST_HEAT_THRESHOLDS_1920_TO_1950, DATA_DIR, POST_HEAT_OUTPUT_1920_1950_BASE
import xarray
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, remove, system
import imageio
from uuid import uuid4
from multiprocessing import Process, Queue
from subprocess import Popen, PIPE
import preprocessing as pp

dataset_names = listdir(RESAMPLED_YEARLY_AVG)
xaer_datasets = [name for name in dataset_names if 'XAER' in name]
xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
hist_datasets = [name for name in dataset_names if 'HIST' in name]

trefht_xaer_datasets = [name for name in xaer_datasets if 'TREFHT_' in name]
trefht_xghg_datasets = [name for name in xghg_datasets if 'TREFHT_' in name]
trefht_hist_datasets = [name for name in hist_datasets if 'TREFHT_' in name]

dataset_names = listdir(POST_HEAT_THRESHOLDS_1920_TO_1950)
threshold_xaer_datasets = [name for name in dataset_names if 'XAER' in name]
threshold_xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
threshold_hist_datasets = [name for name in dataset_names if 'ALL' in name]

print("Dataset paths loaded.")

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
    hist_mean_temps = []
    hist_temp_lists = []
    for ds_name in trefht_hist_datasets:
        ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
        ds = ds.sel(year=slice(time_slice_begin, time_slice_end))

        n_lat = ds.TREFHT.lat.size
        n_lon = ds.TREFHT.lon.size
        total_area = n_lat * n_lon

        ds_avg = ds.TREFHT.sum(dim="lat").sum(dim="lon").values / total_area
        hist_temp_lists.append(ds_avg*1)
        if len(hist_mean_temps) == 0:
            hist_mean_temps = ds_avg
        else:
            hist_mean_temps += ds_avg

    hist_mean_temps = hist_mean_temps / len(trefht_hist_datasets)
    hist_baseline = np.mean(hist_mean_temps[0:baseline_end - baseline_begin])

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

    hist_temp_avg, hist_temp_min, hist_temp_max = avg_min_max_list_of_lists(hist_temp_lists)

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
    figure_axis.plot(years, hist_mean_temps - hist_baseline, color="black", label="ALL")
    figure_axis.fill_between(years, hist_temp_min - hist_baseline, hist_temp_max - hist_baseline, alpha=0.2, color='black')
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
    years = xarray.open_dataset(RESAMPLED_YEARLY_AVG + spec_dataset_names[0]).sel(year=slice(time_slice_begin, time_slice_end)).year
    temp_lists = []
    avg_ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + spec_dataset_names[0]).sel(year=slice(time_slice_begin, time_slice_end)) * 0
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
    output_animated_yrly_trefht_based(trefht_xaer_datasets, "xaer-output.gif", "XAER", color_bar_max=3)
    output_animated_yrly_trefht_based(trefht_xghg_datasets, "xghg-output.gif", "XGHG")
    output_animated_yrly_trefht_based(trefht_hist_datasets, "all-output.gif", "ALL", color_bar_max=3)


def process_heat_thresholds_1920_1950() -> None:
    processes = []
    formers = [(pp.trefhtmax_xghg_former_em, pp.trefhtmin_xghg_former_em, "XGHG"),
               (pp.trefhtmax_xaer_former_em, pp.trefhtmin_xaer_former_em, "XAER"),
               (pp.trefhtmax_hist_former_em, pp.trefhtmin_hist_former_em, "ALL")]
    for max_em, min_em, label in formers:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = DATA_DIR + max_former_path
            min_ds_path = DATA_DIR + min_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_threshold.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {POST_HEAT_THRESHOLDS_1920_TO_1950}{label}-{index}-'
                                 + f' --t90pc --base=1920-1950 -d CESM2 -p 90 --vnamex TREFHTMX --vnamen TREFHTMN',))
            proc.daemon = True
            proc.start()
            processes.append(proc)

    for process in processes:
        process.join()


def calculate_heat_metrics_1920_1950_baseline() -> None:
    processes = []
    ensemble_members = [(pp.trefhtmax_xghg_former_em, pp.trefhtmin_xghg_former_em,
                         threshold_xghg_datasets, "former-XGHG"),
                        (pp.trefhtmax_xaer_former_em, pp.trefhtmin_xaer_former_em,
                         threshold_xaer_datasets, "former-XAER"),
                        (pp.trefhtmax_hist_former_em, pp.trefhtmin_hist_former_em,
                         threshold_hist_datasets, "former-ALL"),
                        (pp.trefhtmax_xghg_latter_em, pp.trefhtmin_xghg_latter_em,
                         threshold_xghg_datasets, "latter-XGHG"),
                        (pp.trefhtmax_xaer_latter_em, pp.trefhtmin_xaer_latter_em,
                         threshold_xaer_datasets, "latter-XAER"),
                        (pp.trefhtmax_hist_latter_em, pp.trefhtmin_hist_latter_em,
                         threshold_hist_datasets, "latter-ALL")]
    for max_em, min_em, threshold_em, label in ensemble_members:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = DATA_DIR + max_former_path
            min_ds_path = DATA_DIR + min_em[index]
            th_path = POST_HEAT_THRESHOLDS_1920_TO_1950 + threshold_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {POST_HEAT_OUTPUT_1920_1950_BASE}{label}-{index}- --thres {th_path}'
                                 + f' --base=1920-1950 -d CESM2 --vnamex TREFHTMX --vnamen TREFHTMN',))
            proc.daemon = True
            proc.start()
            processes.append(proc)
            if (index + 1) % 6 == 0:
                print(index)
                for process in processes:
                    process.join()
        for process in processes:
            process.join()

