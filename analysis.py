import os
import cartopy.crs as ccrs
from settings import RESAMPLED_YEARLY_AVG, SAMPLE_NC, POST_OUT_EM_AVGS_1920_1950, \
    POST_HEAT_THRESHOLDS_1920_TO_1950, DATA_DIR, POST_HEAT_OUTPUT_1920_1950_BASE
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from os import listdir, remove, system
import imageio
from uuid import uuid4
from multiprocessing import Process, Queue
import preprocessing as pp

load_datasets = False
if load_datasets:
    dataset_names = listdir(RESAMPLED_YEARLY_AVG)
    xaer_datasets = [name for name in dataset_names if 'XAER' in name]
    xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
    all_datasets = [name for name in dataset_names if 'all' in name]

    trefht_xaer_datasets = [name for name in xaer_datasets if 'TREFHT_' in name]
    trefht_xghg_datasets = [name for name in xghg_datasets if 'TREFHT_' in name]
    trefht_all_datasets = [name for name in all_datasets if 'TREFHT_' in name]

    dataset_names = listdir(POST_HEAT_THRESHOLDS_1920_TO_1950)
    threshold_xaer_datasets = [name for name in dataset_names if 'XAER' in name]
    threshold_xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
    threshold_all_datasets = [name for name in dataset_names if 'ALL' in name]

    dataset_names = listdir(POST_HEAT_OUTPUT_1920_1950_BASE + "split/")
    heat_out_max_former_xaer_datasets = [name for name in dataset_names if 'former-XAER' in name and 'tx' in name]
    heat_out_max_former_xghg_datasets = [name for name in dataset_names if 'former-XGHG' in name and 'tx' in name]
    heat_out_max_former_all_datasets = [name for name in dataset_names if 'former-ALL' in name and 'tx' in name]
    heat_out_max_latter_xaer_datasets = [name for name in dataset_names if 'latter-XAER' in name and 'tx' in name]
    heat_out_max_latter_xghg_datasets = [name for name in dataset_names if 'latter-XGHG' in name and 'tx' in name]
    heat_out_max_latter_all_datasets = [name for name in dataset_names if 'latter-ALL' in name and 'tx' in name]

    heat_out_min_former_xaer_datasets = [name for name in dataset_names if 'former-XAER' in name and 'tn' in name]
    heat_out_min_former_xghg_datasets = [name for name in dataset_names if 'former-XGHG' in name and 'tn' in name]
    heat_out_min_former_all_datasets = [name for name in dataset_names if 'former-ALL' in name and 'tn' in name]
    heat_out_min_latter_xaer_datasets = [name for name in dataset_names if 'latter-XAER' in name and 'tn' in name]
    heat_out_min_latter_xghg_datasets = [name for name in dataset_names if 'latter-XGHG' in name and 'tn' in name]
    heat_out_min_latter_all_datasets = [name for name in dataset_names if 'latter-ALL' in name and 'tn' in name]

    dataset_names = listdir(POST_HEAT_OUTPUT_1920_1950_BASE)
    heat_out_max_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'max' in name]
    heat_out_max_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'max' in name]
    heat_out_max_all_datasets = [name for name in dataset_names if 'ALL' in name and 'max' in name]

    heat_out_min_xaer_datasets = [name for name in dataset_names if 'XAER' in name and 'min' in name]
    heat_out_min_xghg_datasets = [name for name in dataset_names if 'XGHG' in name and 'min' in name]
    heat_out_min_all_datasets = [name for name in dataset_names if 'ALL' in name and 'min' in name]

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
    all_mean_temps = []
    all_temp_lists = []
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
    output_animated_yrly_trefht_based(trefht_xaer_datasets, "xaer-output.gif", "XAER", color_bar_max=3)
    output_animated_yrly_trefht_based(trefht_xghg_datasets, "xghg-output.gif", "XGHG")
    output_animated_yrly_trefht_based(trefht_all_datasets, "all-output.gif", "ALL", color_bar_max=3)


def process_heat_thresholds_1920_1950() -> None:
    processes = []
    formers = [(pp.trefhtmax_xghg_former_em, pp.trefhtmin_xghg_former_em, "XGHG"),
               (pp.trefhtmax_xaer_former_em, pp.trefhtmin_xaer_former_em, "XAER"),
               (pp.trefhtmax_all_former_em, pp.trefhtmin_all_former_em, "ALL")]
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
                        (pp.trefhtmax_all_former_em, pp.trefhtmin_all_former_em,
                         threshold_all_datasets, "former-ALL"),
                        (pp.trefhtmax_xghg_latter_em, pp.trefhtmin_xghg_latter_em,
                         threshold_xghg_datasets, "latter-XGHG"),
                        (pp.trefhtmax_xaer_latter_em, pp.trefhtmin_xaer_latter_em,
                         threshold_xaer_datasets, "latter-XAER"),
                        (pp.trefhtmax_all_latter_em, pp.trefhtmin_all_latter_em,
                         threshold_all_datasets, "latter-ALL")]
    for max_em, min_em, threshold_em, label in ensemble_members:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = DATA_DIR + max_former_path
            min_ds_path = DATA_DIR + min_em[index]
            th_path = POST_HEAT_THRESHOLDS_1920_TO_1950 + threshold_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {POST_HEAT_OUTPUT_1920_1950_BASE + "split/"}{label}-{index}- --thres {th_path}'
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


def average_heat_outputs() -> None:
    def process_ds(label_: str, exp_num_: str, latter_datasets_: list, former_datasets_: list) -> None:
        full_label = f"{label_}-{exp_num_}.nc"
        print(full_label)
        latter_definitions = [ds for ds in latter_datasets_ if exp_num_ in ds]
        former_definitions = [ds for ds in former_datasets_ if exp_num_ in ds]

        latter_average = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + latter_definitions[0]) * 0
        for dataset_ in latter_definitions:
            ds = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + dataset_)
            latter_average += ds
        latter_average = latter_average / len(latter_definitions)

        if "tn9pct" in latter_average:
            latter_average.drop("tn9pct")
        else:
            latter_average.drop("tx9pct")

        former_average = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + former_definitions[0]) * 0
        for dataset_ in former_definitions:
            ds = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + dataset_)
            former_average += ds
        former_average = former_average / len(former_definitions)

        if "tn9pct" in former_average:
            former_average.drop("tn9pct")
        else:
            former_average.drop("tx9pct")

        average = xarray.concat([former_average, latter_average], dim="time")
        average.to_netcdf(POST_OUT_EM_AVGS_1920_1950 + full_label)

    groups = [(heat_out_max_latter_all_datasets, heat_out_max_former_all_datasets, "ALL-max"),
              (heat_out_max_latter_xaer_datasets, heat_out_max_former_xaer_datasets, "XAER-max"),
              (heat_out_max_latter_xghg_datasets, heat_out_max_former_xghg_datasets, "XGHG-max"),
              (heat_out_min_latter_all_datasets, heat_out_min_former_all_datasets, "ALL-min"),
              (heat_out_min_latter_xaer_datasets, heat_out_min_former_xaer_datasets, "XAER-min"),
              (heat_out_min_latter_xghg_datasets, heat_out_min_former_xghg_datasets, "XGHG-min")]

    processes = []

    for latter_datasets, former_datasets, label in groups:
        exp_nums = ["3336", "3314", "3236", "3214", "3136", "3114"]
        for exp_num in exp_nums:
            proc = Process(target=process_ds, args=(label, exp_num, latter_datasets, former_datasets))
            proc.daemon = True
            proc.start()
            processes.append(proc)

        for process in processes:
            process.join()


def concatenate_ensemble_members_heat_outputs() -> None:
    def concat(former_name: str) -> None:
        former_ds = xarray.open_dataset(path + former_name)
        latter_ds = xarray.open_dataset(path + former_name.replace("former", "latter"))
        concatenated = xarray.concat([former_ds, latter_ds], dim="time")
        concat_name = former_name.replace("former-", "")
        print(concat_name)
        concatenated.to_netcdf(POST_HEAT_OUTPUT_1920_1950_BASE + concat_name)

    path = POST_HEAT_OUTPUT_1920_1950_BASE + "split/"
    file_names = os.listdir(path)
    file_names = [name for name in file_names if 'former' in name]

    processes = []

    for index, former_name_ in enumerate(file_names):
        print(f"Process {index}")
        proc = Process(target=concat, args=(former_name_,))
        proc.daemon = True
        proc.start()
        processes.append(proc)
        if (index + 1) % 2 == 0:
            for process in processes:
                process.join()
    for process in processes:
        process.join()


def output_heat_maps(past_begin: str, past_end: str, fut_begin: str, fut_end: str, out_dir: str) -> None:
    exp_nums = ["3336", "3314", "3236", "3214", "3136", "3114"]
    simulations = ["ALL", "XGHG", "XAER"]
    variants = [("min", "tn9pct"), ("max", "tx9pct")]
    for exp_num in exp_nums:
        f, axis = plt.subplots(2, 3, figsize=(35, 13), subplot_kw=dict(projection=ccrs.PlateCarree()))
        f.suptitle(f"EXP: {exp_num} | {past_begin}-{past_end} vs {fut_begin}-{fut_end}", fontsize=30)
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
                ds = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"{simulation}-{variant}-{exp_num}.nc")
                past = ds[f"HWF_{suffix}"].sel(time=(past_begin, past_end)).mean(dim="time").astype("timedelta64[D]")
                future = ds[f"HWF_{suffix}"].sel(time=(fut_begin, fut_end)).mean(dim="time").astype("timedelta64[D]")
                perc_change = (future - past) / past
                data_plot = perc_change.plot(ax=cell)
                cell.set_title(title)
                cell.coastlines()
                iindex += 1
            index += 1

        plt.tight_layout()
        plt.savefig(f"{out_dir}/exp-{exp_num}-{past_begin}-{past_end}--{fut_begin}-{fut_end}.png")




