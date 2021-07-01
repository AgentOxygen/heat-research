import os
import cartopy.crs as ccrs
from settings import RESAMPLED_YEARLY_AVG, SAMPLE_NC, POST_OUT_EM_AVGS_1920_1950, \
    POST_HEAT_THRESHOLDS_1920_TO_1950, DATA_DIR, POST_HEAT_OUTPUT_1920_1950_BASE, \
    MERRA2_DATA, FIGURE_IMAGE_OUTPUT, CONCATENATED_DATA, POST_HEAT_OUTPUT_CONCAT_1920_1950_BASE, \
    POST_ALL_THRESHOLDS_1980_TO_2000, POST_HEAT_OUTPUT_1980_2000_BASE
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from os import listdir, remove, system
import imageio
from uuid import uuid4
from multiprocessing import Process, Queue
from textwrap import wrap
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
import preprocessing as pp

load_datasets = True

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


def process_all_thresholds_1980_2000() -> None:
    processes = []
    em_dataset_names = listdir(CONCATENATED_DATA)
    max_em_set = [name for name in em_dataset_names if "all" in name and "max" in name]
    min_em_set = [name for name in em_dataset_names if "all" in name and "min" in name]
    max_em_set.sort()
    min_em_set.sort()

    ensemble_sets = [(max_em_set, min_em_set, "ALL")]
    for max_em, min_em, label in ensemble_sets:
        for index, max_path in enumerate(max_em):
            max_ds_path = CONCATENATED_DATA + max_path
            min_ds_path = CONCATENATED_DATA + min_em[index]
            print(max_ds_path)
            print(min_ds_path)
            dir_path = "/projects/dgs/persad_research/heat_research/postprocessing/1980to2010_ensemble_members/thresholds"
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_threshold.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {dir_path}{label}-{index}-'
                                 + f' --t90pc --base=1980-2010 -d CESM2 -p 90 --vnamex TREFHTMX --vnamen TREFHTMN',))
            proc.daemon = True
            proc.start()
            processes.append(proc)

    for process in processes:
        process.join()


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

    datasets = listdir(CONCATENATED_DATA)
    trefhtmax_xghg_em = [name for name in datasets if "trefhtmax_xghg" in name]
    trefhtmin_xghg_em = [name for name in datasets if "trefhtmin_xghg" in name]
    trefhtmax_xaer_em = [name for name in datasets if "trefhtmax_xaer" in name]
    trefhtmin_xaer_em = [name for name in datasets if "trefhtmin_xaer" in name]
    trefhtmax_all_em = [name for name in datasets if "trefhtmax_all" in name]
    trefhtmin_all_em = [name for name in datasets if "trefhtmin_all" in name]

    ensemble_members = [(trefhtmax_xghg_em, trefhtmin_xghg_em,
                         threshold_xghg_datasets, "XGHG"),
                        (trefhtmax_xaer_em, trefhtmin_xaer_em,
                         threshold_xaer_datasets, "XAER"),
                        (trefhtmax_all_em, trefhtmin_all_em,
                         threshold_all_datasets, "ALL")]
    for max_em, min_em, threshold_em, label in ensemble_members:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = CONCATENATED_DATA + max_former_path
            min_ds_path = CONCATENATED_DATA + min_em[index]
            th_path = POST_HEAT_THRESHOLDS_1920_TO_1950 + threshold_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {POST_HEAT_OUTPUT_CONCAT_1920_1950_BASE + "split/"}{label}-{index}- --thres {th_path}'
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


def calculate_heat_ALL_1980_2000_baseline() -> None:
    processes = []

    thresholds = [name for name in listdir(POST_ALL_THRESHOLDS_1980_TO_2000) if "ALL" in name]
    min_all_ensemble_members = [name for name in listdir(CONCATENATED_DATA) if "min_all" in name]
    max_all_ensemble_members = [name for name in listdir(CONCATENATED_DATA) if "max_all" in name]

    thresholds.sort()
    min_all_ensemble_members.sort()
    max_all_ensemble_members.sort()

    for index, max_path in enumerate(max_all_ensemble_members):
        max_ds_path = CONCATENATED_DATA + max_path
        min_ds_path = CONCATENATED_DATA + min_all_ensemble_members[index]
        th_path = POST_ALL_THRESHOLDS_1980_TO_2000 + thresholds[index]
        print(max_ds_path)
        print(min_ds_path)
        proc = Process(target=system,
                       args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                             + f' --change_dir {POST_HEAT_OUTPUT_1980_2000_BASE}ALL-{index}-1980-2000'
                               f'- --thres {th_path}'
                             + f' --base=1980-2000 -d CESM2 --vnamex TREFHTMX --vnamen TREFHTMN',))
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
        print(former_name)
        print(concat_name)
        concatenated.to_netcdf(POST_HEAT_OUTPUT_1920_1950_BASE + concat_name)

    path = POST_HEAT_OUTPUT_1920_1950_BASE + "split/"
    file_names = os.listdir(path)
    file_names = [name for name in file_names if 'former' in name]
    tmp = []
    for name in file_names:
        if not os.path.isfile(POST_HEAT_OUTPUT_1920_1950_BASE + name.replace("former-", "")):
            tmp.append(name)
    file_names = tmp

    print(f"{len(file_names)} missing netCDF files")

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


def output_heat_maps(variable:str, past_begin: str, past_end: str, fut_begin: str, fut_end: str,
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
                ds = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"{simulation}-{variant}-{exp_num}.nc")
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

    ensemble_min_avg = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"ALL-min-{exp_num}.nc")[f"{variable}_tn9pct"]\
        .sel(time=("1980", "2015"))
    ensemble_min_avg = ensemble_min_avg.sel(time=("1980", "2015")).mean(dim="time").dt.days
    ensemble_max_avg = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"ALL-max-{exp_num}.nc")[f"{variable}_tx9pct"]\
        .sel(time=("1980", "2015"))
    ensemble_max_avg = ensemble_max_avg.sel(time=("1980", "2015")).mean(dim="time").dt.days

    merra2_min = merra2_min.assign_coords(lon=((merra2_min.lon + 180) % 360))
    merra2_max = merra2_max.assign_coords(lon=((merra2_max.lon + 180) % 360))

    merra2_min = ndimage.zoom(merra2_min, (ensemble_min_avg.lat.size / merra2_min.lat.size,
                                           ensemble_min_avg.lon.size / merra2_min.lon.size))
    merra2_max = ndimage.zoom(merra2_max, (ensemble_max_avg.lat.size / merra2_max.lat.size,
                                           ensemble_max_avg.lon.size / merra2_max.lon.size))

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13))
    f.suptitle(f"{variable} Exp. {exp_num} MERRA2 Comparison (1980-2015 Avg)", fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)

    vmin = -100
    vmax = 100
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    print("Plotting..")

    ax1.set_title(f"ALL {variable} Min. Top 90 Perc.")
    ax1.pcolor(ensemble_min_avg, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

    ax2.set_title(f"MERRA2 {variable} Min. Top 90 Perc.")
    ax2.pcolor(merra2_min, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

    ax3.set_title(f"ALL vs MERRA2 {variable} Min. Top 90 Perc.")
    min_diff = ensemble_min_avg - merra2_min
    ax3.pcolor(min_diff, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

    ax4.set_title(f"ALL {variable} Max. Top 90 Perc.")
    ax4.pcolor(ensemble_max_avg, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

    ax5.set_title(f"MERRA2 {variable} Max. Top 90 Perc.")
    ax5.pcolor(merra2_max, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

    ax6.set_title(f"ALL vs MERRA2 {variable} Max. Top 90 Perc.")
    max_diff = ensemble_max_avg - merra2_max
    ax6.pcolor(max_diff, cmap='seismic', vmax=vmax, vmin=vmin, rasterized=True)

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
        all_ds = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"ALL-{variant}-{exp_num}.nc")
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
                xghg_ds = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"XGHG-{variant}-{exp_num}.nc")
                xghg_ds = xghg_ds[f"{variable}_{suffix}"]
                xghg_ds = xghg_ds.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
                data_plot = all_ds - xghg_ds
                data_plot.plot(ax=cell, cmap="seismic", norm=norm, vmax=vmax, vmin=vmin, rasterized=True)
                title = f"GHG {variant}"
            else:
                xaer_ds = xarray.open_dataset(POST_OUT_EM_AVGS_1920_1950 + f"XAER-{variant}-{exp_num}.nc")
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
    dataset_names = listdir(POST_HEAT_OUTPUT_1920_1950_BASE)
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
            data = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + ds_name)
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
    dataset_names = listdir(POST_HEAT_OUTPUT_1920_1950_BASE)
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
        data = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + ds_name)
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
        data = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + ds_name)
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
            data = xarray.open_dataset(POST_HEAT_OUTPUT_1920_1950_BASE + ds_name)
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

calculate_heat_ALL_1980_2000_baseline()
