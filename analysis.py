from settings import RESAMPLED_YEARLY_AVG
import xarray
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, remove
import imageio
from uuid import uuid4

dataset_names = listdir(RESAMPLED_YEARLY_AVG)
xaer_datasets = [name for name in dataset_names if 'XAER' in name]
xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
hist_datasets = [name for name in dataset_names if 'HIST' in name]

trefht_xaer_datasets = [name for name in xaer_datasets if 'TREFHT_' in name]
trefht_xghg_datasets = [name for name in xghg_datasets if 'TREFHT_' in name]
trefht_hist_datasets = [name for name in hist_datasets if 'TREFHT_' in name]

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


# def avg_anomaly_map(img_output_path: str, time_slice_begin=1920, time_slice_end=2020, baseline_begin=1920,
#                     baseline_end=1970):
img_output_path = "output2.png"
time_slice_begin = 1920
time_slice_end = 2020
baseline_begin = 1920
baseline_end = 1970
# hist_mean_temps = []
# hist_temp_lists = []
# for ds_name in trefht_hist_datasets:
#     ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
#     ds = ds.sel(year=slice(time_slice_begin, time_slice_end))
#
#     n_lat = ds.TREFHT.lat.size
#     n_lon = ds.TREFHT.lon.size
#     total_area = n_lat * n_lon
#
#     ds_avg = ds.TREFHT
#     hist_temp_lists.append(ds_avg * 1)
#     if len(hist_mean_temps) == 0:
#         hist_mean_temps = ds_avg
#     else:
#         hist_mean_temps += ds_avg
#
# hist_mean_temps = hist_mean_temps / len(trefht_hist_datasets)
# hist_baseline = np.mean(hist_mean_temps[0:baseline_end - baseline_begin])
#
# xghg_year_lists = []
# xghg_temp_lists = []
# xghg_avg_ds = trefht_xghg_datasets[0]*0
# for ds_name in trefht_xghg_datasets:
#     ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
#     ds = ds.sel(year=slice(time_slice_begin, time_slice_end))
#
#     xghg_year_lists.append(ds.year.values)
#     xghg_avg_ds += ds.TREFHT / len(trefht_xghg_datasets)

xghg_years = trefht_xghg_datasets[0].year
xghg_temp_lists = []
xghg_avg_ds = trefht_xghg_datasets[0]*0
for ds_name in trefht_xghg_datasets:
    ds = xarray.open_dataset(RESAMPLED_YEARLY_AVG + ds_name)
    ds = ds.sel(year=slice(time_slice_begin, time_slice_end))

    years = ds.year.values
    xghg_avg_ds += ds.TREFHT / len(trefht_xghg_datasets)

filenames = []
for year in xghg_years:
    plt.clf()
    figure = plt.figure()
    figure_axis = figure.add_subplot()

    # create file name and append it to a list
    filename = f'{uuid4()}.png'
    filenames.append(filename)

    # save frame
    figure.savefig(filename)
    figure.close()
# build gif
with imageio.get_writer('output3.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    remove(filename)
