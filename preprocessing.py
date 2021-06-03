from settings import DATA_DIR, RESAMPLED_YEARLY_AVG
import xarray
from os import listdir
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from multiprocessing import Pool

# Group datasets in directory by variable, type, and former/latter half of the time series
dataset_names = listdir(DATA_DIR)
xaer_datasets = [name for name in dataset_names if 'xaer' in name]
xghg_datasets = [name for name in dataset_names if 'xghg' in name]
all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name]

# em -> ensemble members
trefht_xaer_latter_em = [name for name in xaer_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_xaer_latter_em = [name for name in xaer_datasets if '.TREFHTMN.20060101-20801231' in name]
trefhtmax_xaer_latter_em = [name for name in xaer_datasets if '.TREFHTMX.20060101-20801231' in name]
trefht_xaer_former_em = [name for name in xaer_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_xaer_former_em = [name for name in xaer_datasets if '.TREFHTMN.19200101-20051231' in name]
trefhtmax_xaer_former_em = [name for name in xaer_datasets if '.TREFHTMX.19200101-20051231' in name]

trefht_xghg_latter_em = [name for name in xghg_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_xghg_latter_em = [name for name in xghg_datasets if '.TREFHTMN.20060101-20801231' in name]
trefhtmax_xghg_latter_em = [name for name in xghg_datasets if '.TREFHTMX.20060101-20801231' in name]
trefht_xghg_former_em = [name for name in xghg_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_xghg_former_em = [name for name in xghg_datasets if '.TREFHTMN.19200101-20051231' in name]
trefhtmax_xghg_former_em = [name for name in xghg_datasets if '.TREFHTMX.19200101-20051231' in name]

trefht_all_latter_em = [name for name in all_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_all_latter_em = [name for name in all_datasets if '.TREFHTMN.20060101-20801231' in name]
trefhtmax_all_latter_em = [name for name in all_datasets if '.TREFHTMX.20060101-20801231' in name]
trefht_all_former_em = [name for name in all_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_all_former_em = [name for name in all_datasets if '.TREFHTMN.19200101-20051231' in name]
trefhtmax_all_former_em = [name for name in all_datasets if '.TREFHTMX.19200101-20051231' in name]

print("Paths initialized")


def preprocessing_resample_yearly_avg(dataset_file_name: str) -> xarray.Dataset:
    """
    Resamples specified datasets into yearly groups, summing the daily values into a total and
    then dividing by the number of days in a year.
    """
    ds = xarray.open_dataset(DATA_DIR + dataset_file_name)
    ds = ds.resample(time="Y").sum(dim="time")
    ds = ds.groupby("time.year").sum(dim="time") / 365
    return ds


def preprocess_xaer_data() -> None:
    """
    Executes all preprocessing functions on the XAER data
    """
    def process_index(index: int) -> None:
        trefht = [preprocessing_resample_yearly_avg(trefht_xaer_former_em[index]),
                preprocessing_resample_yearly_avg(trefht_xaer_latter_em[index])]
        trefht_min = [preprocessing_resample_yearly_avg(trefhtmin_xaer_former_em[index]),
                    preprocessing_resample_yearly_avg(trefhtmin_xaer_latter_em[index])]
        trefht_max = [preprocessing_resample_yearly_avg(trefhtmax_xaer_former_em[index]),
                    preprocessing_resample_yearly_avg(trefhtmax_xaer_latter_em[index])]

        trefht = xarray.concat(trefht, dim="year")
        trefht_min = xarray.concat(trefht_min, dim="year")
        trefht_max = xarray.concat(trefht_max, dim="year")

        trefht.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHT_XAER_yearly_avg_conc_{index}.nc")
        trefht_min.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMIN_XAER_yearly_avg_conc_{index}.nc")
        trefht_max.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMAX_XAER_yearly_avg_conc_{index}.nc")

    processes = []

    for index in range(1, 21):
        proc = Process(target=process_index, args=(index,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()


def preprocess_xghg_data() -> None:
    """
    Executes all preprocessing functions on the XGHG data
    """
    def process_index(index: int, trefht_former_em: list, trefht_latter_em: list,
                      trefhtmin_former_em: list, trefhtmin_latter_em: list,
                      trefhtmax_former_em: list, trefhtmax_latter_em: list) -> None:
        trefht = [preprocessing_resample_yearly_avg(trefht_former_em[index]),
                  preprocessing_resample_yearly_avg(trefht_latter_em[index])]
        trefht_min = [preprocessing_resample_yearly_avg(trefhtmin_former_em[index]),
                      preprocessing_resample_yearly_avg(trefhtmin_latter_em[index])]
        trefht_max = [preprocessing_resample_yearly_avg(trefhtmax_former_em[index]),
                      preprocessing_resample_yearly_avg(trefhtmax_latter_em[index])]

        trefht = xarray.concat(trefht, dim="year")
        trefht_min = xarray.concat(trefht_min, dim="year")
        trefht_max = xarray.concat(trefht_max, dim="year")

        trefht.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHT_XGHG_yearly_avg_conc_{index}.nc")
        trefht_min.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMIN_XGHG_yearly_avg_conc_{index}.nc")
        trefht_max.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMAX_XGHG_yearly_avg_conc_{index}.nc")

    processes = []
    for index in range(1, 21):
        print(f"Initializing process {index}")
        proc = Process(target=process_index, args=(index, trefht_xghg_former_em, trefht_xghg_latter_em,
                                                   trefhtmin_xghg_former_em, trefhtmin_xghg_latter_em,
                                                   trefhtmax_xghg_former_em, trefhtmax_xghg_latter_em))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()

def preprocess_all_data() -> None:
    """
    Executes all preprocessing functions on the ALL data
    """
    pass

# exec = Process(target=func, args=(arg1,))
# exec.daemon = True
# exec.start()


# time_slices = []
#
# for year_data in ds.year:
#     slice_year = year_data.item()
#     time_slices.append(ds.TREFHT.sel(year=slice_year) / 365)
# year = []
# value = []
# for slice in time_slices:
#     year.append(slice.year)
#     value.append(slice.mean().item())
# plt.plot(year, value)
# plt.savefig("stuff.png")

