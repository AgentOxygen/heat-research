from settings import DATA_DIR, RESAMPLED_YEARLY_AVG, TIME_SLICED_1920_TO_1950
import xarray
from os import listdir
from multiprocessing import Process

# Group datasets in directory by variable, type, and former/latter half of the time series
dataset_names = listdir(DATA_DIR)
xaer_datasets = [name for name in dataset_names if 'xaer' in name]
xghg_datasets = [name for name in dataset_names if 'xghg' in name]
all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name or 'B20TRC5CNBDRD' in name]

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

print("File Name Lists populated.")


def preprocess_time_slice(start: str, end: str, dataset_file_name: str) -> xarray.Dataset:
    """
    Slices specified dataset using specified start and end dates and returns the slice
    """
    ds = xarray.open_dataset(DATA_DIR + dataset_file_name)
    return ds.sel(time=slice(start, end))


def preprocess_output_time_slice(start: str, end: str, dataset_file_name: str, output_file_name: str) -> None:
    """
    Slices specified dataset using specified start and end dates and outputs the slice
    """
    preprocess_time_slice(start, end, dataset_file_name).to_netcdf(output_file_name)


def preprocess_resample_yearly_avg(dataset_file_name: str) -> xarray.Dataset:
    """
    Resamples specified datasets into yearly groups, summing the daily values into a total and
    then dividing by the number of days in a year.
    """
    ds = xarray.open_dataset(DATA_DIR + dataset_file_name)
    ds = ds.resample(time="Y").sum(dim="time")
    ds = ds.groupby("time.year").sum(dim="time") / 365
    return ds


def resample_xaer_data() -> None:
    """
    Resamples XAER data into annual averages
    """
    def process_index(index: int) -> None:
        trefht = [preprocess_resample_yearly_avg(trefht_xaer_former_em[index]),
                preprocess_resample_yearly_avg(trefht_xaer_latter_em[index])]
        trefht_min = [preprocess_resample_yearly_avg(trefhtmin_xaer_former_em[index]),
                    preprocess_resample_yearly_avg(trefhtmin_xaer_latter_em[index])]
        trefht_max = [preprocess_resample_yearly_avg(trefhtmax_xaer_former_em[index]),
                    preprocess_resample_yearly_avg(trefhtmax_xaer_latter_em[index])]

        trefht = xarray.concat(trefht, dim="year")
        trefht_min = xarray.concat(trefht_min, dim="year")
        trefht_max = xarray.concat(trefht_max, dim="year")

        trefht.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHT_XAER_yearly_avg_conc_{index}.nc")
        trefht_min.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMIN_XAER_yearly_avg_conc_{index}.nc")
        trefht_max.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMAX_XAER_yearly_avg_conc_{index}.nc")

    processes = []

    for p_index in range(1, 21):
        proc = Process(target=process_index, args=(p_index,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()


def resample_xghg_data() -> None:
    """
    Resamples XGHG data into annual averages
    """
    def process_index(index: int) -> None:
        trefht = [preprocess_resample_yearly_avg(trefht_xghg_former_em[index]),
                  preprocess_resample_yearly_avg(trefht_xghg_latter_em[index])]
        trefht_min = [preprocess_resample_yearly_avg(trefhtmin_xghg_former_em[index]),
                      preprocess_resample_yearly_avg(trefhtmin_xghg_latter_em[index])]
        trefht_max = [preprocess_resample_yearly_avg(trefhtmax_xghg_former_em[index]),
                      preprocess_resample_yearly_avg(trefhtmax_xghg_latter_em[index])]

        trefht = xarray.concat(trefht, dim="year")
        trefht_min = xarray.concat(trefht_min, dim="year")
        trefht_max = xarray.concat(trefht_max, dim="year")

        trefht.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHT_XGHG_yearly_avg_conc_{index}.nc")
        trefht_min.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMIN_XGHG_yearly_avg_conc_{index}.nc")
        trefht_max.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMAX_XGHG_yearly_avg_conc_{index}.nc")

    processes = []
    for p_index in range(1, 21):
        print(f"Initializing process {p_index}")
        proc = Process(target=process_index, args=(p_index,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()


def resample_all_data() -> None:
    """
    Resamples ALL data into annual averages
    """
    def process_index(index: int) -> None:
        trefht = [preprocess_resample_yearly_avg(trefht_all_former_em[index]),
                  preprocess_resample_yearly_avg(trefht_all_latter_em[index])]
        trefht_min = [preprocess_resample_yearly_avg(trefhtmin_all_former_em[index]),
                      preprocess_resample_yearly_avg(trefhtmin_all_latter_em[index])]
        trefht_max = [preprocess_resample_yearly_avg(trefhtmax_all_former_em[index]),
                      preprocess_resample_yearly_avg(trefhtmax_all_latter_em[index])]

        trefht = xarray.concat(trefht, dim="year")
        trefht_min = xarray.concat(trefht_min, dim="year")
        trefht_max = xarray.concat(trefht_max, dim="year")

        trefht.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHT_ALL_yearly_avg_conc_{index}.nc")
        trefht_min.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMIN_ALL_yearly_avg_conc_{index}.nc")
        trefht_max.to_netcdf(f"{RESAMPLED_YEARLY_AVG}TREFHTMAX_ALL_yearly_avg_conc_{index}.nc")

    processes = []
    for p_index in range(1, 21):
        print(f"Initializing process {p_index}")
        proc = Process(target=process_index, args=(p_index,))
        proc.daemon = True
        proc.start()
        processes.append(proc)

    for process in processes:
        process.join()


def slice_former_ensemble_datasets(start_date: str, end_date: str, output_dir: str) -> None:
    processes = []
    ensemble_samples = [trefht_xaer_former_em, trefhtmin_xaer_former_em, trefhtmax_xaer_former_em,
                        trefht_xghg_former_em, trefhtmin_xghg_former_em, trefhtmax_xghg_former_em,
                        trefht_all_former_em, trefhtmin_all_former_em, trefhtmax_all_former_em]
    labels = ["TREFHT_XAER", "TREFHTMIN_XAER", "TREFHTMAX_XAER",
              "TREFHT_XGHG", "TREFHTMIN_XGHG", "TREFHTMAX_XGHG",
              "TREFHT_ALL", "TREFHTMIN_ALL", "TREFHTMAX_ALL"]

    for index, em in enumerate(ensemble_samples):
        for iindex, ds in enumerate(em):
            proc = Process(target=preprocess_output_time_slice,
                           args=(start_date, end_date, ds, f"{output_dir}{start_date}_to_{end_date}_{labels[index]}_{iindex}.nc"))
            proc.daemon = True
            proc.start()
            processes.append(proc)

    for process in processes:
        process.join()
