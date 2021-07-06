import paths
from paths import HEAT_OUTPUT_1920_1950_BASE, HEAT_THRESHOLDS_1920_TO_1950, DOWNLOADED_ODATA_DIR, \
    HEAT_OUTPUT_1980_2000_BASE, CONCATENATED_ODATA, ALL_THRESHOLDS_1980_TO_2010, HEAT_OUTPUT_CONCAT_1920_1950_BASE, \
    OUT_ALL_AVGS_1980_2000
import xarray
import numpy as np
from os import listdir, remove, system
from multiprocessing import Process, Queue


def process_all_thresholds_1980_2010() -> None:
    processes = []
    em_dataset_names = listdir(CONCATENATED_ODATA)
    max_em_set = [name for name in em_dataset_names if "all" in name and "max" in name]
    min_em_set = [name for name in em_dataset_names if "all" in name and "min" in name]
    max_em_set.sort()
    min_em_set.sort()

    ensemble_sets = [(max_em_set, min_em_set, "ALL")]
    for max_em, min_em, label in ensemble_sets:
        for index, max_path in enumerate(max_em):
            max_ds_path = CONCATENATED_ODATA + max_path
            min_ds_path = CONCATENATED_ODATA + min_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_threshold.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {ALL_THRESHOLDS_1980_TO_2010}{label}-{index}-'
                                 + f' --t90pc --base=1980-2010 -d CESM2 -p 90 --vnamex TREFHTMX --vnamen TREFHTMN',))
            proc.daemon = True
            proc.start()
            processes.append(proc)

    for process in processes:
        process.join()


def process_heat_thresholds_1920_1950() -> None:
    processes = []

    max_all_former_em, max_xaer_former_em, max_xghg_former_em = paths.get_paths_trefht_odata()[0:3]
    min_all_former_em, min_xaer_former_em, min_xghg_former_em = paths.get_paths_trefht_odata()[0:3]

    formers = [(max_xghg_former_em, min_xghg_former_em, "XGHG"),
               (max_xaer_former_em, min_xaer_former_em, "XAER"),
               (max_all_former_em, min_all_former_em, "ALL")]
    for max_em, min_em, label in formers:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = DOWNLOADED_ODATA_DIR + max_former_path
            min_ds_path = DOWNLOADED_ODATA_DIR + min_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_threshold.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {HEAT_THRESHOLDS_1920_TO_1950}{label}-{index}-'
                                 + f' --t90pc --base=1920-1950 -d CESM2 -p 90 --vnamex TREFHTMX --vnamen TREFHTMN',))
            proc.daemon = True
            proc.start()
            processes.append(proc)

    for process in processes:
        process.join()


def calculate_heat_metrics_1920_1950_baseline() -> None:
    processes = []

    threshold_all_datasets, threshold_xaer_datasets, threshold_xghg_datasets = paths.get_paths_thresholds()
    max_all_em, max_xaer_em, max_xghg_em, min_all_em, min_xaer_em, min_xghg_em = paths.get_paths_cdata()[3:9]

    ensemble_members = [(max_xghg_em, min_xghg_em,
                         threshold_xghg_datasets, "XGHG"),
                        (max_xaer_em, min_xaer_em,
                         threshold_xaer_datasets, "XAER"),
                        (max_all_em, min_all_em,
                         threshold_all_datasets, "ALL")]
    for max_em, min_em, threshold_em, label in ensemble_members:
        for index, max_former_path in enumerate(max_em):
            max_ds_path = max_former_path
            min_ds_path = min_em[index]
            th_path = threshold_em[index]
            print(max_ds_path)
            print(min_ds_path)
            proc = Process(target=system,
                           args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                                 + f' --change_dir {HEAT_OUTPUT_CONCAT_1920_1950_BASE + "split/"}{label}-{index}- '
                                 + f'--thres {th_path} --base=1920-1950 -d CESM2 --vnamex TREFHTMX --vnamen TREFHTMN',))
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

    thresholds = [name for name in listdir(ALL_THRESHOLDS_1980_TO_2010) if "ALL" in name]
    min_all_ensemble_members = [name for name in listdir(CONCATENATED_ODATA) if "min_all" in name]
    max_all_ensemble_members = [name for name in listdir(CONCATENATED_ODATA) if "max_all" in name]

    thresholds.sort()
    min_all_ensemble_members.sort()
    max_all_ensemble_members.sort()

    for index, max_path in enumerate(max_all_ensemble_members):
        max_ds_path = CONCATENATED_ODATA + max_path
        min_ds_path = CONCATENATED_ODATA + min_all_ensemble_members[index]
        th_path = ALL_THRESHOLDS_1980_TO_2010 + thresholds[index]
        print(max_ds_path)
        print(min_ds_path)
        proc = Process(target=system,
                       args=(f'python3 ehfheatwaves_compound_inputthres_3.py -x {max_ds_path} -n {min_ds_path}'
                             + f' --change_dir {HEAT_OUTPUT_1980_2000_BASE}ALL-{index}-1980-2000'
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

        latter_average = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + latter_definitions[0]) * 0
        for dataset_ in latter_definitions:
            ds = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + dataset_)
            latter_average += ds
        latter_average = latter_average / len(latter_definitions)

        if "tn9pct" in latter_average:
            latter_average.drop("tn9pct")
        else:
            latter_average.drop("tx9pct")

        former_average = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + former_definitions[0]) * 0
        for dataset_ in former_definitions:
            ds = xarray.open_dataset(HEAT_OUTPUT_1920_1950_BASE + dataset_)
            former_average += ds
        former_average = former_average / len(former_definitions)

        if "tn9pct" in former_average:
            former_average.drop("tn9pct")
        else:
            former_average.drop("tx9pct")

        average = xarray.concat([former_average, latter_average], dim="time")
        average.to_netcdf(OUT_EM_AVGS_1920_1950 + full_label)

    min_former_all_datasets, min_former_xaer_datasets, min_former_xghg_datasets, \
    min_latter_all_datasets, min_latter_xaer_datasets, min_latter_xghg_datasets = paths.get_paths_trefhtmin_odata()

    max_former_all_datasets, max_former_xaer_datasets, max_former_xghg_datasets, \
    max_latter_all_datasets, max_latter_xaer_datasets, max_latter_xghg_datasets = paths.get_paths_trefhtmax_odata()

    groups = [(max_latter_all_datasets, max_former_all_datasets, "ALL-max"),
              (max_latter_xaer_datasets, max_former_xaer_datasets, "XAER-max"),
              (max_latter_xghg_datasets, max_former_xghg_datasets, "XGHG-max"),
              (min_latter_all_datasets, min_former_all_datasets, "ALL-min"),
              (min_latter_xaer_datasets, min_former_xaer_datasets, "XAER-min"),
              (min_latter_xghg_datasets, min_former_xghg_datasets, "XGHG-min")]

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
        concatenated.to_netcdf(HEAT_OUTPUT_1920_1950_BASE + concat_name)

    path = HEAT_OUTPUT_1920_1950_BASE + "split/"
    file_names = os.listdir(path)
    file_names = [name for name in file_names if 'former' in name]
    tmp = []
    for name in file_names:
        if not os.path.isfile(HEAT_OUTPUT_1920_1950_BASE + name.replace("former-", "")):
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


def average_all_1980_2000_heat_outputs() -> None:
    def process_ds(label_: str, exp_num_: str, datasets_: list) -> None:
        full_label = f"{label_}-{exp_num_}.nc"
        print(full_label)
        definitions = [ds for ds in datasets_ if exp_num_ in ds]

        average = xarray.open_dataset(definitions[0]) * 0
        for dataset_ in definitions:
            ds = xarray.open_dataset(dataset_)
            average += ds
        average = average / len(definitions)

        average.to_netcdf(OUT_ALL_AVGS_1980_2000 + full_label)

    max_all_datasets, min_all_datasets = paths.get_1980_2000_ALL_datasets()

    groups = [(max_all_datasets, "ALL-max"),
              (min_all_datasets, "ALL-min")]
    processes = []

    for datasets, label in groups:
        print(len(datasets))
        exp_nums = ["3336", "3314", "3236", "3214", "3136", "3114"]
        for exp_num in exp_nums:
            proc = Process(target=process_ds, args=(label, exp_num, datasets))
            proc.daemon = True
            proc.start()
            processes.append(proc)

        for process in processes:
            process.join()

#average_all_1980_2000_heat_outputs()

