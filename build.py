import analysis
import paths
import xarray
import cProfile, pstats, io
from pstats import SortKey
from os.path import isfile
from multiprocessing import Process


def build_sim_1920_1950_thresholds() -> None:
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

        
def build_control_1920_1950_thresholds() -> None:
    for index, control_ds in enumerate(control_download):
        
        

def build_heatwave_statistics() -> None:
    all_ds, xghg_ds, xaer_ds = paths.trefhtmn_members()
    all_ds, xghg_ds, xaer_ds = paths.trefhtmx_members()
    all_thresh, xghg_thresh, xaer_thresh = paths.thresholds_members_1920_1950()
    analysis.calculate_heatwave_statistics("out_test.nc", tmaxfile, tminfile, thresfile, "1111")

build_control_1920_1950_thresholds()