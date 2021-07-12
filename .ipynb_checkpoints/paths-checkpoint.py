# Contains necessary settings and paths for accessing data
# "ODATA" refers to the "original data"
# "CDATA" refers to the "concatenated original data" (both time series stitched together)
# "ALL" refers to the ALL-forcing ensemble datafiles
# "EM" refers to "ensemble members"
# "{YEAR}_{YEAR}_BASE" indicates the base period used for calculating the heat stats
import warnings
import functools
import xarray
from os import listdir

DOWNLOADED_ODATA_DIR = "/projects/dgs/persad_research/heat_research/data/"
RESAMPLED_YEARLY_AVG = "/projects/dgs/persad_research/heat_research/preprocessing/resampled_yearly_avg/"
TIME_SLICED_ODATA_1920_TO_1950 = "/projects/dgs/persad_research/heat_research/preprocessing/resampled_1920to1950_slice/"
CONCATENATED_ODATA = "/projects/dgs/persad_research/heat_research/preprocessing/concatenated_data/"
HEAT_THRESHOLDS_1920_TO_1950 = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                                    "1920to1950_ensemble_members/thresholds/"
ALL_THRESHOLDS_1980_TO_2010 = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                                   "1980to2010_ensemble_members/"
HEAT_OUTPUT_1920_1950_BASE = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                                  "heat_output_1920_1950_baseline/"
HEAT_OUTPUT_1980_2000_BASE = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                                   "heat_output_all_1980_2000_baseline/"
HEAT_OUTPUT_CONCAT_1920_1950_BASE = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                                         "heat_output_1920_1950_baseline/concatted/"
OUT_EM_AVGS_1920_1950 = "/projects/dgs/persad_research/heat_research/postprocessing/" \
                             "heat_ensemble_averages_1920_1950/"
OUT_ALL_AVGS_1980_2000 = "/projects/dgs/persad_research/heat_research/postprocessing/heat_ALL_averages_1980_2000/"
SAMPLE_NC = "/projects/dgs/persad_research/heat_research/postprocessing/heat_output_1920_1950_baseline/" \
            "former-XGHG-4-tx9pct_heatwaves_CESM2_rNone_3336_yearly_summer.nc"
MERRA2_DATA = "/projects/dgs/persad_research/heat_research/postprocessing/MERRA2_shift_0_heat_output/"
FIGURE_IMAGE_OUTPUT = "/home/persad_users/csc3323/figure_outputs/"


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"Call to deprecated function {func.__name__}",
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)
    return new_func


def get_paths_annual_averages(trefht_type: str = "TREFHT_") -> tuple:
    """
    Values averaged for each year from 1920 to 2080. Useful for inter-annual analysis.

    :param trefht_type: Used for filtering files: "TREFHT_", "TREFHTMAX", or "TREFHTMIN"
    :return: Three lists of ALL, XAER, and XGHG in that order, each containing paths to the netCDF files
    """
    dataset_names = listdir(RESAMPLED_YEARLY_AVG)
    xaer_datasets = [name for name in dataset_names if 'XAER' in name]
    xghg_datasets = [name for name in dataset_names if 'XGHG' in name]
    all_datasets = [name for name in dataset_names if 'all' in name]
    trefht_xaer_datasets = [RESAMPLED_YEARLY_AVG + name for name in xaer_datasets if trefht_type in name]
    trefht_xghg_datasets = [RESAMPLED_YEARLY_AVG + name for name in xghg_datasets if trefht_type in name]
    trefht_all_datasets = [RESAMPLED_YEARLY_AVG + name for name in all_datasets if trefht_type in name]
    return trefht_all_datasets, trefht_xaer_datasets, trefht_xghg_datasets


def get_paths_thresholds() -> tuple:
    """
    Heat thresholds calculated by "ehfheatwaves_threshold.py"

    :return: Three lists of ALL, XAER, and XGHG in that order, each containing paths to the netCDF files
    """
    dataset_names = listdir(HEAT_THRESHOLDS_1920_TO_1950)
    threshold_xaer_datasets = [HEAT_THRESHOLDS_1920_TO_1950 + name for name in dataset_names if 'XAER' in name]
    threshold_xghg_datasets = [HEAT_THRESHOLDS_1920_TO_1950 + name for name in dataset_names if 'XGHG' in name]
    threshold_all_datasets = [HEAT_THRESHOLDS_1920_TO_1950 + name for name in dataset_names if 'ALL' in name]
    return threshold_all_datasets, threshold_xaer_datasets, threshold_xghg_datasets


@deprecated # Use concatenated output instead
def get_paths_split_heat_output(tmp_type: str = "tn") -> tuple:
    """
    Heat outputs calculated by "ehfheatwaves_compound_inputthres_3.py" for the split time series data (former/latter).

    :param tmp_type: Used for filtering files: "tn" or "tx"
    :return: Six lists for Former (1920-2005) ALL, XAER, XGHG and Latter (2006-2080) ALL, XAER, XGHG in that order,
    each containing paths to the netCDF files.
    """
    dir_path = HEAT_OUTPUT_1920_1950_BASE + "split/"
    dataset_names = listdir(dir_path)
    min_former_xaer_datasets = [dir_path + name for name in dataset_names if 'former-XAER' in name and tmp_type in name]
    min_former_xghg_datasets = [dir_path + name for name in dataset_names if 'former-XGHG' in name and tmp_type in name]
    min_former_all_datasets = [dir_path + name for name in dataset_names if 'former-ALL' in name and tmp_type in name]
    min_latter_xaer_datasets = [dir_path + name for name in dataset_names if 'latter-XAER' in name and tmp_type in name]
    min_latter_xghg_datasets = [dir_path + name for name in dataset_names if 'latter-XGHG' in name and tmp_type in name]
    min_latter_all_datasets = [dir_path + name for name in dataset_names if 'latter-ALL' in name and tmp_type in name]
    return min_former_all_datasets, min_former_xaer_datasets, min_former_xghg_datasets, \
           min_latter_all_datasets, min_latter_xaer_datasets, min_latter_xghg_datasets


def get_paths_trefht_odata() -> tuple:
    """
    Original datasets as downloaded for TREFHT

    :return: Three lists for ALL, XAER, and XGHG in that order, each containing paths to the netCDF files.
    """
    dataset_names = listdir(DOWNLOADED_ODATA_DIR)
    xaer_datasets = [name for name in dataset_names if 'xaer' in name]
    xghg_datasets = [name for name in dataset_names if 'xghg' in name]
    all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name or 'B20TRC5CNBDRD' in name]

    # em -> ensemble members
    trefht_all_former_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                            if '.TREFHT.19200101-20051231' in name]
    trefht_xaer_former_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                             if '.TREFHT.19200101-20051231' in name]
    trefht_xghg_former_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                             if '.TREFHT.19200101-20051231' in name]
    trefht_all_latter_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                            if '.TREFHT.20060101-20801231' in name]
    trefht_xaer_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                             if '.TREFHT.20060101-20801231' in name]
    trefht_xghg_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                             if '.TREFHT.20060101-20801231' in name]

    return trefht_all_former_em, trefht_xaer_former_em, trefht_xghg_former_em, \
           trefht_all_latter_em, trefht_xaer_latter_em, trefht_xghg_latter_em


def get_paths_trefhtmin_odata() -> tuple:
    """
    Original datasets as downloaded for TREFHTMN

    :return: Three lists for ALL, XAER, and XGHG in that order, each containing paths to the netCDF files.
    """
    dataset_names = listdir(DOWNLOADED_ODATA_DIR)
    xaer_datasets = [name for name in dataset_names if 'xaer' in name]
    xghg_datasets = [name for name in dataset_names if 'xghg' in name]
    all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name or 'B20TRC5CNBDRD' in name]

    trefhtmin_all_former_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                               if '.TREFHTMN.19200101-20051231' in name]
    trefhtmin_xaer_former_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                                if '.TREFHTMN.19200101-20051231' in name]
    trefhtmin_xghg_former_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                                if '.TREFHTMN.19200101-20051231' in name]
    trefhtmin_all_latter_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                               if '.TREFHTMN.20060101-20801231' in name]
    trefhtmin_xaer_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                                if '.TREFHTMN.20060101-20801231' in name]
    trefhtmin_xghg_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                                if '.TREFHTMN.20060101-20801231' in name]

    return trefhtmin_all_former_em, trefhtmin_xaer_former_em, trefhtmin_xghg_former_em, \
           trefhtmin_all_latter_em, trefhtmin_xaer_latter_em, trefhtmin_xghg_latter_em


def get_paths_trefhtmax_odata() -> tuple:
    """
    Original datasets as downloaded for TREFHTMX

    :return: Three lists for ALL, XAER, and XGHG in that order, each containing paths to the netCDF files.
    """
    dataset_names = listdir(DOWNLOADED_ODATA_DIR)
    xaer_datasets = [name for name in dataset_names if 'xaer' in name]
    xghg_datasets = [name for name in dataset_names if 'xghg' in name]
    all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name or 'B20TRC5CNBDRD' in name]

    trefhtmax_all_former_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                               if '.TREFHTMX.19200101-20051231' in name]
    trefhtmax_xaer_former_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                                if '.TREFHTMX.19200101-20051231' in name]
    trefhtmax_xghg_former_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                                if '.TREFHTMX.19200101-20051231' in name]
    trefhtmax_all_latter_em = [DOWNLOADED_ODATA_DIR + name for name in all_datasets
                               if '.TREFHTMX.20060101-20801231' in name]
    trefhtmax_xaer_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xaer_datasets
                                if '.TREFHTMX.20060101-20801231' in name]
    trefhtmax_xghg_latter_em = [DOWNLOADED_ODATA_DIR + name for name in xghg_datasets
                                if '.TREFHTMX.20060101-20801231' in name]

    return trefhtmax_all_former_em, trefhtmax_xaer_former_em, trefhtmax_xghg_former_em, \
           trefhtmax_all_latter_em, trefhtmax_xaer_latter_em, trefhtmax_xghg_latter_em


def get_paths_heat_output(tmp_type: str = "tn", exp_num: str = "3114") -> tuple:
    dataset_names = listdir(HEAT_OUTPUT_1920_1950_BASE)
    all_datasets = [HEAT_OUTPUT_1920_1950_BASE + name for name in dataset_names if 'ALL' in name and
                    tmp_type in name and exp_num in name]
    xghg_datasets = [HEAT_OUTPUT_1920_1950_BASE + name for name in dataset_names if 'XGHG' in name and
                     tmp_type in name and exp_num in name]
    xaer_datasets = [HEAT_OUTPUT_1920_1950_BASE + name for name in dataset_names if 'XAER' in name and
                     tmp_type in name and exp_num in name]

    return all_datasets, xghg_datasets, xaer_datasets


def get_paths_cdata() -> tuple:
    """
    Original datasets as downloaded for TREFHT, TREFHTMX, TREFHTMN

    :return: Nine lists for ALL, XAER, XGHG for all three variables, in these orders.
    """
    datasets = listdir(CONCATENATED_ODATA)
    trefht_all_em = [CONCATENATED_ODATA + name for name in datasets if "trefht_all" in name]
    trefht_xaer_em = [CONCATENATED_ODATA + name for name in datasets if "trefht_xaer" in name]
    trefht_xghg_em = [CONCATENATED_ODATA + name for name in datasets if "trefht_xghg" in name]
    trefhtmax_all_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmax_all" in name]
    trefhtmax_xaer_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmax_xaer" in name]
    trefhtmax_xghg_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmax_xghg" in name]
    trefhtmin_all_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmin_all" in name]
    trefhtmin_xaer_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmin_xaer" in name]
    trefhtmin_xghg_em = [CONCATENATED_ODATA + name for name in datasets if "trefhtmin_xghg" in name]

    return trefht_all_em, trefht_xaer_em, trefht_xghg_em, \
           trefhtmax_all_em, trefhtmax_xaer_em, trefhtmax_xghg_em, \
           trefhtmin_all_em, trefhtmin_xaer_em, trefhtmin_xghg_em


def get_1980_2000_ALL_datasets() -> tuple:
    datasets = listdir(HEAT_OUTPUT_1980_2000_BASE)
    max_all_datasets = [HEAT_OUTPUT_1980_2000_BASE + name for name in datasets if "tx" in name]
    min_all_datasets = [HEAT_OUTPUT_1980_2000_BASE + name for name in datasets if "tn" in name]

    return max_all_datasets, min_all_datasets


def get_paths_heat_output_avg(tmp_type: str = "min", exp_num: str = "3114") -> tuple:
    """
    Heat outputs for ensemble members averaged by ALL, XGHG, and XAER
    :param tmp_type: "min" or "max"
    :param exp_num: Experiment number
    :return:
    """
    dataset_names = listdir(OUT_EM_AVGS_1920_1950)
    all_datasets = [OUT_EM_AVGS_1920_1950 + name for name in dataset_names if 'ALL' in name and
                    tmp_type in name and exp_num in name]
    xghg_datasets = [OUT_EM_AVGS_1920_1950 + name for name in dataset_names if 'XGHG' in name and
                     tmp_type in name and exp_num in name]
    xaer_datasets = [OUT_EM_AVGS_1920_1950 + name for name in dataset_names if 'XAER' in name and
                     tmp_type in name and exp_num in name]

    return all_datasets, xghg_datasets, xaer_datasets