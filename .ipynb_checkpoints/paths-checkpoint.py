from os import listdir

# Path to head of data directory
DIR_PATH = "/projects/dgs/persad_research/heat_research/data/"

## ===================================== TREFHT/MN/MX DATA =====================================


def trefht_members() -> tuple:
    """TREFHT ensemble members for ALL, XGHG, and XAER, concatenated time series"""
    all_ds = [DIR_PATH + "TREFHT/ALL/concatenated/" + name for name in listdir(DIR_PATH + "TREFHT/ALL/concatenated/")]
    xghg_ds = [DIR_PATH + "TREFHT/XGHG/concatenated/" + name for name in listdir(DIR_PATH + "TREFHT/XGHG/concatenated/")]
    xaer_ds = [DIR_PATH + "TREFHT/XAER/concatenated/" + name for name in listdir(DIR_PATH + "TREFHT/XAER/concatenated/")]
    return all_ds, xghg_ds, xaer_ds


def trefht_members_raw() -> tuple:
    """TREFHT ensemble members for ALL, XGHG, and XAER, unformatted/non-concatenated time series"""
    all_ds = [DIR_PATH + "TREFHT/ALL/download/" + name for name in listdir(DIR_PATH + "TREFHT/ALL/download/")]
    xghg_ds = [DIR_PATH + "TREFHT/XGHG/download/" + name for name in listdir(DIR_PATH + "TREFHT/XGHG/download/")]
    xaer_ds = [DIR_PATH + "TREFHT/XAER/download/" + name for name in listdir(DIR_PATH + "TREFHT/XAER/download/")]
    return all_ds, xghg_ds, xaer_ds


def trefhtmn_members() -> tuple:
    """TREFHTMN ensemble members for ALL, XGHG, and XAER, concatenated time series"""
    all_ds = [DIR_PATH + "TREFHTMN/ALL/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMN/ALL/concatenated/")]
    xghg_ds = [DIR_PATH + "TREFHTMN/XGHG/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMN/XGHG/concatenated/")]
    xaer_ds = [DIR_PATH + "TREFHTMN/XAER/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMN/XAER/concatenated/")]
    return all_ds, xghg_ds, xaer_ds


def trefhtmn_members_raw() -> tuple:
    """TREFHTMN ensemble members for ALL, XGHG, and XAER, unformatted/non-concatenated time series"""
    all_ds = [DIR_PATH + "TREFHTMN/ALL/download/" + name for name in listdir(DIR_PATH + "TREFHTMN/ALL/download/")]
    xghg_ds = [DIR_PATH + "TREFHTMN/XGHG/download/" + name for name in listdir(DIR_PATH + "TREFHTMN/XGHG/download/")]
    xaer_ds = [DIR_PATH + "TREFHTMN/XAER/download/" + name for name in listdir(DIR_PATH + "TREFHTMN/XAER/download/")]
    return all_ds, xghg_ds, xaer_ds


def trefhtmx_members() -> tuple:
    """TREFHTMX ensemble members for ALL, XGHG, and XAER, concatenated time series"""
    all_ds = [DIR_PATH + "TREFHTMX/ALL/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMX/ALL/concatenated/")]
    xghg_ds = [DIR_PATH + "TREFHTMX/XGHG/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMX/XGHG/concatenated/")]
    xaer_ds = [DIR_PATH + "TREFHTMX/XAER/concatenated/" + name for name in listdir(DIR_PATH + "TREFHTMX/XAER/concatenated/")]
    return all_ds, xghg_ds, xaer_ds


def trefhtmx_members_raw() -> tuple:
    """TREFHTMX ensemble members for ALL, XGHG, and XAER, unformatted/non-concatenated time series"""
    all_ds = [DIR_PATH + "TREFHTMX/ALL/download/" + name for name in listdir(DIR_PATH + "TREFHTMX/ALL/download/")]
    xghg_ds = [DIR_PATH + "TREFHTMX/XGHG/download/" + name for name in listdir(DIR_PATH + "TREFHTMX/XGHG/download/")]
    xaer_ds = [DIR_PATH + "TREFHTMX/XAER/download/" + name for name in listdir(DIR_PATH + "TREFHTMX/XAER/download/")]
    return all_ds, xghg_ds, xaer_ds


def thresholds_members_1920_1950() -> tuple:
    """ALL, XGHG, XAER thresholds for each ensemble member calculated using 1920 to 1950 baseline"""
    all_ds = [DIR_PATH + "thresholds/ALL/1920_1950/" + name for name in listdir(DIR_PATH + "thresholds/ALL/1920_1950/")]
    xghg_ds = [DIR_PATH + "thresholds/XGHG/1920_1950/" + name for name in listdir(DIR_PATH + "thresholds/XGHG/1920_1950/")]
    xaer_ds = [DIR_PATH + "thresholds/XAER/1920_1950/" + name for name in listdir(DIR_PATH + "thresholds/XAER//1920_1950")]
    return all_ds, xghg_ds, xaer_ds


## ===================================== MERRA2 DATA =====================================


def merra2_threshold() -> str:
    """Path to MERRA2 threshold netCDF"""
    return DIR_PATH + "MERRA2/thresholds/threshold_1980-2010_90pcntl.nc"


def merra2_download() -> str:
    """Path to downloaded MERRA2 dataset"""
    return DIR_PATH + "MERRA2/downloads/MERRA2_1980-2015.nc"


def heat_out_merra2() -> list:
    """Heat outputs for MERRA2 data"""
    return [DIR_PATH + "MERRA2/heat_outputs/" + name for name in listdir(DIR_PATH + "MERRA2/heat_outputs/")]

## ===================================== PREINDUSTRIAL CONTROL DATA =====================================


def control_threshold() -> str:
    """Path to pre-industrial contro threshold netCDF"""
    return ""


def control_trefht_download() -> list:
    """Path to downloaded pre-industrial control datasets"""
    return [DIR_PATH + "/TREFHT/CONTROL/downloads/" + name for name in listdir(DIR_PATH + "/TREFHT/CONTROL/downloads/")]

## ===================================== HEAT OUTPUTS =====================================


def heat_out_trefht_tmin_members_1920_1950() -> tuple:
    """Minimum temperature heat outputs for each ensemble member with 1920 to 1950 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/tmin") if "1980-2000" not in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/tmin") if "1980-2000" not in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/tmin") if "1980-2000" not in name]
    return all_ds, xghg_ds, xaer_ds


def heat_out_trefht_tmax_members_1920_1950() -> tuple:
    """Maximum temperature heat outputs for each ensemble member with 1920 to 1950 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/tmax") if "1980-2000" not in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/tmax") if "1980-2000" not in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/tmax") if "1980-2000" not in name]
    return all_ds, xghg_ds, xaer_ds


def heat_out_trefht_tmin_members_1980_2000() -> tuple:
    """Minimum temperature heat outputs for each ensemble member with 1980 to 2000 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/tmin") if "1980-2000" in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/tmin") if "1980-2000" in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/tmin/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/tmin") if "1980-2000" in name]
    return all_ds, xghg_ds, xaer_ds


def heat_out_trefht_tmax_members_1980_2000() -> tuple:
    """Maximum temperature heat outputs for each ensemble member with 1980 to 2000 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/tmax") if "1980-2000" in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/tmax") if "1980-2000" in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/tmax/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/tmax") if "1980-2000" in name]
    return all_ds, xghg_ds, xaer_ds


def heat_out_trefht_tmin_averages_1920_1950() -> tuple:
    """Minimum temperature heat outputs averaged from each ensemble member with 1920 to 1950 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/1920_1950_base_avg") if "min" in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/1920_1950_base_avg") if "min" in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/1920_1950_base_avg") if "min" in name]
    return all_ds, xghg_ds, xaer_ds


def heat_out_trefht_tmax_averages_1920_1950() -> tuple:
    """Maximum temperature heat outputs averaged from each ensemble member with 1920 to 1950 baseline"""
    all_ds = [DIR_PATH + "/heat_output/ALL/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/ALL/1920_1950_base_avg") if "max" in name]
    xghg_ds = [DIR_PATH + "/heat_output/XGHG/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/XGHG/1920_1950_base_avg") if "max" in name]
    xaer_ds = [DIR_PATH + "/heat_output/XAER/1920_1950_base_avg/" + name for name in listdir(DIR_PATH + "/heat_output/XAER/1920_1950_base_avg") if "max" in name]
    return all_ds, xghg_ds, xaer_ds

