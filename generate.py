import analysis
import paths
import pop_weighted as pop
import xarray
import cProfile, pstats, io
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib import colors
from matplotlib import rc
from pstats import SortKey
from os.path import isfile
from multiprocessing import Process
from analysis import bilinear_interpolation
from regionmask.defined_regions import ar6
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from paths import DIR_PATH, heat_out_trefht_tmax_members_1920_1950_CONTROL as tmax_paths, heat_out_trefht_tmin_members_1920_1950_CONTROL as tmin_paths

def gen_weighted_v_unweighted_perc(variable: str, exp_num: str):
    pop_size = 7.96949658e+09

    max_weighted_all = (xarray.open_dataset(f"../data/populations/weighted/ALL/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)
    max_weighted_aer = max_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XAER/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)
    max_weighted_ghg = max_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XGHG/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)


    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]

    land_mask = ar6.land.mask(tx_all_dataset.mean(dim="member"))

    max_unweighted_all = tx_all_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days
    max_unweighted_aer = max_unweighted_all - tx_xaer_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days
    max_unweighted_ghg = max_unweighted_all - tx_xghg_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days


    min_weighted_all = (xarray.open_dataset(f"../data/populations/weighted/ALL/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)
    min_weighted_aer = min_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XAER/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)
    min_weighted_ghg = min_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XGHG/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon").mean(dim="member", skipna=True)

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]

    land_mask = ar6.land.mask(tn_all_dataset.mean(dim="member"))

    min_unweighted_all = tn_all_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days
    min_unweighted_aer = min_unweighted_all - tn_xaer_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days
    min_unweighted_ghg = min_unweighted_all - tn_xghg_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).mean(dim="member", skipna=True).dt.days

    min_all_perc = (((min_weighted_all - min_unweighted_all) / min_unweighted_all) * min_unweighted_all.where(min_unweighted_all == 0, other=1)).fillna(0)
    min_aer_perc = (((min_weighted_aer - min_unweighted_aer) / min_unweighted_aer) * min_unweighted_aer.where(min_unweighted_aer == 0, other=1)).fillna(0)
    min_ghg_perc = (((min_weighted_ghg - min_unweighted_ghg) / min_unweighted_ghg) * min_unweighted_ghg.where(min_unweighted_ghg == 0, other=1)).fillna(0)

    max_all_perc = (((max_weighted_all - max_unweighted_all) / max_unweighted_all) * max_unweighted_all.where(max_unweighted_all == 0, other=1)).fillna(0)
    max_aer_perc = (((max_weighted_aer - max_unweighted_aer) / max_unweighted_aer) * max_unweighted_aer.where(max_unweighted_aer == 0, other=1)).fillna(0)
    max_ghg_perc = (((max_weighted_ghg - max_unweighted_ghg) / max_unweighted_ghg) * max_unweighted_ghg.where(max_unweighted_ghg == 0, other=1)).fillna(0)


    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Simple Population-Weighted Isolated Signal Comparison(Land Masked)", fontsize=30)

    min_all_perc.sel(time=slice(1920,2079)).plot(ax=ax1, color="darkgreen", label="ALL Perc. Change", linewidth=4)
    min_aer_perc.sel(time=slice(1920,2079)).plot(ax=ax1, color="darkblue", label="AER Perc. Change", linewidth=4)
    min_ghg_perc.sel(time=slice(1920,2079)).plot(ax=ax1, color="darkred", label="GHG Perc. Change", linewidth=4)

    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")


    max_all_perc.sel(time=slice(1920,2079)).plot(ax=ax2, color="darkgreen", label="ALL Perc. Change", linewidth=4)
    max_aer_perc.sel(time=slice(1920,2079)).plot(ax=ax2, color="darkblue", label="AER Perc. Change", linewidth=4)
    max_ghg_perc.sel(time=slice(1920,2079)).plot(ax=ax2, color="darkred", label="GHG Perc. Change", linewidth=4)

    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")


    rc('font', **{'weight': 'bold', 'size': 24})
    f.tight_layout()


def gen_weighted_v_unweighted(variable: str, exp_num: str):
    pop_size = 7.96949658e+09

    max_weighted_all = (xarray.open_dataset(f"../data/populations/weighted/ALL/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon")
    max_weighted_aer = max_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XAER/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon")
    max_weighted_ghg = max_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XGHG/{variable}-{exp_num}-tx.nc").days / pop_size).sum(dim="lat").sum(dim="lon")


    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_tx90"]

    land_mask = ar6.land.mask(tx_all_dataset.mean(dim="member"))

    max_unweighted_all = tx_all_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days
    max_unweighted_aer = max_unweighted_all - tx_xaer_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days
    max_unweighted_ghg = max_unweighted_all - tx_xghg_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days


    min_weighted_all = (xarray.open_dataset(f"../data/populations/weighted/ALL/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon")
    min_weighted_aer = min_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XAER/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon")
    min_weighted_ghg = min_weighted_all - (xarray.open_dataset(f"../data/populations/weighted/XGHG/{variable}-{exp_num}-tn.nc").days / pop_size).sum(dim="lat").sum(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_tn90"]

    land_mask = ar6.land.mask(tn_all_dataset.mean(dim="member"))

    min_unweighted_all = tn_all_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days
    min_unweighted_aer = min_unweighted_all - tn_xaer_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days
    min_unweighted_ghg = min_unweighted_all - tn_xghg_dataset.where(land_mask > 0).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True).dt.days

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Simple Population-Weighted Isolated Signal Comparison(Land Masked)", fontsize=30)

    min_weighted_all.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="darkgreen", label="Weighted ALL", linewidth=4)
    min_weighted_aer.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="darkblue", label="Weighted AER", linewidth=4)
    min_weighted_ghg.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="darkred", label="Weighted GHG", linewidth=4)
    ax1.fill_between(min_weighted_all.sel(time=slice(1920,2079)).time, min_weighted_all.min(dim="member").sel(time=slice(1920,2079)), min_weighted_all.max(dim="member").sel(time=slice(1920,2079)), color="darkgreen", alpha=0.35)
    ax1.fill_between(min_weighted_aer.sel(time=slice(1920,2079)).time, min_weighted_aer.min(dim="member").sel(time=slice(1920,2079)), min_weighted_aer.max(dim="member").sel(time=slice(1920,2079)), color="darkblue", alpha=0.35)
    ax1.fill_between(min_weighted_ghg.sel(time=slice(1920,2079)).time, min_weighted_ghg.min(dim="member").sel(time=slice(1920,2079)), min_weighted_ghg.max(dim="member").sel(time=slice(1920,2079)), color="darkred", alpha=0.35)

    min_unweighted_all.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="lime", label="Unweighted ALL", linewidth=4)
    min_unweighted_aer.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="aqua", label="Unweighted AER", linewidth=4)
    min_unweighted_ghg.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax1, color="tomato", label="Unweighted GHG", linewidth=4)
    ax1.fill_between(min_unweighted_all.sel(time=slice(1920,2079)).time, min_unweighted_all.min(dim="member").sel(time=slice(1920,2079)), min_unweighted_all.max(dim="member").sel(time=slice(1920,2079)), color="lime", alpha=0.35)
    ax1.fill_between(min_unweighted_aer.sel(time=slice(1920,2079)).time, min_unweighted_aer.min(dim="member").sel(time=slice(1920,2079)), min_unweighted_aer.max(dim="member").sel(time=slice(1920,2079)), color="aqua", alpha=0.35)
    ax1.fill_between(min_unweighted_ghg.sel(time=slice(1920,2079)).time, min_unweighted_ghg.min(dim="member").sel(time=slice(1920,2079)), min_unweighted_ghg.max(dim="member").sel(time=slice(1920,2079)), color="tomato", alpha=0.35)

    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")

    max_weighted_all.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="darkgreen", label="Weighted ALL", linewidth=4)
    max_weighted_aer.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="darkblue", label="Weighted AER", linewidth=4)
    max_weighted_ghg.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="darkred", label="Weighted GHG", linewidth=4)
    ax2.fill_between(max_weighted_all.sel(time=slice(1920,2079)).time, max_weighted_all.min(dim="member").sel(time=slice(1920,2079)), max_weighted_all.max(dim="member").sel(time=slice(1920,2079)), color="green", alpha=0.35)
    ax2.fill_between(max_weighted_aer.sel(time=slice(1920,2079)).time, max_weighted_aer.min(dim="member").sel(time=slice(1920,2079)), max_weighted_aer.max(dim="member").sel(time=slice(1920,2079)), color="blue", alpha=0.35)
    ax2.fill_between(max_weighted_ghg.sel(time=slice(1920,2079)).time, max_weighted_ghg.min(dim="member").sel(time=slice(1920,2079)), max_weighted_ghg.max(dim="member").sel(time=slice(1920,2079)), color="red", alpha=0.35)

    max_unweighted_all.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="lime", label="Unweighted ALL", linewidth=4)
    max_unweighted_aer.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="aqua", label="Unweighted AER", linewidth=4)
    max_unweighted_ghg.mean(dim="member", skipna=True).sel(time=slice(1920,2079)).plot(ax=ax2, color="tomato", label="Unweighted GHG", linewidth=4)
    ax2.fill_between(max_unweighted_all.sel(time=slice(1920,2079)).time, max_unweighted_all.min(dim="member").sel(time=slice(1920,2079)), max_unweighted_all.max(dim="member").sel(time=slice(1920,2079)), color="lime", alpha=0.35)
    ax2.fill_between(max_unweighted_aer.sel(time=slice(1920,2079)).time, max_unweighted_aer.min(dim="member").sel(time=slice(1920,2079)), max_unweighted_aer.max(dim="member").sel(time=slice(1920,2079)), color="aqua", alpha=0.35)
    ax2.fill_between(max_unweighted_ghg.sel(time=slice(1920,2079)).time, max_unweighted_ghg.min(dim="member").sel(time=slice(1920,2079)), max_unweighted_ghg.max(dim="member").sel(time=slice(1920,2079)), color="tomato", alpha=0.35)


    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    
    rc('font', **{'weight': 'bold', 'size': 24})
    f.tight_layout()
    return f


def gen_isolated_signal_ratio(exp_num: str, variable_num: str, variable_den: str):
    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tx_all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_num}_tx90"].dt.days
    tx_xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_num}_tx90"].dt.days
    tx_xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_num}_tx90"].dt.days
    land_mask = ar6.land.mask(tx_all_data)

    tx_all_divided = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tx90"].dt.days
    tx_xaer_divided = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tx90"].dt.days
    tx_xghg_divided = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tx90"].dt.days
    tx_all_den = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tx90"].dt.days
    tx_xaer_den = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tx90"].dt.days
    tx_xghg_den = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tx90"].dt.days
    tx_all_divided /= tx_all_den.where(tx_all_den != 0)
    tx_xaer_divided /= tx_xaer_den.where(tx_xaer_den != 0)
    tx_xghg_divided /= tx_xghg_den.where(tx_xghg_den != 0)

    tx_all_data_den = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tx90"].dt.days
    tx_xaer_data_den = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tx90"].dt.days
    tx_xghg_data_den = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tx90"].dt.days

    tx_all_data /= tx_all_data_den.where(tx_all_data_den != 0)
    tx_xaer_data /= tx_xaer_data_den.where(tx_xaer_data_den != 0)
    tx_xghg_data /= tx_xghg_data_den.where(tx_xghg_data_den != 0)

    tx_all_data = tx_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xaer_data = tx_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xghg_data = tx_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tn_all_data = tn_all_dataset.mean(dim="member")[f"{variable_num}_tn90"].dt.days
    tn_xaer_data = tn_xaer_dataset.mean(dim="member")[f"{variable_num}_tn90"].dt.days
    tn_xghg_data = tn_xghg_dataset.mean(dim="member")[f"{variable_num}_tn90"].dt.days

    tn_all_divided = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tn90"].dt.days
    tn_xaer_divided = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tn90"].dt.days
    tn_xghg_divided = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable_num}_tn90"].dt.days
    tn_all_den = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tn90"].dt.days
    tn_xaer_den = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tn90"].dt.days
    tn_xghg_den = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable_den}_tn90"].dt.days
    tn_all_divided /= tn_all_den.where(tn_all_den != 0)
    tn_xaer_divided /= tn_xaer_den.where(tn_xaer_den != 0)
    tn_xghg_divided /= tn_xghg_den.where(tn_xghg_den != 0)

    tn_all_data_den = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tn90"].dt.days
    tn_xaer_data_den = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tn90"].dt.days
    tn_xghg_data_den = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable_den}_tn90"].dt.days

    tn_all_data /= tn_all_data_den.where(tx_all_data_den != 0)
    tn_xaer_data /= tn_xaer_data_den.where(tx_xaer_data_den != 0)
    tn_xghg_data /= tn_xghg_data_den.where(tx_xghg_data_den != 0)

    tn_all_data = tn_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xaer_data = tn_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xghg_data = tn_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable_num}/{variable_den} {exp_num} Isolated Signal Comparison (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    tn_xghg_data.plot(ax=ax1, label="XGHG", color="red", linewidth=4)
    ax1.fill_between(tn_xghg_data.time, tn_xghg_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tn_xghg_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                    color="red", alpha=0.35)
    tn_xaer_data.plot(ax=ax1, label="XAER", color="blue", linewidth=4)
    ax1.fill_between(tn_xaer_data.time, tn_xaer_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tn_xaer_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                    color="blue", alpha=0.35)
    tn_all_data.plot(ax=ax1, label="ALL", color="green", linewidth=4)
    ax1.fill_between(tn_all_data.time, tn_all_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tn_all_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                    color="green", alpha=0.35)
    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")

    tx_xghg_data.plot(ax=ax2, label="XGHG", color="red", linewidth=4)
    ax2.fill_between(tx_xghg_data.time, tx_xghg_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tx_xghg_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                    color="red", alpha=0.35)

    tx_xaer_data.plot(ax=ax2, label="XAER", color="blue", linewidth=4)
    ax2.fill_between(tx_xaer_data.time, tx_xaer_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tx_xaer_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                     color="blue", alpha=0.35)
    tx_all_data.plot(ax=ax2, label="ALL", color="green", linewidth=4)
    ax2.fill_between(tx_all_data.time, tx_all_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member"), 
                     tx_all_divided.where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member"),
                    color="green", alpha=0.35)
    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    f.tight_layout()
    return f


def gen_isolated_signal_ghg_aer_all(variable: str, exp_num: str):
    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tx_all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    land_mask = ar6.land.mask(tx_all_data)

    tx_all_data = tx_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xaer_data = tx_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xghg_data = tx_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tn_all_data = tn_all_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xaer_data = tn_xaer_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xghg_data = tn_xghg_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days

    tn_all_data = tn_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xaer_data = tn_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xghg_data = tn_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Isolated Signal Comparison (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})


    (tn_all_data - tn_xghg_data).plot(ax=ax1, label="GHG", color="red", linewidth=4)
    ax1.fill_between(tn_xghg_data.time, (tn_all_dataset - tn_xghg_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tn_all_dataset - tn_xghg_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)
    (tn_all_data - tn_xaer_data).plot(ax=ax1, label="AER", color="blue", linewidth=4)
    ax1.fill_between(tn_xaer_data.time, (tn_all_dataset - tn_xaer_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tn_all_dataset - tn_xaer_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="blue", alpha=0.35)
    tn_all_data.plot(ax=ax1, label="ALL", color="green", linewidth=4)
    ax1.fill_between(tn_all_data.time, tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="green", alpha=0.35)
    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")

    (tx_all_data - tx_xghg_data).plot(ax=ax2, label="GHG", color="red", linewidth=4)
    ax2.fill_between(tx_xghg_data.time, (tx_all_dataset - tx_xghg_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tx_all_dataset - tx_xghg_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)

    (tx_all_data - tx_xaer_data).plot(ax=ax2, label="AER", color="blue", linewidth=4)
    ax2.fill_between(tx_xaer_data.time, (tx_all_dataset - tx_xaer_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tx_all_dataset - tx_xaer_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                     color="blue", alpha=0.35)
    tx_all_data.plot(ax=ax2, label="ALL", color="green", linewidth=4)
    ax2.fill_between(tx_all_data.time, tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="green", alpha=0.35)

    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    f.tight_layout()
    return f


def gen_isolated_signal_ghg(variable: str, exp_num: str):
    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tx_all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    land_mask = ar6.land.mask(tx_all_data)

    tx_all_data = tx_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xghg_data = tx_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tn_all_data = tn_all_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xghg_data = tn_xghg_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days

    tn_all_data = tn_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xghg_data = tn_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Isolated Signal Comparison (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    (tn_all_data).plot(ax=ax1, label="ALL", color="black", linewidth=4)
    ax1.fill_between(tn_xghg_data.time, tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="black", alpha=0.35)
    
    (tn_all_data - tn_xghg_data).plot(ax=ax1, label="GHG", color="red", linewidth=4)
    ax1.fill_between(tn_xghg_data.time, (tn_all_dataset - tn_xghg_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tn_all_dataset - tn_xghg_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)
    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")
    
    (tx_all_data).plot(ax=ax2, label="ALL", color="black", linewidth=4)
    ax2.fill_between(tx_xghg_data.time, tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                    tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="black", alpha=0.35)
    (tx_all_data - tx_xghg_data).plot(ax=ax2, label="GHG", color="red", linewidth=4)
    ax2.fill_between(tx_xghg_data.time, (tx_all_dataset - tx_xghg_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tx_all_dataset - tx_xghg_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)
    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    f.tight_layout()
    return f
    
def gen_isolated_signal_aer(variable: str, exp_num: str):
    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    
    tx_all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    land_mask = ar6.land.mask(tx_all_data)

    tx_all_data = tx_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xaer_data = tx_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    
    tn_all_data = tn_all_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xaer_data = tn_xaer_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days

    tn_all_data = tn_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xaer_data = tn_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Isolated Signal Comparison (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    (tn_all_data - tn_xaer_data).plot(ax=ax1, label="AER", color="blue", linewidth=4)
    ax1.fill_between(tn_xaer_data.time, (tn_all_dataset - tn_xaer_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tn_all_dataset - tn_xaer_dataset)[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="blue", alpha=0.35)
    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")

    (tx_all_data - tx_xaer_data).plot(ax=ax2, label="AER", color="blue", linewidth=4)
    ax2.fill_between(tx_xaer_data.time, (tx_all_dataset - tx_xaer_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     (tx_all_dataset - tx_xaer_dataset)[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="blue", alpha=0.35)
    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    f.tight_layout()
    return f


def gen_isolated_signal_comparison(variable: str, exp_num: str):
    all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]

    tx_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tx_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tx_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tx_all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    tx_xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested").mean(dim="member")[f"{variable}_tx90"].dt.days
    land_mask = ar6.land.mask(tx_all_data)

    tx_all_data = tx_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xaer_data = tx_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tx_xghg_data = tx_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
    xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
    xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]

    tn_all_dataset = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")
    tn_xaer_dataset = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")
    tn_xghg_dataset = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")

    tn_all_data = tn_all_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xaer_data = tn_xaer_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days
    tn_xghg_data = tn_xghg_dataset.mean(dim="member")[f"{variable}_tn90"].dt.days

    tn_all_data = tn_all_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xaer_data = tn_xaer_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    tn_xghg_data = tn_xghg_data.where(land_mask > 0).mean(dim="lat").mean(dim="lon")

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), facecolor='w')
    f.suptitle(f"{variable} {exp_num} Isolated Signal Comparison (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    tn_all_data.plot(ax=ax1, label="ALL", color="green", linewidth=4)
    ax1.fill_between(tn_all_data.time, tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tn_all_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="green", alpha=0.35)
    tn_xaer_data.plot(ax=ax1, label="XAER", color="blue", linewidth=4)
    ax1.fill_between(tn_xaer_data.time, tn_xaer_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tn_xaer_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="blue", alpha=0.35)
    tn_xghg_data.plot(ax=ax1, label="XGHG", color="red", linewidth=4)
    ax1.fill_between(tn_xghg_data.time, tn_xghg_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tn_xghg_dataset[f"{variable}_tn90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)

    ax1.grid()
    ax1.legend()
    ax1.set_title("Using Min. Temperature")

    tx_all_data.plot(ax=ax2, label="ALL", color="green", linewidth=4)
    ax2.fill_between(tx_all_data.time, tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tx_all_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="green", alpha=0.35)
    tx_xaer_data.plot(ax=ax2, label="XAER", color="blue", linewidth=4)
    ax2.fill_between(tx_xaer_data.time, tx_xaer_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tx_xaer_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="blue", alpha=0.35)
    tx_xghg_data.plot(ax=ax2, label="XGHG", color="red", linewidth=4)
    ax2.fill_between(tx_xghg_data.time, tx_xghg_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").min(dim="member").dt.days, 
                     tx_xghg_dataset[f"{variable}_tx90"].where(land_mask > 0).mean(dim="lat").mean(dim="lon").max(dim="member").dt.days,
                    color="red", alpha=0.35)

    ax2.grid()
    ax2.legend()
    ax2.set_title("Using Max. Temperature")

    f.tight_layout()
    return f

def gen_pop_isolated_signal_comparison(variable: str, exp_num: str, min_max: str):
    land_mask = ar6.land.mask(xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc"))

    weighted_all = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_xaer = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/XAER/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_xghg = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/XGHG/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_aer = ((weighted_all - weighted_xaer) / weighted_xaer)*100
    weighted_ghg = ((weighted_all - weighted_xghg) / weighted_xghg)*100

    if min_max == "tx":
        all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]
    elif min_max == "tn":
        all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]
    all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")

    # Shift data
    unweighted_all = all_data.assign_coords(lon=(((all_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xaer = xaer_data.assign_coords(lon=(((xaer_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xghg = xghg_data.assign_coords(lon=(((xghg_data.lon + 180) % 360) - 180)).sortby('lon')
    
    unweighted_all = unweighted_all.where(land_mask > 0).days.mean(dim="lat").mean(dim="lon")
    unweighted_xaer = unweighted_xaer.where(land_mask > 0).days.mean(dim="lat").mean(dim="lon")
    unweighted_xghg = unweighted_xghg.where(land_mask > 0).days.mean(dim="lat").mean(dim="lon")
    unweighted_aer = ((unweighted_all - unweighted_xaer) / unweighted_xaer)*100
    unweighted_ghg = ((unweighted_all - unweighted_xghg) / unweighted_xghg)*100

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 20), facecolor='w')
    f.suptitle(f"{variable} {exp_num} {min_max} Population Unweighted vs Weighted (Signal Divided by XSignal) (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    unweighted_aer.rename("Perc. Change (%)").plot(ax=ax1, color="blue", label="Unweighted AER")
    weighted_aer.rename("Perc. Change (%)").plot(ax=ax1, color="red", label="Weighted AER")
    ax1.grid()
    ax1.legend()
    ax1.set_title("AER Signal")

    unweighted_ghg.rename("Perc. Change (%)").plot(ax=ax2, color="blue", label="Unweighted GHG")
    weighted_ghg.rename("Perc. Change (%)").plot(ax=ax2, color="red", label="Weighted GHG")
    ax2.grid()
    ax2.legend()
    ax2.set_title("GHG Signal")

    f.tight_layout()
    return f


def gen_pop_region_bar_charts(variable: str, exp_num: str, min_max: str, begin_time: int, end_time: int):
    land_mask = ar6.land.mask(xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc"))

    weighted_all = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_xaer = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/XAER/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_xghg = xarray.open_dataset(f"{paths.DIR_PATH}/populations/weighted/XGHG/{variable}-{exp_num}-{min_max}.nc").days.where(land_mask > 0).mean(dim="lat").mean(dim="lon")
    weighted_aer = weighted_all - weighted_xaer
    weighted_aer = weighted_aer / weighted_xaer

    region_mask_aer = ar6.land.mask(weighted_aer)
    region_aer_values = weighted_aer.groupby(region_mask_aer).mean()

    for sel_region in region_aer_values.region:
        value = region_aer_values.sel(region=sel_region).days
        region_mask_aer = region_mask_aer.where(region_mask_aer != sel_region, other=value)
    region_mask_aer *= 100


    weighted_ghg = weighted_all - weighted_xghg
    weighted_ghg = weighted_ghg / weighted_xghg

    region_mask_ghg = ar6.land.mask(weighted_ghg)
    region_ghg_values = weighted_ghg.groupby(region_mask_ghg).mean()

    for sel_region in region_ghg_values.region:
        value = region_ghg_values.sel(region=sel_region).days
        region_mask_ghg = region_mask_ghg.where(region_mask_ghg != sel_region, other=value)
    region_mask_ghg *= 100


    if min_max == "tx":
        all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]
    elif min_max == "tn":
        all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]
    all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")

    # Shift data
    unweighted_all = all_data.assign_coords(lon=(((all_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xaer = xaer_data.assign_coords(lon=(((xaer_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xghg = xghg_data.assign_coords(lon=(((xghg_data.lon + 180) % 360) - 180)).sortby('lon')

    unweighted_all = unweighted_all.where(land_mask > 0).sel(time=slice(begin_time,end_time)).mean(dim="time")
    unweighted_xaer = unweighted_xaer.where(land_mask > 0).sel(time=slice(begin_time,end_time)).mean(dim="time")
    unweighted_xghg = unweighted_xghg.where(land_mask > 0).sel(time=slice(begin_time,end_time)).mean(dim="time")

    unweighted_aer = unweighted_all - unweighted_xaer
    unweighted_aer = unweighted_aer / unweighted_xaer

    uregion_mask_aer = ar6.land.mask(unweighted_aer)
    uregion_aer_values = unweighted_aer.groupby(uregion_mask_aer).mean()

    for sel_region in uregion_aer_values.region:
        value = uregion_aer_values.sel(region=sel_region).days
        uregion_mask_aer = uregion_mask_aer.where(uregion_mask_aer != sel_region, other=value)
    uregion_mask_aer *= 100


    unweighted_ghg = unweighted_all - unweighted_xghg
    unweighted_ghg = unweighted_ghg / unweighted_xghg

    uregion_mask_ghg = ar6.land.mask(unweighted_ghg)
    uregion_ghg_values = unweighted_ghg.groupby(uregion_mask_ghg).mean()

    for sel_region in uregion_ghg_values.region:
        value = uregion_ghg_values.sel(region=sel_region).days
        uregion_mask_ghg = uregion_mask_ghg.where(uregion_mask_ghg != sel_region, other=value)
    uregion_mask_ghg *= 100

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(60, 90), facecolor='w')
    f.suptitle(f"HWF Population Unweighted vs Weighted Averaged Over {begin_time} to {end_time} (Land Masked)", fontsize=75)
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 50})

    labels = [ar6.land.regions[index].name for index in ar6.land.regions]

    ax1.barh(labels, uregion_aer_values.days.values, 0.9, label='Unweighted')
    ax1.barh(labels, region_aer_values.days.values, 0.6, label='Weighted')
    ax1.legend()
    ax1.set_ylabel('Regions')
    ax1.set_xlabel('Perc Change(%)')
    ax1.grid()
    ax1.set_title("AER Signal")

    ax2.barh(labels, uregion_ghg_values.days.values, 0.9, label='Unweighted')
    ax2.barh(labels, region_ghg_values.days.values, 0.6, label='Weighted')
    ax2.legend()
    ax2.set_ylabel('Regions')
    ax2.set_xlabel('Perc Change(%)')
    ax2.grid()
    ax2.set_title("GHG Signal")
    
    return f


def gen_pop_signal_ratios(variable: str, exp_num: str, min_max: str, time_begin: int, time_end: int):
    pop_size = 7.96949658e+09

    land_mask = ar6.land.mask(xarray.open_dataset(f"{DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc"))

    weighted_all = (xarray.open_dataset(f"{DIR_PATH}/populations/weighted/ALL/{variable}-{exp_num}-{min_max}.nc").where(land_mask > 0).days / pop_size).sum(dim="lat").sum(dim="lon").sel(time=slice(time_begin, time_end))
    weighted_xaer = (xarray.open_dataset(f"{DIR_PATH}/populations/weighted/XAER/{variable}-{exp_num}-{min_max}.nc").where(land_mask > 0).days / pop_size).sum(dim="lat").sum(dim="lon").sel(time=slice(time_begin, time_end))
    weighted_xghg = (xarray.open_dataset(f"{DIR_PATH}/populations/weighted/XGHG/{variable}-{exp_num}-{min_max}.nc").where(land_mask > 0).days / pop_size).sum(dim="lat").sum(dim="lon").sel(time=slice(time_begin, time_end))
    weighted_aer = (weighted_all - weighted_xaer) / weighted_xaer
    weighted_ghg = (weighted_all - weighted_xghg) / weighted_xghg

    if min_max == "tx":
        all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]
    elif min_max == "tn":
        all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]
    all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")

    # Shift data
    unweighted_all = all_data.assign_coords(lon=(((all_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xaer = xaer_data.assign_coords(lon=(((xaer_data.lon + 180) % 360) - 180)).sortby('lon')
    unweighted_xghg = xghg_data.assign_coords(lon=(((xghg_data.lon + 180) % 360) - 180)).sortby('lon')

    unweighted_all = unweighted_all.where(land_mask > 0).mean(dim="lat").mean(dim="lon").sel(time=slice(time_begin, time_end))
    unweighted_xaer = unweighted_xaer.where(land_mask > 0).mean(dim="lat").mean(dim="lon").sel(time=slice(time_begin, time_end))
    unweighted_xghg = unweighted_xghg.where(land_mask > 0).mean(dim="lat").mean(dim="lon").sel(time=slice(time_begin, time_end))
    unweighted_aer = (unweighted_all - unweighted_xaer) / unweighted_xaer
    unweighted_ghg = (unweighted_all - unweighted_xghg) / unweighted_xghg

    f, ax1 = plt.subplots(1, 1, figsize=(28, 10), facecolor='w')
    f.suptitle(f"HWF Population Unweighted over Weighted (Signal Divided by XSignal) (Land Masked)", fontsize=30)
    rc('font', **{'weight': 'bold', 'size': 22})

    ((unweighted_aer / weighted_aer.mean(dim="member"))*100).rename("Ratio (%)").plot(ax=ax1, color="blue", label="AER Pop. Unweighted/Weighted")
    ((unweighted_ghg / weighted_ghg.mean(dim="member"))*100).rename("Ratio (%)").plot(ax=ax1, color="green", label="GHG Pop. Unweighted/Weighted")
    ax1.grid()
    ax1.legend()
    ax1.set_title(f"Ratios for AER vs GHG (Note: 1920-{time_begin} omitted due to large values skewing data) Land Masked")

    f.tight_layout()
    return f


def gen_autocorrelation_maps():
    merra2_max = xarray.open_dataset(paths.merra2_download())["T2MMAX"]
    merra2_min = xarray.open_dataset(paths.merra2_download())["T2MMIN"]
    max_coeffs = merra2_max.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    min_coeffs = merra2_min.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    merra2_max_detrended = merra2_max - xarray.polyval(merra2_max.time, max_coeffs)
    merra2_min_detrended = merra2_min - xarray.polyval(merra2_min.time, min_coeffs)
    merra2_max_anomalies = merra2_max_detrended.groupby("time.dayofyear") - merra2_max_detrended.groupby("time.dayofyear").mean()
    merra2_min_anomalies = merra2_min_detrended.groupby("time.dayofyear") - merra2_min_detrended.groupby("time.dayofyear").mean()

    merra2_max_auto_anomalies = analysis.autocorrelation(merra2_max_anomalies)
    merra2_min_auto_anomalies = analysis.autocorrelation(merra2_min_anomalies)
    
    all_max_avg = xarray.open_mfdataset(paths.trefhtmx_members()[0], concat_dim="member", combine="nested")["TREFHTMX"].mean(dim="member").sel(time=slice("1980-01-01", "2015-01-01"))
    all_min_avg =xarray.open_mfdataset(paths.trefhtmn_members()[0], concat_dim="member", combine="nested")["TREFHTMN"].mean(dim="member").sel(time=slice("1980-01-01", "2015-01-01"))

    max_coeffs = all_max_avg.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    min_coeffs = all_min_avg.stack(paired_points=['lat','lon']).groupby('paired_points').apply(lambda data: data.polyfit("time", 1).polyfit_coefficients).unstack()
    all_max_detrended = all_max_avg - xarray.polyval(all_max_avg.time, max_coeffs)
    all_min_detrended = all_min_avg - xarray.polyval(all_min_avg.time, min_coeffs)
    all_max_anomalies = all_max_detrended.groupby("time.dayofyear") - all_max_detrended.groupby("time.dayofyear").mean()
    all_min_anomalies = all_min_detrended.groupby("time.dayofyear") - all_min_detrended.groupby("time.dayofyear").mean()
    
    all_max_auto_anomalies = analysis.autocorrelation(all_max_anomalies.drop("dayofyear"))
    all_min_auto_anomalies = analysis.autocorrelation(all_min_anomalies.drop("dayofyear"))
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 15), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"MERRA2 vs ALL Anomalies Autocorrelation", fontsize=26)
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})

    cmap="hot"
    vmax=1
    vmin=0

    merra2_max_auto_anomalies.plot(ax=ax1, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title("MERRA2 Temp. Max. Autocorrelation 1980-2014")
    ax1.coastlines()
    merra2_min_auto_anomalies.plot(ax=ax2, cmap=cmap, vmax=vmax, vmin=vmin)
    ax2.set_title("MERRA2 Temp. Min. Autocorrelation 1980-2014")
    ax2.coastlines()
    all_max_auto_anomalies.plot(ax=ax3, cmap=cmap, vmax=vmax, vmin=vmin)
    ax3.set_title("ALL Temp. Max. Autocorrelation 1980-2014")
    ax3.coastlines()
    all_min_auto_anomalies.plot(ax=ax4, cmap=cmap, vmax=vmax, vmin=vmin)
    ax4.set_title("ALL Temp. Min. Autocorrelation 1980-2014")
    ax4.coastlines()

    f.tight_layout()
    return f


def gen_isolated_signal_map(time_begin: int, time_end: int, var: str, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{var} Isoalted Signal Maps EXP-{exp_num} from {time_begin} to {time_end}", fontsize=26)

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_min_tavg.plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (all_min_tavg - xghg_min_tavg).plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("GHG Min. Avg.")
    ax2.coastlines()
    (all_min_tavg - xaer_min_tavg).plot(ax=ax3, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax3.set_title("AER Min. Avg.")
    ax3.coastlines()

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_max_tavg.plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (all_max_tavg - xghg_max_tavg).plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("GHG Max. Avg.")
    ax5.coastlines()
    (all_max_tavg - xaer_max_tavg).plot(ax=ax6, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax6.set_title("AER Max. Avg.")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_member_range_isolated_signal_map(time_begin: int, time_end: int, var: str, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_minval = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)
    xghg_minval = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)
    xaer_minval = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)
    
    all_min_var = all_minval.mean(dim="time").max(dim="member")[f"{var}_tn90"].dt.days - all_minval.mean(dim="time").min(dim="member")[f"{var}_tn90"].dt.days
    xghg_min_var = xghg_minval.mean(dim="time").max(dim="member")[f"{var}_tn90"].dt.days - xghg_minval.mean(dim="time").min(dim="member")[f"{var}_tn90"].dt.days
    xaer_min_var = xaer_minval.mean(dim="time").max(dim="member")[f"{var}_tn90"].dt.days - xaer_minval.mean(dim="time").min(dim="member")[f"{var}_tn90"].dt.days
    
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_maxval = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)
    xghg_maxval = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)
    xaer_maxval = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="member", combine="nested", preprocess=time_slice)

    all_max_var = all_maxval.mean(dim="time").max(dim="member")[f"{var}_tx90"].dt.days - all_maxval.mean(dim="time").min(dim="member")[f"{var}_tx90"].dt.days
    xghg_max_var = xghg_maxval.mean(dim="time").max(dim="member")[f"{var}_tx90"].dt.days - xghg_maxval.mean(dim="time").min(dim="member")[f"{var}_tx90"].dt.days
    xaer_max_var = xaer_maxval.mean(dim="time").max(dim="member")[f"{var}_tx90"].dt.days - xaer_maxval.mean(dim="time").min(dim="member")[f"{var}_tx90"].dt.days
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{var} Ensemble Member Range Maps EXP-{exp_num} from {time_begin} to {time_end}", fontsize=26)

    cmap="Reds"
    vmin=0
    vmax=40
    #norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_min_var.plot(ax=ax1, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Range")
    ax1.coastlines()
    xghg_min_var.plot(ax=ax2, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax2.set_title("XGHG Min. Range")
    ax2.coastlines()
    xaer_min_var.plot(ax=ax3, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax3.set_title("XAER Min. Range")
    ax3.coastlines()

    all_max_var.plot(ax=ax4, vmax=vmax, vmin=vmin, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Range")
    ax4.coastlines()
    xghg_max_var.plot(ax=ax5, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax5.set_title("XGHG Max. Range")
    ax5.coastlines()
    xaer_max_var.plot(ax=ax6, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    ax6.set_title("XAER Max. Range")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_global_spatial_avg(var: str, exp_num: str):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 25), facecolor='w')
    f.suptitle(f"{var} Exp. {exp_num} Global Spatial Average Over Time", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)

    lat_lon_avg = lambda ds : ds.mean(dim="lat").mean(dim="lon")

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tn90"].dt.days

    all_min_savgs.plot(ax=ax1, alpha=0.7, color="green", linestyle="dotted")
    (all_min_savgs.groupby("time").mean()).plot(ax=ax1, color="green", linewidth=4, label="ALL")
    xghg_min_savgs.plot(ax=ax1, alpha=0.7, color="red", linestyle="dotted")
    (xghg_min_savgs.groupby("time").mean()).plot(ax=ax1, color="red", linewidth=4, label="XGHG")
    xaer_min_savgs.plot(ax=ax1, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_min_savgs.groupby("time").mean()).plot(ax=ax1, color="blue", linewidth=4, label="XAER")

    ax1.grid()
    ax1.set_title("Using Min. Temperature")
    ax1.legend()

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{var}_tx90"].dt.days

    all_max_savgs.plot(ax=ax2, alpha=0.7, color="green", linestyle="dotted")
    (all_max_savgs.groupby("time").mean()).plot(ax=ax2, color="green", linewidth=4, label="ALL")
    xghg_max_savgs.plot(ax=ax2, alpha=0.7, color="red", linestyle="dotted")
    (xghg_max_savgs.groupby("time").mean()).plot(ax=ax2, color="red", linewidth=4, label="XGHG")
    xaer_max_savgs.plot(ax=ax2, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_max_savgs.groupby("time").mean()).plot(ax=ax2, color="blue", linewidth=4, label="XAER")

    ax2.grid()
    ax2.set_title("Using Max. Temperature")
    ax2.legend()

    f.tight_layout()
    return f


def gen_merra2_comparison(var: str, exp_num: str):
    time_range=[str(date) for date in list(range(1980, 2015 + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{var} ALL vs MERRA2 Comparinson EXP-{exp_num} from 1980 to 2015", fontsize=26)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    merra2_min = bilinear_interpolation(xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])["HWF_tn90pct"].mean(dim="time").dt.days, all_min_tavg)
    merra2_max = bilinear_interpolation(xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])["HWF_tx90pct"].mean(dim="time").dt.days, all_max_tavg)

    cmap="seismic"
    vmin=-60
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    perc_vmin=-200
    perc_vmax=300
    perc_norm = colors.TwoSlopeNorm(vmin=perc_vmin, vcenter=0, vmax=perc_vmax)

    all_min_tavg.plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    merra2_min.plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("MERRA2 Min. Avg.")
    ax2.coastlines()
    (((all_min_tavg - merra2_min) / merra2_min)*100).rename("%").plot(ax=ax3, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax3.set_title("MERRA2-ALL Min. Percent Differences")
    ax3.coastlines()

    all_max_tavg.plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    merra2_max.plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("MERRA2 Max. Avg.")
    ax5.coastlines()
    (((all_max_tavg - merra2_max) / merra2_max)*100).rename("%").plot(ax=ax6, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax6.set_title("MERRA2-ALL Max. Percent Differences")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_ar6_regional_avgs(time_begin: int, time_end: int, var: str, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    spatial_avg_time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tn90"].dt.days

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{var}_tx90"].dt.days

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 35), facecolor='w')
    f.suptitle(f"{var} Exp. {exp_num} AR6 Regional Averages from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 16}
    rc('font', **font)

    all_max_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_max_tavg.groupby(ar6.land.mask(all_max_tavg)).mean().values)]
    xghg_max_values = [value for value in xghg_max_tavg.groupby(ar6.land.mask(xghg_max_tavg)).mean().values]
    xaer_max_values = [value for value in xaer_max_tavg.groupby(ar6.land.mask(xaer_max_tavg)).mean().values]

    bar_height = 5
    y_labels = []
    legend_elements = [Patch(facecolor='blue', label='ALL'),
                      Patch(facecolor='green', label='XGHG'),
                      Patch(facecolor='red', label='XAER')]

    for region, all_value in all_max_regions:
        num = region.number
        y_labels.append(region.name)
        xghg_value = xghg_max_values[num]
        xaer_value = xaer_max_values[num]
        plot_all = lambda : ax1.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax1.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax1.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        plots.sort()
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax1.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax1.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax1.set_title("Using Max. Temperature", fontsize=30)
    ax1.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax1.set_yticklabels(y_labels)
    ax1.grid()
    ax1.legend(handles=legend_elements, loc='upper right')

    all_min_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_min_tavg.groupby(ar6.land.mask(all_min_tavg)).mean().values)]
    xghg_min_values = [value for value in xghg_min_tavg.groupby(ar6.land.mask(xghg_min_tavg)).mean().values]
    xaer_min_values = [value for value in xaer_min_tavg.groupby(ar6.land.mask(xaer_min_tavg)).mean().values]

    for region, all_value in all_min_regions:
        num = region.number
        xghg_value = xghg_min_values[num]
        xaer_value = xaer_min_values[num]
        plot_all = lambda : ax2.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax2.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax2.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax2.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax2.text(val, num*bar_height-1, np.round(val, 0), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax2.set_title("Using Min. Temperature", fontsize=30)
    ax2.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax2.set_yticklabels(y_labels)
    ax2.grid()
    ax2.legend(handles=legend_elements, loc='upper right')

    return f


def gen_fig_deck(variable: str, exp_num: str) -> None:
    with PdfPages(f'{variable}_{exp_num}_figure_deck.pdf') as pdf:
        print("Generating isolate signal maps for time series...")
        pdf.savefig(gen_isolated_signal_map(1930, 1960, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(1960, 1990, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(1990, 2020, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(2020, 2050, variable, exp_num))
        pdf.savefig(gen_isolated_signal_map(2050, 2080, variable, exp_num))
        print("Generating AR6 regional averages for time series...")
        pdf.savefig(gen_ar6_regional_avgs(1930, 1960, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(1960, 1990, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(1990, 2020, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(2020, 2050, variable, exp_num))
        pdf.savefig(gen_ar6_regional_avgs(2050, 2080, variable, exp_num))
        print("Generating global spatial average maps...")
        pdf.savefig(gen_global_spatial_avg(variable, exp_num))
        print("Generating MERRA2 comparison maps...")
        pdf.savefig(gen_merra2_comparison(variable, exp_num))
        

def gen_isolated_signal_map_ratio(numerator: str, denominator: str, time_begin: int, time_end: int, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / all_min_tavg
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / xghg_min_tavg
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days.where(all_min_tavg != 0) / xaer_min_tavg
    all_min_tavg = all_min_tavg*100
    xghg_min_tavg = xghg_min_tavg*100
    xaer_min_tavg = xaer_min_tavg*100
    
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / all_max_tavg
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / xghg_max_tavg
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days.where(all_min_tavg != 0) / xaer_max_tavg
    all_max_tavg = all_max_tavg*100
    xghg_max_tavg = xghg_max_tavg*100
    xaer_max_tavg = xaer_max_tavg*100
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{numerator}/{denominator} Isoalted Signal Maps EXP-{exp_num} from {time_begin} to {time_end}", fontsize=26)

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_min_tavg.rename("%").plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (all_min_tavg - xghg_min_tavg).rename("%").plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("GHG Min. Avg.")
    ax2.coastlines()
    (all_min_tavg - xaer_min_tavg).rename("%").plot(ax=ax3, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax3.set_title("AER Min. Avg.")
    ax3.coastlines()

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    all_max_tavg.rename("%").plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (all_max_tavg - xghg_max_tavg).rename("%").plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("GHG Max. Avg.")
    ax5.coastlines()
    (all_max_tavg - xaer_max_tavg).rename("%").plot(ax=ax6, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax6.set_title("AER Max. Avg.")
    ax6.coastlines()

    f.tight_layout()
    return f


def gen_global_spatial_avg_ratio(numerator: str, denominator: str, exp_num: str):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 25), facecolor='w')
    f.suptitle(f"{numerator}/{denominator} Exp. {exp_num} Global Spatial Average Over Time", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)

    lat_lon_avg = lambda ds : ds.mean(dim="lat").mean(dim="lon")

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tn90"].dt.days

    all_min_savgs = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / all_min_savgs.where(all_min_savgs != 0)
    xghg_min_savgs = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / xghg_min_savgs.where(xghg_min_savgs != 0)
    xaer_min_savgs = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tn90"].dt.days / xaer_min_savgs.where(xaer_min_savgs != 0)

    all_min_savgs.plot(ax=ax1, alpha=0.7, color="green", linestyle="dotted")
    (all_min_savgs.groupby("time").mean()).plot(ax=ax1, color="green", linewidth=4, label="ALL")
    xghg_min_savgs.plot(ax=ax1, alpha=0.7, color="red", linestyle="dotted")
    (xghg_min_savgs.groupby("time").mean()).plot(ax=ax1, color="red", linewidth=4, label="XGHG")
    xaer_min_savgs.plot(ax=ax1, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_min_savgs.groupby("time").mean()).plot(ax=ax1, color="blue", linewidth=4, label="XAER")

    ax1.grid()
    ax1.set_title("Using Min. Temperature")
    ax1.legend()

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{denominator}_tx90"].dt.days

    all_max_savgs = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / all_max_savgs.where(all_max_savgs != 0)
    xghg_max_savgs = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / xghg_max_savgs.where(xghg_max_savgs != 0)
    xaer_max_savgs = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=lat_lon_avg)[f"{numerator}_tx90"].dt.days / xaer_max_savgs.where(xaer_max_savgs != 0)

    all_max_savgs.plot(ax=ax2, alpha=0.7, color="green", linestyle="dotted")
    (all_max_savgs.groupby("time").mean()).plot(ax=ax2, color="green", linewidth=4, label="ALL")
    xghg_max_savgs.plot(ax=ax2, alpha=0.7, color="red", linestyle="dotted")
    (xghg_max_savgs.groupby("time").mean()).plot(ax=ax2, color="red", linewidth=4, label="XGHG")
    xaer_max_savgs.plot(ax=ax2, alpha=0.7, color="blue", linestyle="dotted")
    (xaer_max_savgs.groupby("time").mean()).plot(ax=ax2, color="blue", linewidth=4, label="XAER")

    ax2.grid()
    ax2.set_title("Using Max. Temperature")
    ax2.legend()

    f.tight_layout()
    return f
    
    
def gen_merra2_comparison_ratio(numerator: str, denominator: str, exp_num: str):
    time_range=[str(date) for date in list(range(1980, 2015 + 1))]
    time_slice = lambda ds : ds.sel(time=time_range)
    
    rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 22})
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{numerator}/{denominator} ALL vs MERRA2 Comparinson EXP-{exp_num} from {1980} to {2015}", fontsize=26)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / all_min_tavg.where(all_min_tavg != 0)
    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days / all_max_tavg.where(all_max_tavg != 0)
    merra2_min = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])[f"{denominator}_tn90pct"].mean(dim="time").dt.days
    merra2_min = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tn" in path][0])[f"{numerator}_tn90pct"].mean(dim="time").dt.days / merra2_min.where(merra2_min != 0)
    merra2_min = bilinear_interpolation(merra2_min, all_min_tavg)
    merra2_max = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])[f"{denominator}_tx90pct"].mean(dim="time").dt.days
    merra2_max = xarray.open_dataset([path for path in paths.heat_out_merra2() if exp_num in path and "tx" in path][0])[f"{numerator}_tx90pct"].mean(dim="time").dt.days / merra2_max.where(merra2_max != 0)
    merra2_max = bilinear_interpolation(merra2_max, all_max_tavg)

    cmap="seismic"
    vmin=-20
    vmax=60
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    perc_vmin=-200
    perc_vmax=300
    perc_norm = colors.TwoSlopeNorm(vmin=perc_vmin, vcenter=0, vmax=perc_vmax)

    (all_min_tavg*100).plot(ax=ax1, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax1.set_title("ALL Min. Avg.")
    ax1.coastlines()
    (merra2_min*100).plot(ax=ax2, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax2.set_title("MERRA2 Min. Avg.")
    ax2.coastlines()
    (((all_min_tavg - merra2_min) / merra2_min)*100).rename("%").plot(ax=ax3, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax3.set_title("MERRA2-ALL Min. Percent Differences")
    ax3.coastlines()

    (all_max_tavg*100).plot(ax=ax4, vmax=vmax, vmin=vmin, norm=norm, cmap=cmap, rasterized=True)
    ax4.set_title("ALL Max. Avg.")
    ax4.coastlines()
    (merra2_max*100).plot(ax=ax5, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, rasterized=True)
    ax5.set_title("MERRA2 Max. Avg.")
    ax5.coastlines()
    (((all_max_tavg - merra2_max) / merra2_max)*100).rename("%").plot(ax=ax6, vmin=perc_vmin, vmax=perc_vmax, norm=perc_norm, cmap=cmap, rasterized=True)
    ax6.set_title("MERRA2-ALL Max. Percent Differences")
    ax6.coastlines()

    f.tight_layout()
    return f
    
    
def gen_ar6_regional_avgs_ratio(numerator: str, denominator: str, time_begin: int, time_end: int, exp_num: str):
    time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
    spatial_avg_time_slice = lambda ds : ds.sel(time=time_range)

    all_min, xghg_min, xaer_min = paths.heat_out_trefht_tmin_members_1920_1950_CONTROL()
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tn90"].dt.days
    all_min_tavg = xarray.open_mfdataset([path for path in all_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / all_min_tavg.where(all_min_tavg != 0)
    xghg_min_tavg = xarray.open_mfdataset([path for path in xghg_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / xghg_min_tavg.where(xghg_min_tavg != 0)
    xaer_min_tavg = xarray.open_mfdataset([path for path in xaer_min if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tn90"].dt.days / xaer_min_tavg.where(xaer_min_tavg != 0)
    all_min_tavg = all_min_tavg*100
    xghg_min_tavg = xghg_min_tavg*100
    xaer_min_tavg = xaer_min_tavg*100

    all_max, xghg_max, xaer_max = paths.heat_out_trefht_tmax_members_1920_1950_CONTROL()
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{denominator}_tx90"].dt.days
    all_max_tavg = xarray.open_mfdataset([path for path in all_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / all_max_tavg.where(all_max_tavg != 0)
    xghg_max_tavg = xarray.open_mfdataset([path for path in xghg_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / xghg_max_tavg.where(xghg_max_tavg != 0)
    xaer_max_tavg = xarray.open_mfdataset([path for path in xaer_max if exp_num in path], concat_dim="time", combine="nested", preprocess=spatial_avg_time_slice).mean(dim="time")[f"{numerator}_tx90"].dt.days / xaer_max_tavg.where(xaer_max_tavg != 0)
    all_max_tavg = all_max_tavg*100
    xghg_max_tavg = xghg_max_tavg*100
    xaer_max_tavg = xaer_max_tavg*100

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 35), facecolor='w')
    f.suptitle(f"{numerator}/{denominator} Exp. {exp_num} AR6 Regional Averages from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 16}
    rc('font', **font)

    all_max_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_max_tavg.groupby(ar6.land.mask(all_max_tavg)).mean().values)]
    xghg_max_values = [value for value in xghg_max_tavg.groupby(ar6.land.mask(xghg_max_tavg)).mean().values]
    xaer_max_values = [value for value in xaer_max_tavg.groupby(ar6.land.mask(xaer_max_tavg)).mean().values]

    bar_height = 5
    y_labels = []
    legend_elements = [Patch(facecolor='blue', label='ALL'),
                      Patch(facecolor='green', label='XGHG'),
                      Patch(facecolor='red', label='XAER')]

    for region, all_value in all_max_regions:
        num = region.number
        y_labels.append(region.name)
        xghg_value = xghg_max_values[num]
        xaer_value = xaer_max_values[num]
        plot_all = lambda : ax1.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax1.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax1.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax1.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax1.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax1.set_title("Using Max. Temperature (%)", fontsize=30)
    ax1.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax1.set_yticklabels(y_labels)
    ax1.grid()
    ax1.legend(handles=legend_elements, loc='upper right')

    all_min_regions = [(ar6.land.regions[index], value) for index, value in enumerate(all_min_tavg.groupby(ar6.land.mask(all_min_tavg)).mean().values)]
    xghg_min_values = [value for value in xghg_min_tavg.groupby(ar6.land.mask(xghg_min_tavg)).mean().values]
    xaer_min_values = [value for value in xaer_min_tavg.groupby(ar6.land.mask(xaer_min_tavg)).mean().values]

    for region, all_value in all_min_regions:
        num = region.number
        xghg_value = xghg_min_values[num]
        xaer_value = xaer_min_values[num]
        plot_all = lambda : ax2.barh(num*bar_height, all_value, height=bar_height-1, align='center', color="blue")
        plot_xghg = lambda : ax2.barh(num*bar_height, xghg_value, height=bar_height-1, align='center', color="green")
        plot_xaer = lambda : ax2.barh(num*bar_height, xaer_value, height=bar_height-1, align='center', color="red")
        plots = [(all_value, plot_all), (xghg_value, plot_xghg), (xaer_value, plot_xaer)]
        try:
            plots.sort()
        except TypeError:
            pass
        for index, (val, func) in enumerate(plots[::-1]):
            func()
            if index == 1:
                ax2.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
            else:
                ax2.text(val, num*bar_height-1, np.round(val, 1), color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
    ax2.set_title("Using Min. Temperature (%)", fontsize=30)
    ax2.set_yticks(np.linspace(0, 225, len(y_labels)))
    ax2.set_yticklabels(y_labels)
    ax2.grid()
    ax2.legend(handles=legend_elements, loc='upper right')
    return f


def gen_ratio_fig_deck(numerator: str, denominator: str, exp_num: str) -> None:
    with PdfPages(f'{numerator}-{denominator}_{exp_num}_figure_deck.pdf') as pdf:
        print("Generating isolate signal maps for time series...")
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1930, 1960, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1960, 1990, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 1990, 2020, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 2020, 2050, exp_num))
        pdf.savefig(gen_isolated_signal_map_ratio(numerator, denominator, 2050, 2080, exp_num))
        print("Generating AR6 regional averages for time series...")
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1930, 1960, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1960, 1990, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 1990, 2020, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 2020, 2050, exp_num))
        pdf.savefig(gen_ar6_regional_avgs_ratio(numerator, denominator, 2050, 2080, exp_num))
        print("Generating global spatial average maps...")
        pdf.savefig(gen_global_spatial_avg_ratio(numerator, denominator, exp_num))
        print("Generating MERRA2 comparison maps...")
        pdf.savefig(gen_merra2_comparison_ratio(numerator, denominator, exp_num))

        
def gen_data_sim_1920_1950_thresholds() -> None:
    """
    Generates thresholds using 1920-1950 base period for ALL, XGHG, and XAER in appropriate directories
    This is deprecated at the momnent and will probably be removed (we use the pre-industrial control threshold now)
    """
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
        
        
def gen_data_control_1920_1950_threshold() -> None:
    """
    Generates control threshold in appropriate directory
    Set up for generating multiple thresholds later (will need to modify the base period)
    """
    trefht, trefhtmn, trefhtmx = paths.control_downloads()
    analysis.calculate_thresholds(paths.DIR_PATH + "thresholds/CONTROL/control_threshold.nc", trefhtmx, "TREFHTMX", trefhtmn, "TREFHTMN", bp="1920-1950", control_data=True)


def gen_data_heatwave_statistics() -> None:
    """
    Generates heatwave statistics for ALL, XGHG, XAER, and MERRA2 ensemble members in the approriate directories
    """
    all_mn, xghg_mn, xaer_mn = paths.trefhtmn_members()
    all_mx, xghg_mx, xaer_mx = paths.trefhtmx_members()
    threshold_path = paths.control_threshold()
    exp_nums = ["3114", "3336" ,"3314", "3236", "3214", "3136", "1112", "1212", "1312", "1111"]
    
    for num in exp_nums:
        for index, ds_mn in enumerate(all_mn):
            print(f"{num} ALL {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/ALL/1920_1950_control_base/ALL-{num}-{index}-", all_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xghg_mn):
            print(f"{num} XGHG {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/XGHG/1920_1950_control_base/XGHG-{num}-{index}-", xghg_mx[index], ds_mn, threshold_path, num)
        for index, ds_mn in enumerate(xaer_mn):
            print(f"{num} XAER {index}")
            analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/heat_output/XAER/1920_1950_control_base/XAER-{num}-{index}-", xaer_mx[index], ds_mn, threshold_path, num)
        merra2_path = paths.merra2_download()
        analysis.calculate_heatwave_statistics(f"{paths.DIR_PATH}/MERRA2/heat_ouputs/MERRA2-{num}-", merra2_path, merra2_path, threshold_path, num, tmaxvname="T2MMAX", tminvname="T2MMIN", model="MERRA2")
        
        