import xarray
import xesmf as xe
import numpy as np
from paths import heat_out_trefht_tmax_members_1920_1950_CONTROL as tmax_paths
from paths import heat_out_trefht_tmin_members_1920_1950_CONTROL as tmin_paths
from paths import population_2020_aggregated
from os.path import exists
import dask


def weighted(variable: str, exp_num: str, min_max: str) -> xarray.DataArray:
    if min_max == "tx":
        all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]
    elif min_max == "tn":
        all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
        xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
        xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]
    else:
        return None
    all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    
    print("Shifting data..")
    
    # Shift data
    all_data = all_data.assign_coords(lon=(((all_data.lon + 180) % 360) - 180)).sortby('lon')
    xaer_data = xaer_data.assign_coords(lon=(((xaer_data.lon + 180) % 360) - 180)).sortby('lon')
    xghg_data = xghg_data.assign_coords(lon=(((xghg_data.lon + 180) % 360) - 180)).sortby('lon')
    
    print("Loading pop data..")
    pop_data = xarray.open_rasterio(population_2020_aggregated()).rename({"x":"lon", "y":"lat"})
    
    lat_delta = (all_data.lat.values[1] - all_data.lat.values[0]) / 2
    lon_delta = (all_data.lon.values[1] - all_data.lon.values[0]) / 2
    lats = np.append(all_data.lat.values - lat_delta, all_data.lat.values[-1::])
    lons = np.append(all_data.lon.values - lon_delta, all_data.lon.values[-1::])

    print("Resampling pop data..")
    resampled_pop = pop_data.where(pop_data > 0).astype(np.float64).groupby_bins("lat", lats).sum().groupby_bins("lon", lons).sum()
    resampled_pop = resampled_pop.rename({"lat_bins":"lat", "lon_bins":"lon"})
    error = resampled_pop.sum(dim="lat").sum(dim="lon") - np.sum(pop_data.where(pop_data > 0))
    
    print("Time Grouping ... ", end="")
    weighted_all = all_data.groupby("time").apply(lambda data: np.multiply(resampled_pop.values, data.values))
    print("ALL ... ", end="")
    weighted_xghg = xghg_data.groupby("time").apply(lambda data: np.multiply(resampled_pop.values, data.values))
    print("XGHG ... ", end="")
    weighted_xaer = xaer_data.groupby("time").apply(lambda data: np.multiply(resampled_pop.values, data.values))
    print("XAER Complete!")
    
    return weighted_all, weighted_xghg, weighted_xaer, all_data, xghg_data, xaer_data, error


def gen_weighted() -> xarray.DataArray:
    print("Generating population-weighted datasets")
    dir_path = "../data/populations/weighted/"
    variables = ["HWF", "HWD"]
    exp_nums = ["3136", "3114", "3336", "3314", "3236", "3214", "1112", "1212", "1312", "1111"]
    init = True
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        for exp_num in exp_nums:
            for variable in variables:
                for min_max in ["tx", "tn"]:
                    if min_max == "tx":
                        all_member_paths = [path for path in tmax_paths()[0] if exp_num in path]
                        xghg_member_paths = [path for path in tmax_paths()[1] if exp_num in path]
                        xaer_member_paths = [path for path in tmax_paths()[2] if exp_num in path]
                    elif min_max == "tn":
                        all_member_paths = [path for path in tmin_paths()[0] if exp_num in path]
                        xghg_member_paths = [path for path in tmin_paths()[1] if exp_num in path]
                        xaer_member_paths = [path for path in tmin_paths()[2] if exp_num in path]
                    all_data = xarray.open_mfdataset(all_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days
                    xaer_data = xarray.open_mfdataset(xaer_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days
                    xghg_data = xarray.open_mfdataset(xghg_member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days


                    # Shift data
                    all_data = all_data.assign_coords(lon=(((all_data.lon + 180) % 360) - 180)).sortby('lon')
                    xaer_data = xaer_data.assign_coords(lon=(((xaer_data.lon + 180) % 360) - 180)).sortby('lon')
                    xghg_data = xghg_data.assign_coords(lon=(((xghg_data.lon + 180) % 360) - 180)).sortby('lon')

                    if init:
                        print("Initializing...")
                        init = False
                        lat_delta = (all_data.lat.values[1] - all_data.lat.values[0]) / 2
                        lon_delta = (all_data.lon.values[1] - all_data.lon.values[0]) / 2
                        lats = np.append(all_data.lat.values - lat_delta, all_data.lat.values[-1::])
                        lons = np.append(all_data.lon.values - lon_delta, all_data.lon.values[-1::])
                        pop_data = xarray.open_rasterio(population_2020_aggregated()).rename({"x":"lon", "y":"lat"})
                        resampled_pop = pop_data.where(pop_data > 0).astype(np.float64).groupby_bins("lat", lats).sum().groupby_bins("lon", lons).sum()
                        resampled_pop = resampled_pop.rename({"lat_bins":"lat", "lon_bins":"lon"}).sel(band=1)

                    all_export_path = f"{dir_path}/ALL/{variable}-{exp_num}-{min_max}.nc"
                    if exists("all_export_path"):
                        print("Already exists: " + all_export_path)
                    else:
                        all_data.groupby("time").apply(lambda time_slice: time_slice*resampled_pop.values).to_netcdf(all_export_path)
                        print(all_export_path)

                    xghg_export_path = f"{dir_path}/XGHG/{variable}-{exp_num}-{min_max}.nc"
                    if exists("xghg_export_path"):
                        print("Already exists: " + xghg_export_path)
                    else:
                        xghg_data.groupby("time").apply(lambda time_slice: time_slice*resampled_pop.values).to_netcdf(xghg_export_path)
                        print(xghg_export_path)

                    xaer_export_path = f"{dir_path}/XAER/{variable}-{exp_num}-{min_max}.nc"
                    if exists("xaer_export_path"):
                        print("Already exists: " + xaer_export_path)
                    else:
                        xaer_data.groupby("time").apply(lambda time_slice: time_slice*resampled_pop.values).to_netcdf(xaer_export_path)
                        print(xaer_export_path)
                
