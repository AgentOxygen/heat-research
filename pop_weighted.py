import xarray
import xesmf as xe
import numpy as np
from paths import heat_out_trefht_tmax_members_1920_1950_CONTROL as tmax_paths
from paths import heat_out_trefht_tmin_members_1920_1950_CONTROL as tmin_paths
from paths import population_2020_aggregated

def weighted(variable: str, exp_num: str, min_max: str) -> xarray.DataArray:
    if min_max == "tx":
        member_paths = [path for path in tmax_paths()[0] if exp_num in path]
        heat_data = xarray.open_mfdataset(member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    elif min_max == "tn":
        member_paths = [path for path in tmin_paths()[0] if exp_num in path]
        heat_data = xarray.open_mfdataset(member_paths, concat_dim="member", combine="nested")[f"{variable}_{min_max}90"].dt.days.mean(dim="member")
    else:
        return None
    shift_heat_data = heat_data.assign_coords(lon=(((heat_data.lon + 180) % 360) - 180)).sortby('lon')
    pop_data = xarray.open_rasterio(population_2020_aggregated()).rename({"x":"lon", "y":"lat"})
    
    
    lat_delta = (shift_heat_data.lat.values[1] - shift_heat_data.lat.values[0]) / 2
    lon_delta = (shift_heat_data.lon.values[1] - shift_heat_data.lon.values[0]) / 2
    lats = np.append(shift_heat_data.lat.values - lat_delta, shift_heat_data.lat.values[-1::])
    lons = np.append(shift_heat_data.lon.values - lon_delta, shift_heat_data.lon.values[-1::])

    resampled_data = pop_data.where(pop_data > 0).astype(np.float64).groupby_bins("lat", lats).sum().groupby_bins("lon", lons).sum()
    new_data = resampled_data.rename({"lat_bins":"lon", "lon_bins":"lat"}).mean(dim="band")
    error = new_data.sum(dim="lat").sum(dim="lon") - np.sum(pop_data.where(pop_data > 0))

    weighted_new = shift_heat_data.groupby("time").apply(lambda data: np.multiply(new_data.values, data.values))

    return weighted_new, shift_heat_data, error


