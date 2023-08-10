from paths import trefhtmn_members
from paths import trefhtmx_members
from paths import land_mask
import xarray

daily_min_temps = xarray.open_mfdataset([path for path in trefhtmn_members()[0] if ".nc" in path], concat_dim="member", combine="nested")["TREFHTMN"].mean(dim="member").load()
daily_min_temps = daily_min_temps.groupby("time.dayofyear").mean().load()

daily_max_temps = xarray.open_mfdataset([path for path in trefhtmx_members()[0] if ".nc" in path], concat_dim="member", combine="nested")["TREFHTMX"].mean(dim="member").load()
daily_max_temps = daily_max_temps.groupby("time.dayofyear").mean().load()


daily_min_temps.var(dim="dayofyear").to_netcdf("tmin_var.nc")
daily_max_temps.var(dim="dayofyear").to_netcdf("tmax_var.nc")
