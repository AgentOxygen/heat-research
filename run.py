from paths import trefhtmn_members
from paths import trefhtmx_members
from paths import land_mask
import xarray

p = [path for path in trefhtmn_members()[0] if ".nc" in path]
cesm_mn_var = None
for path in p:
    print("i", end="")
    if cesm_mn_var == None:
        cesm_mn_var = xarray.open_dataset(path)["TREFHTMN"].groupby("time.dayofyear").mean()
    else:
        cesm_mn_var = (cesm_mn_var+xarray.open_dataset(path)["TREFHTMN"].groupby("time.dayofyear").mean())/2
cesm_mn_var.to_netcdf("TREFHTMN_var.nc")

p = [path for path in trefhtmx_members()[0] if ".nc" in path]
cesm_mn_var = None
for path in p:
    print("i", end="")
    if cesm_mn_var == None:
        cesm_mn_var = xarray.open_dataset(path)["TREFHTMX"].groupby("time.dayofyear").mean()
    else:
        cesm_mn_var = (cesm_mn_var+xarray.open_dataset(path)["TREFHTMX"].groupby("time.dayofyear").mean())/2
cesm_mn_var.to_netcdf("TREFHTMX_var.nc")