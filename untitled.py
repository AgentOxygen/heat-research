from paths import trefhtmn_members, trefhtmx_members
import xarray
import cftime
from dask.distributed import Client, progress

client = Client(processes=True, threads_per_worker=2,
                n_workers=24, memory_limit="20GB")

chunk={"lat":48, "lon":48}
all_mn_ds = xarray.open_mfdataset([path for path in trefhtmn_members()[0] if ".nc" in path], parallel=True, concat_dim="member", combine="nested", chunks=chunk)["TREFHTMN"].sel(time=slice(cftime.DatetimeNoLeap(1980, 1, 1, 0, 0, 0, 0, has_year_zero=True), cftime.DatetimeNoLeap(2015, 12, 30, 0, 0, 0, 0, has_year_zero=True))).mean(dim="member").groupby("time.dayofyear").mean().var(dim="dayofyear")
all_mx_ds = xarray.open_mfdataset([path for path in trefhtmx_members()[0] if ".nc" in path], parallel=True, concat_dim="member", combine="nested", chunks=chunk)["TREFHTMX"].sel(time=slice(cftime.DatetimeNoLeap(1980, 1, 1, 0, 0, 0, 0, has_year_zero=True), cftime.DatetimeNoLeap(2015, 12, 30, 0, 0, 0, 0, has_year_zero=True))).mean(dim="member").groupby("time.dayofyear").mean().var(dim="dayofyear")

xaer_mn_ds = xarray.open_mfdataset([path for path in trefhtmn_members()[2] if ".nc" in path], parallel=True, concat_dim="member", combine="nested", chunks=chunk)["TREFHTMN"].sel(time=slice(cftime.DatetimeNoLeap(1980, 1, 1, 0, 0, 0, 0, has_year_zero=True), cftime.DatetimeNoLeap(2015, 12, 30, 0, 0, 0, 0, has_year_zero=True))).mean(dim="member").groupby("time.dayofyear").mean().var(dim="dayofyear")
xaer_mx_ds = xarray.open_mfdataset([path for path in trefhtmx_members()[2] if ".nc" in path], parallel=True, concat_dim="member", combine="nested", chunks=chunk)["TREFHTMX"].sel(time=slice(cftime.DatetimeNoLeap(1980, 1, 1, 0, 0, 0, 0, has_year_zero=True), cftime.DatetimeNoLeap(2015, 12, 30, 0, 0, 0, 0, has_year_zero=True))).mean(dim="member").groupby("time.dayofyear").mean().var(dim="dayofyear")

print("Computing ALL TREFHTMN Variance")
all_mn_variance = client.compute(all_mn_ds).result()
print("Computing ALL TREFHTMX Variance")
all_mx_variance = client.compute(all_mx_ds).result()
print("Computing XAER TREFHTMN Variance")
xaer_mn_variance = client.compute(xaer_mn_ds).result()
print("Computing XAER TREFHTMX Variance")
xaer_mx_variance = client.compute(xaer_mx_ds).result()

client.shutdown()