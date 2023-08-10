import xarray
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from paths import heat_out_trefht_tmin_members_1920_1950_CONTROL as tmin, land_mask, aod_vis_concatenated as aod_members
from scipy.stats import ttest_ind

exp_num = "3136"
var = "HWF"

all_minp, xghg_minp, xaer_minp = tmin()
land_m = xarray.open_dataset(land_mask())["__xarray_dataarray_variable__"]
all_min = xarray.open_mfdataset([path for path in all_minp if exp_num in path], concat_dim="member", combine="nested")[f"{var}_tn90"].dt.days.sel(time=slice(1920,2079))
xaer_min = xarray.open_mfdataset([path for path in xaer_minp if exp_num in path], concat_dim="member", combine="nested")[f"{var}_tn90"].dt.days.sel(time=slice(1920,2079))
all_min = all_min.where(land_m > 0)
xaer_min = xaer_min.where(land_m > 0)

aod_data = xarray.open_mfdataset(aod_members(), concat_dim="member", combine="nested").mean(dim="member")
aod_data = aod_data["AODVIS"].groupby("time.year").mean().load()

def sig_mask(all_ds, xaer_ds):
    
    def welch_t_test(all_pt, xaer_pt, lat, lon):
        # Confirmed that Sarah's modified t test is just a welches t test using my own code
        t_val, p_val = ttest_ind(all_pt, xaer_pt, equal_var=False)
        return ((lat, lon), p_val)
        
    #print("Converting to numpy array...")
    # (member, time, lat, lon)
    all_array = all_ds.values
    xaer_array = xaer_ds.values
    #print("Done.")
    
    results = []
    #print(f"Preforming test, modified={modified}")
    for lati, lat in enumerate(all_ds.lat.values):
        for loni, lon in enumerate(all_ds.lon.values):
            all_pt = all_array[0:all_ds["member"].size, lati, loni].flatten()
            xaer_pt = xaer_array[0:xaer_ds["member"].size, lati, loni].flatten()
            results.append(welch_t_test(all_pt, xaer_pt, lat, lon))
    
    if len(results) == all_min.lat.size*all_min.lon.size:
        #print("Shapes match, converting back to xarray...")
        pass
    else:
        raise RuntimeError(f'The mask size does not match the original dataset size: {len(mask)} != {all_min.lat.size*all_min.lon.size}')
    
    # Recycle ALL array
    return_array = (all_ds.mean(dim="member").load() * 0).rename("p-value")
    for index, ((lat, lon), value) in enumerate(results):
        return_array.loc[dict(lat=lat, lon=lon)] = value
    
    return return_array

for year in aod_data.year.values:
    f, ax = plt.subplots(1, 1, figsize=(14, 7), facecolor='w', subplot_kw={'projection': ccrs.PlateCarree()})
    f.suptitle(f"Evolution of AOD relative to 1920: {year}", fontsize=30)

    mask = sig_mask(all_min.sel(time=year), xaer_min.sel(time=year))
    mask = mask.where(mask<=0.05).where(land_m>0)
    
    (aod_data.sel(year=year) - aod_data.sel(year=1920)).plot(ax=ax, vmax=0.1, vmin=0, cmap="Blues", rasterized=True)
    X, Y = np.meshgrid(mask.lon, mask.lat)
    ax.hexbin(X.reshape(-1), Y.reshape(-1), mask.data.reshape(-1), hatch='x', alpha=0, transform=ccrs.PlateCarree())
    
    ax.coastlines()
    f.savefig(f"./tmp/{year}.png")
    plt.close()
    print(year, end=", ")

# Build GIF
from os import listdir
with imageio.get_writer('animation.gif', mode='I') as writer:
    paths = [f"./tmp/{path}" for path in listdir("tmp") if ".png" in path]
    paths.sort()
    for filename in paths:
        image = imageio.imread(filename)
        writer.append_data(image)