# Give the file with the variable specified. Example ( uncertainty_3d(S4_AREL.AREL, S3_AREL.AREL) )
# A is s4, 2000 aerosol levels
# B is s3, 1850 aerosol levels
# This function will take the last 40 years to allow for better representation

# Also will need to check if the variable needs to be averaged across levels. This function does take the vertically averaged values

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray.ufuncs as xrf
from scipy import stats

from os import listdir

def uncertainty_3d(a, b):
    s4_variable = a.isel(time=slice(227,720)).mean(dim='lev')  # Here im only wanting a time slice, not a mean along time yet.
    s3_variable = b.isel(time=slice(227,720)).mean(dim='lev')  # Here im only wanting a time slice, not a mean along time yet.
    
    from scipy.stats import ttest_ind # this is the function for computing the Welch's t-test
    
    # Here we convert from xarray to numpy, losing most attributes and features but increasing performance
    s4_array = s4_variable.values
    s3_array = s3_variable.values
    
    def welch_t_test(s4_pt, s3_pt, lat, lon):
        # Confirmed that Sarah's modified t test is just a welches t test using my own code
        t_val, p_val = ttest_ind(s4_pt, s3_pt, equal_var=False)
        return ((lat, lon), p_val)
    results = []
    # We then iterate through each point in the datasets
    for lati, lat in enumerate(s4_variable.lat.values):
        for loni, lon in enumerate(s4_variable.lon.values):
            # This is when things get a bit tricky. Basically, since the numpy arrays don't have fancy labels
            # they get treated as true arrays. Meaning we need to access them like arrays are accessed in C or
            # other lower-level programming languages: we use indices, not values.
            # The first coordinate in the array accesses members, and here we are specifying to access indices
            # 0 through the size of the array, so basically all members.
            # The second coordinate accesses time, and we do the same: from 0 to the end.
            # The thrid and fourth access lat and lon respectively, so we only one for each that specify the point
            s4_pt = s4_array[0:s4_variable["time"].size, lati, loni].flatten()
            # Lastly, we flatten the array, which basically takes all elements and puts them into a single dimension
            # The result is a 1-d array with size equal to the number of members times the number of years
            s3_pt = s3_array[0:s3_variable["time"].size, lati, loni].flatten()
        
            # With the two 1-d arrays, we can preform a t-test and obtain a t-value and p-value
            t_value, p_value = ttest_ind(s4_pt, s3_pt, equal_var=False) # We specify that the variances are not equal
            # We then store the p_value in an array along with the coordinates for the point in a tuple
            results.append(((lat, lon), p_value))
            
    # To carry over our attributes, we make a copy of the ALL dataset, set everything to zero, and rename the variable
    sig_results = (s4_variable.mean(dim="time").load() * 0).rename("p-value")
    # We then iterate through each point in the results
    for (lat, lon), p_value in results: # Note how we unpack the tuple in the for loop header
        # We then locate the point in the ALL copy we made and change the value from 0 to whatever the p-value is
        sig_results.loc[dict(lat=lat, lon=lon)] = p_value
        
        
        
    # Overlaying this on results means we just have to drop the hexbin alpha to zero as not to interfere with our primary plot
    sig_mask = sig_results.where(sig_results >= 0.1)

    X, Y = np.meshgrid(sig_results.lon, sig_results.lat) 
    f, ax = plt.subplots(1, 1, figsize=(15, 8), facecolor='w', subplot_kw={'projection': ccrs.Robinson()})
    f.suptitle(f"Significance of "+ a.long_name, fontsize=26)

    (s4_variable-s3_variable).mean(dim="time").plot(ax=ax, transform=ccrs.PlateCarree())
    ax.hexbin(X.reshape(-1), Y.reshape(-1), sig_mask.data.reshape(-1), hatch='x', alpha=0, transform=ccrs.PlateCarree())
    ax.coastlines()