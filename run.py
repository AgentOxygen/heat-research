import pop_weighted as pw
import xarray
import numpy as np
from regionmask.defined_regions import ar6
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from matplotlib.patches import Patch
import paths
# from matplotlib.backends.backend_pdf import PdfPages
time_begin=2020
time_end=2040
var = "HWF"
exp_num = "3136"

# #weighted_all, weighted_xghg, weighted_xaer, all_data, xghg_data, xaer_data, error = pw.weighted(variable, exp_num, min_max)
# weighted_all = xarray.open_dataset("weighted_all.nc")
# weighted_xghg = xarray.open_dataset("weighted_xghg.nc")
# weighted_xaer = xarray.open_dataset("weighted_xaer.nc")
# all_data = xarray.open_dataset("shifted_all_data.nc")
# xaer_data = xarray.open_dataset("shifted_xaer_data.nc")
# xghg_data = xarray.open_dataset("shifted_xghg_data.nc")

time_range=[str(date) for date in list(range(time_begin, time_end + 1))]
spatial_avg_time_slice = lambda ds : ds.sel(time=time_range)

all_min_tavg, xghg_min_tavg, xaer_min_tavg, all_min, xghg_min, xaer_min, error_min = pw.weighted(var, exp_num, "tn")

all_max_tavg, xghg_max_tavg, xaer_max_tavg, all_max, xghg_max, xaer_max, error_max = pw.weighted(var, exp_num, "tx")

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
f.savefig("out.png")