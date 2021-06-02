from settings import DATA_DIR
import xarray
import numpy as np
from os import listdir, remove
from imageio import get_writer, imread

# Group datasets in directory by variable, type, and former/latter half of the time series
dataset_names = listdir(DATA_DIR)
xaer_datasets = [name for name in dataset_names if 'xaer' in name]
xghg_datasets = [name for name in dataset_names if 'xghg' in name]
all_datasets = [name for name in dataset_names if 'BRCP85C5CNBDRD' in name]

trefht_xaer_latter_datasets = [name for name in xaer_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_xaer_latter_datasets = [name for name in xaer_datasets if '.TREFHTMIN.20060101-20801231' in name]
trefhtmax_xaer_latter_datasets = [name for name in xaer_datasets if '.TREFHTMAX.20060101-20801231' in name]
trefht_xaer_former_datasets = [name for name in xaer_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_xaer_former_datasets = [name for name in xaer_datasets if '.TREFHTMIN.19200101-20051231' in name]
trefhtmax_xaer_former_datasets = [name for name in xaer_datasets if '.TREFHTMAX.19200101-20051231' in name]

trefht_xghg_latter_datasets = [name for name in xghg_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_xghg_latter_datasets = [name for name in xghg_datasets if '.TREFHTMIN.20060101-20801231' in name]
trefhtmax_xghg_latter_datasets = [name for name in xghg_datasets if '.TREFHTMAX.20060101-20801231' in name]
trefht_xghg_former_datasets = [name for name in xghg_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_xghg_former_datasets = [name for name in xghg_datasets if '.TREFHTMIN.19200101-20051231' in name]
trefhtmax_xghg_former_datasets = [name for name in xghg_datasets if '.TREFHTMAX.19200101-20051231' in name]

trefht_all_latter_datasets = [name for name in all_datasets if '.TREFHT.20060101-20801231' in name]
trefhtmin_all_latter_datasets = [name for name in all_datasets if '.TREFHTMIN.20060101-20801231' in name]
trefhtmax_all_latter_datasets = [name for name in all_datasets if '.TREFHTMAX.20060101-20801231' in name]
trefht_all_former_datasets = [name for name in all_datasets if '.TREFHT.19200101-20051231' in name]
trefhtmin_all_former_datasets = [name for name in all_datasets if '.TREFHTMIN.19200101-20051231' in name]
trefhtmax_all_former_datasets = [name for name in all_datasets if '.TREFHTMAX.19200101-20051231' in name]

print("Opening dataset")
ds = xarray.open_dataset(DATA_DIR + trefht_all_latter_datasets[0]).resample(time='Y', dim='time')
print("Creating gif")

filenames = []
for i, data in enumerate(ds.TREFHT):
    plt.pcolor(data)
    filename = f'{i}.png'
    filenames.append(filename)

    plt.savefig(filename)
    plt.close()
    print(str(i))

# build gif
with imageio.get_writer('test.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)