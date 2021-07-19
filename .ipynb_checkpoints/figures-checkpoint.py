import paths
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from paths import FIGURE_IMAGE_OUTPUT
import cartopy.crs as ccrs
import regionmask
from os import listdir
from matplotlib.backends.backend_pdf import PdfPages

def gen_regional_bar_graphs(exp_num, variable, time_begin, time_end):
    def gen_regional_bar_graph(pdf, label: str, data_array, variable: str, exp_num: str):
        f, ax = plt.subplots(1, 1, figsize=(35, 25), facecolor='w')
        f.suptitle(f"{variable} Exp. {exp_num} AR6 Regional Average {label}", fontsize=50)
        font = {'weight': 'bold',
                'size': 25}
        rc('font', **font)

        land = regionmask.defined_regions.natural_earth.land_110
        ar6 = regionmask.defined_regions.ar6.land

        def get_values(dataset):
            dataset = dataset.where(land.mask(all_max) == 0)
            ar6 = regionmask.defined_regions.ar6.land
            vals_labels = []
            for region in ar6:
                vals_labels.append((region, dataset.where(ar6.mask(dataset) == region.number)
                                    .mean(dim="lat", skipna=True).mean(dim="lon", skipna=True)))
            return vals_labels

        bar_height = 4
        vals_labels = get_values(data_array)
        vals_labels.sort(key = lambda x: x[1])
        x_labels = [""]*len(vals_labels)
        for index, (region, val) in enumerate(vals_labels):
            ax.barh(index*(bar_height+1), val, height=bar_height, align='center', label = region.name, color = "blue")
            x_labels[index] = region.name
        for i, (region, v) in enumerate(vals_labels):
            if v >= 0:
                ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')
            else:
                ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')
        ax.set_yticks(np.linspace(0, (bar_height + 0.9)*(len(vals_labels)), len(vals_labels)))
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(f'{variable}')
        ax.grid()
        f.tight_layout()

        pdf.savefig(f)

    pdf_pages = PdfPages(f'{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-{variable}-AR6-bar.pdf')
    datasets = paths.get_paths_heat_output_avg(tmp_type="max", exp_num=exp_num) + paths.get_paths_heat_output_avg(tmp_type="min", exp_num=exp_num)

    all_max = xarray.open_dataset(datasets[0]).HWF_tx90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    gen_regional_bar_graph(pdf_pages, f"ALL Max. from {time_begin} to {time_end}", all_max, variable, exp_num)
    xghg_max = xarray.open_dataset(datasets[1]).HWF_tx90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    ghg_max = all_max - xghg_max
    gen_regional_bar_graph(pdf_pages, f"GHG Max. from {time_begin} to {time_end}", ghg_max, variable, exp_num)
    xaer_max = xarray.open_dataset(datasets[2]).HWF_tx90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    aer_max = all_max - xaer_max
    gen_regional_bar_graph(pdf_pages, f"AER Max. from {time_begin} to {time_end}", aer_max, variable, exp_num)
    all_min = xarray.open_dataset(datasets[3]).HWF_tn90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    gen_regional_bar_graph(pdf_pages, f"ALL Max. from {time_begin} to {time_end}", all_min, variable, exp_num)
    xghg_min = xarray.open_dataset(datasets[4]).HWF_tn90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    ghg_min = all_min - xghg_min
    gen_regional_bar_graph(pdf_pages, f"GHG Max. from {time_begin} to {time_end}", ghg_min, variable, exp_num)
    xaer_min = xarray.open_dataset(datasets[5]).HWF_tn90pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    aer_min = all_min - xaer_min
    gen_regional_bar_graph(pdf_pages, f"AER Max. from {time_begin} to {time_end}", aer_min, variable, exp_num)
    pdf_pages.close()


def gen_stacking_regional_bar_graphs(exp_num: str, variable: str, time_begin: str, time_end: str, label="": str, max_min="max": str):
    f, ax = plt.subplots(1, 1, figsize=(35, 25), facecolor='w')
    f.suptitle(f"{variable} Exp. {exp_num} AR6 Regional Average {max_min}{label} from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)


    datasets = paths.get_paths_heat_output_avg(tmp_type=min_max, exp_num=exp_num)
    all_data = xarray.open_dataset(datasets[0])[f"HWF_t{min_max[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xghg_data = xarray.open_dataset(datasets[1])[f"HWF_t{min_max[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xaer_data = xarray.open_dataset(datasets[2])[f"HWF_t{min_max[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days

    land = regionmask.defined_regions.natural_earth.land_110
    ar6 = regionmask.defined_regions.ar6.land

    def get_values(dataset):
        dataset = dataset.where(land.mask(dataset) == 0)
        ar6 = regionmask.defined_regions.ar6.land
        vals_labels = []
        for region in ar6:
            vals_labels.append((region, dataset.where(ar6.mask(dataset) == region.number).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True)))
        return vals_labels

    def match_sorting(vals_labels_template, vals_labels_out):
        vals_labels_ret = []
        for region_temp, val_temp in vals_labels_template:
            for region_out, val_out in vals_labels_out:
                if region_out.name == region_temp.name:
                    vals_labels_ret.append((region_out, val_out))
        return vals_labels_ret

    bar_height = 4
    all_vals_labels = get_values(all_data)
    all_vals_labels.sort(key = lambda x: x[1])

    xaer_vals_labels = match_sorting(all_vals_labels, get_values(xaer_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xaer_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "green")
        x_labels[index] = region.name
    bar.set_label("XAER")
    for i, (region, v) in enumerate(xaer_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='green', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='green', fontweight='bold')

    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(all_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "blue")
        x_labels[index] = region.name
    bar.set_label("ALL")
    for i, (region, v) in enumerate(all_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')

    xghg_vals_labels = match_sorting(all_vals_labels, get_values(xghg_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xghg_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "red")
        x_labels[index] = region.name
    for i, (region, v) in enumerate(xghg_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='red', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='red', fontweight='bold')
    bar.set_label("XGHG")

    ax.set_yticks(np.linspace(0, (bar_height + 0.9)*(len(all_vals_labels)), len(all_vals_labels)))
    ax.set_yticklabels(x_labels)
    ax.set_xlabel(f'{variable}')
    ax.legend()

    f.tight_layout()
    f.savefig(f"{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-AR6-{variable}-{max_min}-{time_begin}-{time_end}-bar-stacked.png")
    

def gen_stacking_country_bar_graphs(exp_num: str, variable: str, time_begin: str, time_end: str, label="": str, max_min="max": str):
    f, ax = plt.subplots(1, 1, figsize=(40, 110), facecolor='w')
    f.suptitle(f"{variable} Exp. {exp_num} Geopolitical Regional Average {label} from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)


    datasets = paths.get_paths_heat_output_avg(tmp_type=max_min, exp_num=exp_num)
    all_data = xarray.open_dataset(datasets[0])[f"HWF_t{max_min[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xghg_data = xarray.open_dataset(datasets[1])[f"HWF_t{max_min[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xaer_data = xarray.open_dataset(datasets[2])[f"HWF_t{max_min[2]}90pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    
    def get_values(dataset):
        ar6 = regionmask.defined_regions.natural_earth.countries_110
        vals_labels = []
        for region in ar6:
            value = dataset.where(ar6.mask(dataset) == region.number).mean(dim="lat", skipna=True).mean(dim="lon", skipna=True)
            if not np.isnan(value):
                vals_labels.append((region, value))
        return vals_labels

    def match_sorting(vals_labels_template, vals_labels_out):
        vals_labels_ret = []
        for region_temp, val_temp in vals_labels_template:
            for region_out, val_out in vals_labels_out:
                if region_out.name == region_temp.name:
                    vals_labels_ret.append((region_out, val_out))
        return vals_labels_ret

    bar_height = 4
    all_vals_labels = get_values(all_data)
    all_vals_labels.sort(key = lambda x: x[1])

    xaer_vals_labels = match_sorting(all_vals_labels, get_values(xaer_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xaer_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "green", rasterized=True)
        x_labels[index] = region.name
    bar.set_label("XAER")
    for i, (region, v) in enumerate(xaer_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='green', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='green', fontweight='bold')

    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(all_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "blue", rasterized=True)
        x_labels[index] = region.name
    bar.set_label("ALL")
    for i, (region, v) in enumerate(all_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='blue', fontweight='bold')

    xghg_vals_labels = match_sorting(all_vals_labels, get_values(xghg_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xghg_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "red", rasterized=True)
        x_labels[index] = region.name
    for i, (region, v) in enumerate(xghg_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='red', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, int(np.floor(v.values)), color='red', fontweight='bold')
    bar.set_label("XGHG")

    ax.set_yticks(np.linspace(0, (bar_height + 0.97)*(len(all_vals_labels)), len(all_vals_labels)))
    ax.set_yticklabels(x_labels)
    ax.set_xlabel(f'{variable}')
    ax.legend()

    f.tight_layout()
    f.savefig(f"{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-Countries-{variable}-{max_min}-{time_begin}-{time_end}-bar-stacked.png"