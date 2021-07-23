import paths
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from paths import FIGURE_IMAGE_OUTPUT, get_paths_heat_output_concat, OUT_ALL_AVGS_1980_2000, MERRA2_DATA
import cartopy.crs as ccrs
import xesmf as xe
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
                ax.text(v + 0.1, i*(bar_height + 1) - 1.5, np.round((v.values), 2), color='blue', fontweight='bold')
            else:
                ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round((v.values), 2), color='blue', fontweight='bold')
        ax.set_yticks(np.linspace(0, (bar_height + 0.9)*(len(vals_labels)), len(vals_labels)))
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(f'{variable}')
        ax.grid()
        f.tight_layout()

        pdf.savefig(f)

    pdf_pages = PdfPages(f'{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-{variable}-AR6-bar.pdf')
    datasets = paths.get_paths_heat_output_avg(tmp_type="max", exp_num=exp_num) + paths.get_paths_heat_output_avg(tmp_type="min", exp_num=exp_num)

    all_max = xarray.open_dataset(datasets[0]).HWF_tx9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    gen_regional_bar_graph(pdf_pages, f"ALL Max. from {time_begin} to {time_end}", all_max, variable, exp_num)
    xghg_max = xarray.open_dataset(datasets[1]).HWF_tx9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    ghg_max = all_max - xghg_max
    gen_regional_bar_graph(pdf_pages, f"GHG Max. from {time_begin} to {time_end}", ghg_max, variable, exp_num)
    xaer_max = xarray.open_dataset(datasets[2]).HWF_tx9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    aer_max = all_max - xaer_max
    gen_regional_bar_graph(pdf_pages, f"AER Max. from {time_begin} to {time_end}", aer_max, variable, exp_num)
    all_min = xarray.open_dataset(datasets[3]).HWF_tn9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    gen_regional_bar_graph(pdf_pages, f"ALL Max. from {time_begin} to {time_end}", all_min, variable, exp_num)
    xghg_min = xarray.open_dataset(datasets[4]).HWF_tn9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    ghg_min = all_min - xghg_min
    gen_regional_bar_graph(pdf_pages, f"GHG Max. from {time_begin} to {time_end}", ghg_min, variable, exp_num)
    xaer_min = xarray.open_dataset(datasets[5]).HWF_tn9pct.sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    aer_min = all_min - xaer_min
    gen_regional_bar_graph(pdf_pages, f"AER Max. from {time_begin} to {time_end}", aer_min, variable, exp_num)
    pdf_pages.close()


def gen_stacking_regional_bar_graphs(exp_num: str, variable: str, time_begin: str, time_end: str, label="", max_min="max", pdf=None):
    f, ax = plt.subplots(1, 1, figsize=(35, 27), facecolor='w')
    f.suptitle(f"{variable} Exp. {exp_num} AR6 Regional Average {max_min}{label} from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)

    datasets = paths.get_paths_heat_output_avg(tmp_type=max_min, exp_num=exp_num)
    
    if variable == "AHW2F/AHWF":
        all_data = xarray.open_dataset(datasets[0])[f"AHW2F_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        divisor = xarray.open_dataset(datasets[0])[f"AHWF_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        all_data = all_data.where(divisor != 0) / divisor
        
        xghg_data = xarray.open_dataset(datasets[1])[f"AHW2F_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        divisor = xarray.open_dataset(datasets[1])[f"AHWF_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        xghg_data = xghg_data.where(divisor != 0) / divisor
        
        xaer_data = xarray.open_dataset(datasets[2])[f"AHW2F_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        divisor = xarray.open_dataset(datasets[2])[f"AHWF_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        xaer_data = xaer_data.where(divisor != 0) / divisor
    else:
        all_data = xarray.open_dataset(datasets[0])[f"{variable}_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        xghg_data = xarray.open_dataset(datasets[1])[f"{variable}_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
        xaer_data = xarray.open_dataset(datasets[2])[f"{variable}_t{max_min[2]}9pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days

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
            adjustment = 0
            if 0 < v - all_vals_labels[i][1] < 3:
                adjustment = 3
            ax.text(v + 0.1 + adjustment, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='green', fontweight='bold',
                   bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='green', fontweight='bold')

    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(all_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "blue")
        x_labels[index] = region.name
    bar.set_label("ALL")
    for i, (region, v) in enumerate(all_vals_labels):
        if v >= 0:
            adjustment = 0
            if 0 < v - xaer_vals_labels[i][1] < 3:
                adjustment = 3
            if 0 > v - xaer_vals_labels[i][1] > 3:
                adjustment = -3
            ax.text(v + 0.1 + adjustment, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='blue', fontweight='bold',
                   bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='blue', fontweight='bold')

    xghg_vals_labels = match_sorting(all_vals_labels, get_values(xghg_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xghg_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "red")
        x_labels[index] = region.name
    for i, (region, v) in enumerate(xghg_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='red', fontweight='bold',
                   bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0})
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='red', fontweight='bold')
    bar.set_label("XGHG")

    ax.set_yticks(np.linspace(0, (bar_height + 0.9)*(len(all_vals_labels)), len(all_vals_labels)))
    ax.set_yticklabels(x_labels)
    ax.set_xlabel(f'{variable}')
    ax.legend()

    f.tight_layout()
    if pdf is None:
        variable = variable.replace("/", "-")
        f.savefig(f"{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-AR6-{variable}-{max_min}-{time_begin}-{time_end}-bar-stacked.png")
    else:
        pdf.savefig(f)
    

def gen_stacking_country_bar_graphs(exp_num: str, variable: str, time_begin: str, time_end: str, label="", max_min="max", pdf=None, ninety="90"):
    f, ax = plt.subplots(1, 1, figsize=(40, 110), facecolor='w')
    f.suptitle(f"{variable} Exp. {exp_num} Geopolitical Regional Average {label} from {time_begin} to {time_end}", fontsize=50)
    font = {'weight': 'bold',
            'size': 30}
    rc('font', **font)


    datasets = paths.get_paths_heat_output_avg(tmp_type=max_min, exp_num=exp_num)
    all_data = xarray.open_dataset(datasets[0])[f"HWF_t{max_min[2]}{ninety}pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xghg_data = xarray.open_dataset(datasets[1])[f"HWF_t{max_min[2]}{ninety}pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    xaer_data = xarray.open_dataset(datasets[2])[f"HWF_t{max_min[2]}{ninety}pct"].sel(time=(time_begin, time_end)).mean(dim="time").dt.days
    
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
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='green', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='green', fontweight='bold')

    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(all_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "blue", rasterized=True)
        x_labels[index] = region.name
    bar.set_label("ALL")
    for i, (region, v) in enumerate(all_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='blue', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='blue', fontweight='bold')

    xghg_vals_labels = match_sorting(all_vals_labels, get_values(xghg_data))
    x_labels = [""]*len(all_vals_labels)
    for index, (region, val) in enumerate(xghg_vals_labels):
        bar = ax.barh(index*(bar_height+1), val, height=bar_height, align='center', color = "red", rasterized=True)
        x_labels[index] = region.name
    for i, (region, v) in enumerate(xghg_vals_labels):
        if v >= 0:
            ax.text(v + 0.1, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='red', fontweight='bold')
        else:
            ax.text(v - 0.7, i*(bar_height + 1) - 1.5, np.round(v.values, 2), color='red', fontweight='bold')
    bar.set_label("XGHG")

    ax.set_yticks(np.linspace(0, (bar_height + 0.97)*(len(all_vals_labels)), len(all_vals_labels)))
    ax.set_yticklabels(x_labels)
    ax.set_xlabel(f'{variable}')
    ax.legend()

    f.tight_layout()
    if pdf is None:
        f.savefig(f"{FIGURE_IMAGE_OUTPUT}exp-{exp_num}-Countries-{variable}-{max_min}-{time_begin}-{time_end}-bar-stacked.png")
    else:
        pdf.savefig(f)
                  
                  
def gen_aldente_spaghetti_differences(variable: str, exp: str, ninety="90", pdf=None) -> None:
    max_all_datasets, max_xghg_datasets, max_xaer_datasets, \
    min_all_datasets, min_xghg_datasets, min_xaer_datasets = get_paths_heat_output_concat(exp)

    print(f"{len(max_xaer_datasets)} {len(min_xaer_datasets)} \n {len(min_xghg_datasets)} "
          f"{len(max_xghg_datasets)} \n {len(min_all_datasets)} {len(max_all_datasets)}")

    datasets_labels = [(max_xaer_datasets, "max. AER"), (min_xaer_datasets, "min. AER"),
                       (max_xghg_datasets, "max. GHG"), (min_xghg_datasets, "min. GHG")]

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)
    f, (ax_max, ax_min) = plt.subplots(2, 1, figsize=(15, 13))
    f.suptitle(f"Isolated Signal Time Series Data",
               fontsize=26)

    all_max_avg = None
    for ds_name in max_all_datasets:
        data = xarray.open_dataset(ds_name)
        if variable == "AHW2F/AHWF":
            data = data[f"AHW2F_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
            divisor = xarray.open_dataset(ds_name)[f"AHWF_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
            data = data.where(divisor != 0, drop=True) / divisor
        else:
            data = data[f"{variable}_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
        data.plot(ax=ax_max, color="red", alpha=0.2)
        if all_max_avg is None:
            all_max_avg = data
        else:
            all_max_avg += data
    all_max_avg = all_max_avg / len(max_all_datasets)
    all_max_avg.plot(ax=ax_max, color="red", label="max. ALL")

    all_min_avg = None
    for ds_name in min_all_datasets:
        data = xarray.open_dataset(ds_name)
        if variable == "AHW2F/AHWF":
            data = data[f"AHW2F_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
            divisor = xarray.open_dataset(ds_name)[f"AHWF_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
            data = data.where(divisor != 0, drop=True) / divisor
        else:
            data = data[f"{variable}_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
        data.plot(ax=ax_min, color="red", alpha=0.2)
        if all_min_avg is None:
            all_min_avg = data
        else:
            all_min_avg += data
    all_min_avg = all_min_avg / len(min_all_datasets)
    all_min_avg.plot(ax=ax_min, color="red", label="min. ALL")

    for ds_names, label in datasets_labels:
        max_avg = None
        min_avg = None
        color = "red"
        if "GHG" in label:
            color = "green"
        elif "AER" in label:
            color = "blue"
        for ds_name in ds_names:
            data = xarray.open_dataset(ds_name)
            if "max" in label:
                if variable == "AHW2F/AHWF":
                    divisor = data[f"AHWF_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
                    data = all_max_avg - (data[f"AHW2F_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days.where(divisor != 0, drop=True) / divisor)
                else:
                    data = all_max_avg - data[f"{variable}_tx{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
                if max_avg is None:
                    max_avg = data
                else:
                    max_avg += data
                data.plot(ax=ax_max, color=color, alpha=0.2)
            else:
                if variable == "AHW2F/AHWF":
                    divisor = data[f"AHWF_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
                    data = all_max_avg - (data[f"AHW2F_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days / divisor)
                else:
                    data = all_max_avg - data[f"{variable}_tn{ninety}pct"].mean(dim="lat").mean(dim="lon").dt.days
                if min_avg is None:
                    min_avg = data
                else:
                    min_avg += data
                data.plot(ax=ax_min, color=color, alpha=0.2)

        if max_avg is not None:
            max_avg = max_avg / len(ds_names)
            max_avg.plot(ax=ax_max, color=color, label=label)
        if min_avg is not None:
            min_avg = min_avg / len(ds_names)
            min_avg.plot(ax=ax_min, color=color, label=label)
    ax_max.set_title(f"{variable} MAX. EXP {exp}")
    ax_max.legend(loc="upper left")
    ax_min.set_title(f"{variable} MIN. EXP {exp}")
    ax_min.legend(loc="upper left")

    text = f"The number of heat wave days each year, thick line represents the mean value of all the ensemble members."\
           f" GHG and AER were calculated from ALL-XGHG and ALL-XAER respectively. "

    f.text(0.5, 0.005, text, wrap=True, horizontalalignment='center', fontsize=12)
    f.tight_layout()
    if pdf is None:
        variable = variable.replace("/", "-")
        f.savefig(f"{FIGURE_IMAGE_OUTPUT}/exp-{exp}-{variable}-aldente-spaghetti-differences.png")
    else:
        pdf.savefig(f)
                  
                  
def gen_merra2_comparison_maps(variable: str, exp_num: str, pdf=None, ninety="90", v1mx=None, v1mn=None, v2mx=None, v2mn=None):
    merra_min_path = f"tn90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
    merra_max_path = f"tx90pct_heatwaves_MERRA2_rNone_{exp_num}_yearly_summer.nc"
    if variable == "AHW2F/AHWF":
        merra2_min = xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"AHW2F_tn9pct"].mean(dim="time").dt.days
        merra2_min = merra2_min / xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"AHWF_tn9pct"].mean(dim="time").dt.days
        merra2_max = xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"AHW2F_tx9pct"].mean(dim="time").dt.days
        merra2_max = merra2_max / xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"AHWF_tx9pct"].mean(dim="time").dt.days
        
        ensemble_min_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"AHW2F_tn90pct"].sel(time=("1980", "2010")).dt.days
        ensemble_min_avg = ensemble_min_avg / xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"AHWF_tn90pct"].sel(time=("1980", "2010")).dt.days
        ensemble_max_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"AHW2F_tx90pct"].sel(time=("1980", "2010")).dt.days
        ensemble_max_avg = ensemble_max_avg / xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"AHWF_tx90pct"].sel(time=("1980", "2010")).dt.days
        ensemble_min_avg = ensemble_min_avg.mean(dim="time").fillna(0)*100
        ensemble_max_avg = ensemble_max_avg.mean(dim="time").fillna(0)*100
        print(ensemble_max_avg)
    else:
        merra2_min = xarray.open_dataset(MERRA2_DATA + merra_min_path)[f"{variable}_tn9pct"].mean(dim="time").dt.days
        merra2_max = xarray.open_dataset(MERRA2_DATA + merra_max_path)[f"{variable}_tx9pct"].mean(dim="time").dt.days
        
        ensemble_min_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-min-{exp_num}.nc")[f"{variable}_tn90pct"].sel(time=("1980", "2010"))
        ensemble_max_avg = xarray.open_dataset(OUT_ALL_AVGS_1980_2000 + f"ALL-max-{exp_num}.nc")[f"{variable}_tx90pct"].sel(time=("1980", "2010"))
        
        ensemble_min_avg = ensemble_min_avg.mean(dim="time").dt.days
        ensemble_max_avg = ensemble_max_avg.mean(dim="time").dt.days

    min_regridder = xe.Regridder(merra2_min, ensemble_min_avg, 'bilinear')
    max_regridder = xe.Regridder(merra2_max, ensemble_max_avg, 'bilinear')

    merra2_min = min_regridder(merra2_min)
    merra2_max = max_regridder(merra2_max)
    
    f, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(35, 13), facecolor='w', subplot_kw=dict(projection=ccrs.PlateCarree()))
    f.suptitle(f"{variable} Exp. {exp_num} MERRA2 Comparison (1980-2010 Avg)", fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)

    if v1mx is not None and v1mn is not None:
        vmax = v1mx
        vmin = v1mn
    else:
        vmin = 0
        vmax = 20
    
    ensemble_min_avg.plot(ax=ax1, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax1.set_title(f"ALL {variable} Min. Top 90 Perc.")
    ax1.coastlines()

    merra2_min.plot(ax=ax2, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax2.set_title(f"MERRA2 {variable} Min. Top 90 Perc.")
    ax2.coastlines()

    ensemble_max_avg.plot(ax=ax3, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax3.set_title(f"ALL {variable} Max. Top 90 Perc.")
    ax3.coastlines()

    merra2_max.plot(ax=ax4, cmap='Reds', vmax=vmax, vmin=vmin, rasterized=True)
    ax4.set_title(f"MERRA2 {variable} Max. Top 90 Perc.")
    ax4.coastlines()

    if v2mx is not None and v2mn is not None:
        vmax = v2mx
        vmin = v2mn
    else:
        vmin = -200
        vmax = 200
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    max_compare = ((ensemble_max_avg - merra2_max) / merra2_max) * 100
    max_compare = max_compare.rename("Perc. Change %")
    max_compare.where(max_compare < vmax).plot(ax=ax5, cmap='seismic', vmax=vmax, vmin=vmin, norm=norm, rasterized=True)
    ax5.set_title(f"(Model - MERRA2) / MERRA2 {variable} Max. Top 90 Perc.")
    ax5.coastlines()

    min_compare = ((ensemble_min_avg - merra2_min) / merra2_min) * 100
    min_compare = min_compare.rename("Perc. Change %")
    min_compare.where(min_compare < vmax).plot(ax=ax6, cmap='seismic', vmax=vmax, vmin=vmin, norm=norm, rasterized=True)
    ax6.set_title(f"(Model - MERRA2) / MERRA2 {variable} Min. Top 90 Perc.")
    ax6.coastlines()

    f.tight_layout()
    if pdf is None:
        variable = variable.replace("/", "-")
        f.savefig(f"{FIGURE_IMAGE_OUTPUT}/exp-{exp_num}-{variable}-MERRA2-comparison.png")
    else:
        pdf.savefig(f)