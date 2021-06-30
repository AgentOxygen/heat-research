from settings import POST_OUT_EM_AVGS_1920_1950, FIGURE_IMAGE_OUTPUT
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors
from matplotlib import rc


def discover(exp_num: str, variable: str, output_dir: str) -> None:
    all_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}ALL-max-{exp_num}.nc")[variable + "_tx9pct"].dt.days
    xaer_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XAER-max-{exp_num}.nc")[variable + "_tx9pct"].dt.days
    xghg_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XGHG-max-{exp_num}.nc")[variable + "_tx9pct"].dt.days
    aer_max_data = all_max_data - xaer_max_data
    ghg_max_data = all_max_data - xghg_max_data
    years = all_max_data.time.values

    all_max_volitility = all_max_data.sel(time=years[0])*0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            all_max_volitility += all_max_data.sel(time=year) - all_max_data.sel(time=years[index - 1])
    aer_max_volitility = aer_max_data.sel(time=years[0]) * 0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            aer_max_volitility += aer_max_data.sel(time=year) - aer_max_data.sel(time=years[index - 1])
    ghg_max_volitility = ghg_max_data.sel(time=years[0]) * 0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            ghg_max_volitility += ghg_max_data.sel(time=year) - ghg_max_data.sel(time=years[index - 1])

    all_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}ALL-min-{exp_num}.nc")[variable + "_tn9pct"].dt.days
    xaer_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XAER-min-{exp_num}.nc")[variable + "_tn9pct"].dt.days
    xghg_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XGHG-min-{exp_num}.nc")[variable + "_tn9pct"].dt.days
    aer_min_data = all_min_data - xaer_min_data
    ghg_min_data = all_min_data - xghg_min_data

    all_min_volitility = all_min_data.sel(time=years[0]) * 0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            all_min_volitility += all_min_data.sel(time=year) - all_min_data.sel(time=years[index - 1])
    aer_min_volitility = aer_min_data.sel(time=years[0]) * 0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            aer_min_volitility += aer_min_data.sel(time=year) - aer_min_data.sel(time=years[index - 1])
    ghg_min_volitility = ghg_min_data.sel(time=years[0]) * 0
    for index, year in enumerate(years):
        if index > 0 and year != 2005 and year != 2006 and year != 2080:
            ghg_min_volitility += ghg_min_data.sel(time=year) - ghg_min_data.sel(time=years[index - 1])

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13))
    f.suptitle(f"{variable} EXP:{exp_num} Abs. Shifts (1920-2080) (2005, 2006, 2080 excluded)", fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)

    vmin = 0
    vmax = 160

    ax1.set_title("ALL Max")
    fig_map = ax1.pcolor(all_max_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax1, format='%.0f')
    ax2.set_title("AER Max")
    fig_map = ax2.pcolor(aer_max_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax2, format='%.0f')
    ax3.set_title("GHG Max")
    fig_map = ax3.pcolor(ghg_max_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax3, format='%.0f')

    ax4.set_title("ALL Min")
    fig_map = ax4.pcolor(all_min_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax4, format='%.0f')
    ax5.set_title("AER Min")
    fig_map = ax5.pcolor(aer_min_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax5, format='%.0f')
    ax6.set_title("GHG Min")
    fig_map = ax6.pcolor(ghg_min_volitility, cmap='gist_heat', vmax=vmax, vmin=vmin, rasterized=True)
    f.colorbar(fig_map, ax=ax6, format='%.0f')

    f.tight_layout()
    f.savefig(f"{output_dir}exp-{exp_num}-{variable}-abs_shifts.png")


def plot_discover(exp_num: str, variable: str, base_begin: float, base_end: float, output_dir: str) -> None:
    all_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}ALL-max-{exp_num}.nc")[variable + "_tx9pct"]
    xaer_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XAER-max-{exp_num}.nc")[variable + "_tx9pct"]
    xghg_max_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XGHG-max-{exp_num}.nc")[variable + "_tx9pct"]
    aer_max_data = all_max_data - xaer_max_data
    ghg_max_data = all_max_data - xghg_max_data

    base_all_min_data = all_max_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days
    base_aer_max_data = aer_max_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days
    base_ghg_max_data = ghg_max_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days

    all_max_data = all_max_data.mean(dim="lon")
    aer_max_data = aer_max_data.mean(dim="lon")
    ghg_max_data = ghg_max_data.mean(dim="lon")

    all_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}ALL-min-{exp_num}.nc")[variable + "_tn9pct"]
    xaer_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XAER-min-{exp_num}.nc")[variable + "_tn9pct"]
    xghg_min_data = xarray.open_dataset(f"{POST_OUT_EM_AVGS_1920_1950}XGHG-min-{exp_num}.nc")[variable + "_tn9pct"]
    aer_min_data = all_min_data - xaer_min_data
    ghg_min_data = all_min_data - xghg_min_data

    base_all_max_data = all_min_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days
    base_aer_min_data = aer_min_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days
    base_ghg_min_data = ghg_min_data.sel(time=(base_begin, base_end)).mean(dim="time").mean(dim="lon").dt.days

    all_min_data = all_min_data.mean(dim="lon")
    aer_min_data = aer_min_data.mean(dim="lon")
    ghg_min_data = ghg_min_data.mean(dim="lon")

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(35, 13))
    f.suptitle(f"{variable} EXP:{exp_num} Shifts Based on {base_begin}-{base_end} rel. to Latitude", fontsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    rc('font', **font)

    ax1.set_title("ALL Max")
    ax1.plot(all_max_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_all_max_data, label="1960-1990", color="r")
    ax1.plot(all_max_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_all_max_data, label="1990-2020", color="b")
    ax1.plot(all_max_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_all_max_data, label="2020-2050", color="g")
    ax1.plot(all_max_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_all_max_data, label="2050-2080", color="y")
    ax1.legend(loc="upper left")

    ax2.set_title("AER Max")
    ax2.plot(aer_max_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_aer_max_data, label="1960-1990", color="r")
    ax2.plot(aer_max_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_aer_max_data, label="1990-2020", color="b")
    ax2.plot(aer_max_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_aer_max_data, label="2020-2050", color="g")
    ax2.plot(aer_max_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_aer_max_data, label="2050-2080", color="y")
    ax2.legend(loc="upper left")

    ax3.set_title("GHG Max")
    ax3.plot(ghg_max_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_ghg_max_data, label="1960-1990", color="r")
    ax3.plot(ghg_max_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_ghg_max_data, label="1990-2020", color="b")
    ax3.plot(ghg_max_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_ghg_max_data, label="2020-2050", color="g")
    ax3.plot(ghg_max_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_ghg_max_data, label="2050-2080", color="y")
    ax3.legend(loc="upper left")

    ax4.set_title("ALL Min")
    ax4.plot(all_min_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_all_min_data, label="1960-1990", color="r")
    ax4.plot(all_min_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_all_min_data, label="1990-2020", color="b")
    ax4.plot(all_min_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_all_min_data, label="2020-2050", color="g")
    ax4.plot(all_min_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_all_min_data, label="2050-2080", color="y")
    ax4.legend(loc="upper left")

    ax5.set_title("AER Min")
    ax5.plot(aer_min_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_aer_min_data, label="1960-1990", color="r")
    ax5.plot(aer_min_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_aer_min_data, label="1990-2020", color="b")
    ax5.plot(aer_min_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_aer_min_data, label="2020-2050", color="g")
    ax5.plot(aer_min_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_aer_min_data, label="2050-2080", color="y")
    ax5.legend(loc="upper left")

    ax6.set_title("GHG Min")
    ax6.plot(ghg_min_data.sel(time=(1960, 1990)).mean(dim="time").dt.days - base_ghg_min_data, label="1960-1990", color="r")
    ax6.plot(ghg_min_data.sel(time=(1990, 2020)).mean(dim="time").dt.days - base_ghg_min_data, label="1990-2020", color="b")
    ax6.plot(ghg_min_data.sel(time=(2020, 2050)).mean(dim="time").dt.days - base_ghg_min_data, label="2020-2050", color="g")
    ax6.plot(ghg_min_data.sel(time=(2050, 2080)).mean(dim="time").dt.days - base_ghg_min_data, label="2050-2080", color="y")
    ax6.legend(loc="upper left")

    f.tight_layout()
    f.savefig(f"{output_dir}exp-{exp_num}-{variable}-latitude-plots.png")


plot_discover("3114", "HWF", 1920, 1950, FIGURE_IMAGE_OUTPUT)
