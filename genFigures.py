# import generate as gen
# f = gen.gen_weighted_v_unweighted("HWD", "3136")

# f.savefig("auto.png")

from paths import DIR_PATH, trefhtmx_members as tmax_paths, trefhtmn_members as tmin_paths
import xarray
import matplotlib.pyplot as plt
from matplotlib import rc

all_max = xarray.open_mfdataset([path for path in tmax_paths()[0]], concat_dim="member", combine="nested")
all_min = xarray.open_mfdataset([path for path in tmin_paths()[0]], concat_dim="member", combine="nested")

f, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor='w')
rc('font', **{'weight': 'bold', 'size': 24})
f.suptitle(f"TREFHT Max. vs Min.", fontsize=30)

all_max.TREFHTMX.mean(dim="lat").mean(dim="lon").mean(dim="member").plot(ax=ax, color="red", label="Max. Temp.")
all_min.TREFHTMN.mean(dim="lat").mean(dim="lon").mean(dim="member").plot(ax=ax, color="blue", label="Min. Temp.")
ax.legend()

f.tight_layout()
f.savefig("out.png")