#!/bin/sh
CHANGE_DIR="/projects/dgs/persad_research/heat_research/postprocessing/1920to1950_ensemble_members/XGHG-"
MIN_PATH="/projects/dgs/persad_research/heat_research/data/b.e11.B20TRLENS_RCP85.f09_g16.xghg.015.cam.h1.TREFHTMN.19200101-20051231.nc"
MAX_PATH="/projects/dgs/persad_research/heat_research/data/b.e11.B20TRLENS_RCP85.f09_g16.xghg.015.cam.h1.TREFHTMX.19200101-20051231.nc"
VAR_MAX="TREFHTMX"
VAR_MIN="TREFHTMN"
python3 ehfheatwaves_threshold.py -x $MAX_PATH -n $MIN_PATH --change_dir $CHANGE_DIR --t90pc --base=1921-1950 -d CESM2 -p 90 --vnamex $VAR_MAX --vnamen $VAR_MIN 
