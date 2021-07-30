import sys
import pandas as pd
import numpy as np
import datetime as dt
import xarray
import math
from uuid import uuid4
import netcdftime
from multiprocessing import Process, Queue
from os.path import isdir
from os import listdir, remove, mkdir
from netCDF4 import MFDataset, MFTime, Dataset, num2date
from optparse import OptionParser
from multiprocessing import Process
import threading
import cProfile, pstats, io
from pstats import SortKey
import xesmf as xe


def bilinear_interpolation(data_to_interpolate: xarray.DataArray, data_to_match: xarray.DataArray) -> xarray.DataArray:
    regridder = xe.Regridder(data_to_interpolate, data_to_match, 'bilinear')
    return regridder(data_to_interpolate)


def calculate_thresholds(output_path: str, tmaxfile: str, tmaxvname: str, tminfile: str, tminvname: str, bpfx: str=None, bpfn: str=None, maskfile: str=None, maskvname: str="sftlf", timevname: str="time", season: str="summer", pcntl: float=0.9, bp: str="1920-1950", dailyout: bool=True, dailyonly: bool=True, t90pc: bool=True, control_data: bool=False) -> bool:
    """
    tmaxfile -> file containing tmax
    tmaxvname -> tmax variable name
    tminfile -> file containing tmin
    tminvname -> tmin variable name
    bpfx -> Indicates a future simulation, specifying a tmax file containing the historical base period to be used
    bpfn -> Indicates a future simulation, specifying a tmin file containing the historical base period to be used
    maskfile -> file containing land-sea mask
    maskvname -> mask variable name
    timevname -> time variable name
    season -> austal season for annual metrics. Defaults to austral summer
    pcntl -> the percentile to use for thresholds (0 - 1). Defaults to 0.9
    bp -> base period to calculate thresholds. Default 1920-1950
    daily -> output daily EHF values and heatwave indicators. Default True
    dailyonly -> output only daily values and suppress yearly output. Default True
    t90pc -> Calculate tx90pc and tn90pc heatwaves
    output_path -> File path to output threshold to
    control_data -> modifies  for calculating thresholds for control data. Default is False
    """
    
    if not tmaxfile or not tminfile:
        print ("Please specify tmax and tmin files.")
        return False
    if not maskfile:
        print ("You didn't specify a land-sea mask. It's faster if you do,"
            "so this might take a while.")
    if len(bp.split("-"))!=2:
        print ("Incorrect base period format.")
        return False
    else:
        bpstart = int(bp.split("-")[0])
        bpend = int(bp.split("-")[1])
        print(f"{bpstart}-{bpend}")
    if (season!='summer')&(season!='winter'):
        print ("Use either summer or winter.")
        return False
    # save daily EHF output
    yearlyout = True
    if dailyonly: 
        dailyout = True
        yearlyout = False

    # Define period range function for noleap year
    def period_range(*args, without_leap=False, **kwargs):
        """Wraps period_range, removing leap days"""
        periods = pd.period_range(*args, **kwargs)
        if without_leap:
            is_leap_day = (periods.month == 2) & (periods.day == 29)
            periods = periods[~is_leap_day]
        return periods

    # Load time data
    try:
        tmaxnc = MFDataset(tmaxfile, 'r')
    except (IndexError, ValueError, OSError):
        tmaxnc = Dataset(tmaxfile, 'r')
    nctime = tmaxnc.variables[timevname]
    try:
        nctime = MFTime(nctime)
    except AttributeError:
        pass
    calendar = nctime.calendar
    if not calendar:
        print ('Unrecognized calendar. Using gregorian.')
        calendar = 'gregorian'
    elif calendar=='360_day':
        print("360 Day Calendar")
        daysinyear = 360
        # 360 day season start and end indices
        SHS = (301,451)
        SHW = (121,271)
        dayone = num2date(nctime[0], nctime.units,
                calendar=calendar)
        daylast = num2date(nctime[-1], nctime.units,
                calendar=calendar)
        class calendar360():
            def __init__(self,sdate,edate):
                self.year = np.repeat(range(sdate.year,edate.year+1), 360, 0)
                nyears = len(xrange(sdate.year,edate.year+1))
                self.month = np.tile(np.repeat(range(1,12+1), 30, 0), nyears)
                self.day = np.tile(np.tile(range(1,30+1), 12), nyears)
                if (sdate.day!=1)|(edate.month!=1):
                    sdoyi = (sdate.month-1)*30+sdate.day-1
                    self.year = self.year[sdoyi:]
                    self.month = self.month[sdoyi:]
                    self.day = self.day[sdoyi:]
                if (edate.day!=30)|(edate.month!=12):
                    edoyi = (12-edate.month)*30+(30-edate.day)
                    self.year = self.year[:-edoyi]
                    self.month = self.month[:-edoyi]
                    self.day = self.day[:-edoyi]
        dates = calendar360(dayone, daylast)
        shorten = 0
        if (daylast.day!=30)|(daylast.month!=12):
            shorten = 30*(13-daylast.month) - daylast.day
    else:
        print("365 Day Calendar")
        daysinyear = 365
        # 365 day season start and end indices
        SHS = (304,455)
        SHW = (121,274)
        if tmaxnc.variables[timevname].units=='day as %Y%m%d.%f':
            st = str(int(nctime[0]))
            nd = str(int(nctime[-1]))
            dayone = dt.datetime(int(st[:4]), int(st[4:6]), int(st[6:]))
            daylast = dt.datetime(int(nd[:4]), int(nd[4:6]), int(nd[6:]))
        else:
            dayone = num2date(nctime[0], nctime.units,
                    calendar=calendar)
            daylast = num2date(nctime[-1], nctime.units,
                    calendar=calendar)
    #####added to correct for noleap calendar
        if calendar == 'noleap':
            print("No-leap calendar")
            dates = period_range(str(dayone), str(daylast), without_leap = True)
        else:
            print("Leap calendar")
            dates = period_range(str(dayone), str(daylast), without_leap = False)
    ########
    #    dates = pd.period_range(str(dayone), str(daylast))
        shorten = 0
        if (daylast.day!=30)|(daylast.month!=12):
            endofdata = dt.datetime(2000, daylast.month, daylast.day)
            shorten = dt.datetime(2000, 12, 31) - endofdata
            shorten = shorten.days

    # Load land-sea mask
    if maskfile:
        masknc = Dataset(maskfile, 'r')
        vname = maskvname
        mask = masknc.variables[vname][:]
        if mask.max()>1: mask = mask>50
        mask = mask.astype(np.bool)
        masknc.close()

    # Load base period data
    if bpfn:
        try:
            tminnc = MFDataset(bpfn, 'r')
        except (IndexError, ValueError, OSError):
            tminnc = Dataset(bpfn, 'r')
    else:
        try:
            tminnc = MFDataset(tminfile, 'r')
        except (IndexError, ValueError, OSError):
            tminnc = Dataset(tminfile, 'r')
    if bpfx:
        try:
            tmaxnc = MFDataset(bpfx, 'r')
        except (IndexError, ValueError, OSError):
            tmaxnc = Dataset(bpfx, 'r')
    else:
        try:
            tmaxnc = MFDataset(tmaxfile, 'r')
        except (IndexError, ValueError, OSError):
            tmaxnc = Dataset(tmaxfile, 'r')
    vname = tmaxvname
    bptime = tmaxnc.variables[timevname]
    try:
        bptime = MFTime(bptime)
    except AttributeError:
        pass
    bpdayone = num2date(bptime[0], bptime.units, calendar=calendar)
    bpdaylast = num2date(bptime[-1], bptime.units, calendar=calendar)
    #Added to correct for noleap calendar#########
    if calendar == 'noleap':
        print("Period Range without leap")
        bpdates = period_range(str(bpdayone), str(bpdaylast), without_leap = True)
    else:
        print("Period Range with leap")
        bpdates = period_range(str(bpdayone), str(bpdaylast), without_leap = False)
    ###########
    dates_base = bpdates[(bpstart<=bpdates.year)&(bpdates.year<=bpend)]
    print(f"Base: {dates_base}")


    tmax = tmaxnc.variables[vname][(bpstart<=dates.year)&(dates.year<=(bpend))]
    if len(tmax.shape)==4: tmax = tmax.squeeze()
    original_shape = tmax.shape
    if maskfile:
        mask=np.squeeze(mask) #added by Jane on 12/12/15
        tmax = tmax[:,mask]
    if tmaxnc.variables[vname].units=='K': tmax -= 273.15
    vname = tminvname
    tmin = tminnc.variables[vname][(bpstart<=bpdates.year)&(bpdates.year<=bpend)]
    if len(tmin.shape)==4: tmin = tmin.squeeze()
    if maskfile:
        tmin = tmin[:,mask]
    if tminnc.variables[vname].units=='K': tmin -= 273.15
    tave_base = (tmax + tmin)/2.

    # Remove leap days in gregorian calendars
    if (calendar=='gregorian')|(calendar=='proleptic_gregorian')|(calendar=='standard')|(calendar=='julian'):
        tave_base = tave_base[(dates_base.month!=2)|(dates_base.day!=29),...]
        tmax = tmax[(dates_base.month!=2)|(dates_base.day!=29),...]
        tmin = tmin[(dates_base.month!=2)|(dates_base.day!=29),...]
        del dates_base
        
#     # Remove incomplete starting year
    first_year = dayone.year
    if (dayone.month!=1)|(dayone.day!=1) and not control_data:
        first_year = dayone.year+1
        start = np.argmax(dates.year==first_year)
        #tave = tave[start:,...]
        tmax = tmax[start:,...]
        tmin = tmin[start:,...]
    
    print("Calculating 90th percentile...")
    # Caclulate 90th percentile
    tpct = np.ones(((daysinyear,)+tave_base.shape[1:]))*np.nan
    txpct = tpct.copy()
    tnpct = tpct.copy()
    window = np.zeros(daysinyear,dtype=np.bool)
    np.shape(window)
    wsize = 15.
    window[int(-np.floor(wsize/2.)):] = 1
    window[:int(np.ceil(wsize/2.))] = 1
    window = np.tile(window,bpend+1-bpstart)
    
    parameter = False
    for day in range(daysinyear):
        txpct[day,...] = np.quantile(tmax[window,...], pcntl, axis=0)
        #percentile(tmax[window,...], pcntl, parameter)
        tnpct[day,...] = np.quantile(tmin[window,...], pcntl, axis=0)
        #percentile(tmin[window,...], pcntl, parameter)
        window = np.roll(window,1)
    del tave_base
    del window

    # Save to netCDF
    try:
        experiment = tmaxnc.__getattribute__('experiment')
        model = tmaxnc.__getattribute__('model_id')
        realization = tmaxnc.__getattribute__('parent_experiment_rip')
        if realization==u'N/A': realization = u'r1i1p1'
    except AttributeError:
        experiment = ''
        model = ''
        realization = ''
    try:
        space = (tmaxnc.dimensions['lat'].__len__(),tmaxnc.dimensions['lon'].__len__())
        lonname = 'lon'
        latname = 'lat'
    except KeyError:
        lonname = 'longitude'
        latname = 'latitude'
        space = (tmaxnc.dimensions['latitude'].__len__(),tmaxnc.dimensions['longitude'].__len__())
    
    def save_yearly(definition):
        print(f"Saving yearly EXP {definition}")
        yearlyout = Dataset(output_path, 'w')
        yearlyout.createDimension('time', len(range(first_year, daylast.year+1)))
        yearlyout.createDimension('lon', tmaxnc.dimensions[lonname].__len__())
        yearlyout.createDimension('lat', tmaxnc.dimensions[latname].__len__())
        yearlyout.createDimension('day', daysinyear)
        setattr(yearlyout, "author", "Tammas Loughran")
        setattr(yearlyout, "contact", "t.loughran@student.unsw.edu.au")
        setattr(yearlyout, "source", "https://github.com/tammasloughran/ehfheatwaves")
        setattr(yearlyout, "date", dt.datetime.today().strftime('%Y-%m-%d'))
        setattr(yearlyout, "script", sys.argv[0])
        if model:
            setattr(yearlyout, "model_id", model)
            setattr(yearlyout, "experiment", experiment)
            setattr(yearlyout, "realization", "%s"%(realization))
        setattr(yearlyout, "period", "%s-%s"%(str(first_year),str(daylast.year)))
        setattr(yearlyout, "base_period", "%s-%s"%(str(bpstart),str(bpend)))
        setattr(yearlyout, "percentile", "%sth"%(str(pcntl)))
        setattr(yearlyout, "definition", definition)
        setattr(yearlyout, "frequency", "yearly")
        setattr(yearlyout, "season", season)
        setattr(yearlyout, "definition", definition)
        setattr(yearlyout, "season_note", ("The year of a season is the year it starts"
                "in. SH summer: Nov-Mar. NH summer: May-Sep."))
        try:
            file = open('version', 'r')
            commit = file.read()[:-2]
        except IOError:
            commit = "Unknown. Check date for latest version."
        setattr(yearlyout, "git_commit", commit)
        setattr(yearlyout, "tmax_file", tmaxfile)
        setattr(yearlyout, "tmin_file", tminfile)
        if maskfile:
            setattr(yearlyout, "mask_file", maskfile)
        setattr(yearlyout, "quantile_method", "numpy.quantile")
        otime = yearlyout.createVariable('time', 'f8', 'time', fill_value=-999.99)
        setattr(otime, 'units', 'year')
        olat = yearlyout.createVariable('lat', 'f8', 'lat')
        setattr(olat, 'standard_name', 'latitude')
        setattr(olat, 'long_name', 'Latitude')
        setattr(olat, 'units', 'degrees_north')
        setattr(olat, 'axis', 'Y')
        olon = yearlyout.createVariable('lon', 'f8', 'lon')
        setattr(olon, 'standard_name', 'longiitude')
        setattr(olon, 'long_name', 'Longitude')
        setattr(olon, 'units', 'degrees_east')
        setattr(olon, 'axis', 'X')
        otxpct = yearlyout.createVariable('txpct', 'f8', 
            ('day','lat','lon'), fill_value=-999.99)
        setattr(otxpct, 'long_name', 'tmax percentile')
        setattr(otxpct, 'units', 'degC')
        setattr(otxpct, 'description', 
                '90th percentile of %s-%s'%(str(bpstart),str(bpend)))
        otnpct = yearlyout.createVariable('tnpct', 'f8', 
            ('day','lat','lon'), fill_value=-999.99)
        setattr(otnpct, 'long_name', 'tmin percentile')
        setattr(otnpct, 'units', 'degC')
        setattr(otnpct, 'description', 
                '90th percentile of %s-%s'%(str(bpstart),str(bpend)))
        otime[:] = range(first_year, daylast.year+1)
        olat[:] = tmaxnc.variables[latname][:]
        olon[:] = tmaxnc.variables[lonname][:]
        dummy_array = np.ones((daysinyear,)+original_shape[1:])*np.nan
        if maskfile:
            dummy_array[:,mask] = txpct
            dummy_array[np.isnan(dummy_array)] = -999.99
            otxpct[:] = dummy_array.copy()
            dummy_array[:,mask] = tnpct
            dummy_array[np.isnan(dummy_array)] = -999.99
            otnpct[:] = dummy_array.copy()
        else:
            otxpct[:] = txpct
            otnpct[:] = tnpct
        yearlyout.close()

    save_yearly("txn90pct")
    return True


def calculate_heatwave_statistics(output_path: str, tmaxfile: str, tminfile: str, thresfile: str, exp_num: str, tmaxvname="TREFHTMX", tminvname="TREFHTMN", model="CESM1", ensrn=u'N/A', maskfile=None, maskvname="sftlf", timevname="time", season="summer", pcntl=0.9, bp="1920-1950", syn=None) -> bool:
    """
    output_path -> File path to output threshold to
    tmaxfile -> file containing tmax
    tminfile -> file containing tmin
    tmaxvname -> tmax variable name
    tminvname -> tmin variable name
    model -> name of model that supplies data (ie FLOR, NCEP, ERAInt). Default is AM4.
    ensrn -> ensemble realization number (ie 1-30).
    maskfile -> file containing land-sea mask
    maskvname -> mask variable name
    timevname -> time variable name
    season -> austal season for annual metrics. Defaults to austral summer
    pcntl -> the percentile to use for thresholds. Defaults to 0.9
    bp -> base period to calculate thresholds. Default 1920-1950
    thresfile -> file containing thresholds pre-calculated from base period
    syn -> file containing synthetic data to add to daily maximum and daily minimum (varnames=DMAX,DMIN)
    """
    
    elength, bmax, e2length, dmin = [int(num) for num in exp_num]
    if not tmaxfile or not tminfile:
        print ("Please specify tmax and tmin files.")
        return False
    if not maskfile:
        print ("You didn't specify a land-sea mask. It's faster if you do,"
            "so this might take a while.")
    if len(bp)!=9:
        print ("Incorrect base period format.")
        return False
    else:
        bpstart = int(bp[:4])
        bpend = int(bp[5:9])
    
    if (season!='summer')&(season!='winter'):
        print ("Use either summer or winter.")
        return False

    # Define period range function for noleap year
    def period_range(*args, without_leap=False, **kwargs):
        """Wraps period_range, removing leap days"""
        periods = pd.period_range(*args, **kwargs)
        if without_leap:
            is_leap_day = (periods.month == 2) & (periods.day == 29)
            periods = periods[~is_leap_day]
        return periods

    # Load time data
    try:
        tmaxnc = MFDataset(tmaxfile, 'r')
    except (IndexError, ValueError, OSError):
        tmaxnc = Dataset(tmaxfile, 'r')
    nctime = tmaxnc.variables[timevname]
    try:
        nctime = MFTime(nctime)
    except AttributeError:
        pass
    calendar = nctime.calendar
    if not calendar:
        print('Unrecognized calendar. Using gregorian.')
        calendar = 'gregorian'
    elif calendar=='360_day':
        daysinyear = 360
        # 360 day season start and end indices
        SHS = (301,451)
        SHW = (121,271)
        dayone = num2date(nctime[0], nctime.units,
                calendar=calendar)
        daylast = num2date(nctime[-1], nctime.units,
                calendar=calendar)
        class calendar360():
            def __init__(self,sdate,edate):
                self.year = np.repeat(range(sdate.year,edate.year+1), 360, 0)
                nyears = len(range(sdate.year,edate.year+1))
                self.month = np.tile(np.repeat(range(1,12+1), 30, 0), nyears)
                self.day = np.tile(np.tile(range(1,30+1), 12), nyears)
                if (sdate.day!=1)|(edate.month!=1):
                    sdoyi = (sdate.month-1)*30+sdate.day-1
                    self.year = self.year[sdoyi:]
                    self.month = self.month[sdoyi:]
                    self.day = self.day[sdoyi:]
                if (edate.day!=30)|(edate.month!=12):
                    edoyi = (12-edate.month)*30+(30-edate.day)
                    self.year = self.year[:-edoyi]
                    self.month = self.month[:-edoyi]
                    self.day = self.day[:-edoyi]
        dates = calendar360(dayone, daylast)
        shorten = 0
        if (daylast.day!=30)|(daylast.month!=12):
            shorten = 30*(13-daylast.month) - daylast.day
    else:
        daysinyear = 365
        # 365 day season start and end indices
        SHS = (304,455)
        SHW = (121,274)
        if tmaxnc.variables[timevname].units=='day as %Y%m%d.%f':
            st = str(int(nctime[0]))
            nd = str(int(nctime[-1]))
            dayone = dt.datetime(int(st[:4]), int(st[4:6]), int(st[6:]))
            daylast = dt.datetime(int(nd[:4]), int(nd[4:6]), int(nd[6:]))
        else:
            dayone = num2date(nctime[0], nctime.units,
                    calendar=calendar)
            daylast = num2date(nctime[-1], nctime.units,
                    calendar=calendar)
    #####added to correct for noleap calendar
        if calendar == 'noleap':
            dates = period_range(str(dayone), str(daylast), without_leap = True)
        else:
            dates = period_range(str(dayone), str(daylast), without_leap = False)
    ########
    ###    dates = pd.period_range(str(dayone), str(daylast))
        shorten = 0
        if (daylast.day!=30)|(daylast.month!=12):
            endofdata = dt.datetime(2000, daylast.month, daylast.day)
            shorten = dt.datetime(2000, 12, 31) - endofdata
            shorten = shorten.days

    # Load land-sea mask
    if maskfile:
        masknc = Dataset(maskfile, 'r')
        vname = maskvname
        mask = masknc.variables[vname][:]
        mask = mask>0.5
        mask = mask.astype(np.bool)
        masknc.close()

    # Load threshold data
    thresnc = Dataset(thresfile, 'r')
    threstime = thresnc.variables[timevname]
    txpct = thresnc.variables['txpct'][:]
    if len(txpct.shape)==4: txpct = txpct.squeeze()
    tnpct = thresnc.variables['tnpct'][:]
    if len(txpct.shape)==4: tnpct = tnpct.squeeze()
    if maskfile:
        txpct = txpct[:,mask]
        tnpct = tnpct[:,mask]
    vname = tmaxvname
    bptime = tmaxnc.variables[timevname]
    try:
        bptime = MFTime(bptime)
    except AttributeError:
        pass
    if tmaxnc.variables[timevname].units=='day as %Y%m%d.%f':
        st = str(int(bptime[0]))
        nd = str(int(bptime[-1]))
        bpdayone = dt.datetime(int(st[:4]), int(st[4:6]), int(st[6:]))
        bpdaylast = dt.datetime(int(nd[:4]), int(nd[4:6]), int(nd[6:]))
    else:
        bpdayone = num2date(bptime[0], bptime.units, calendar=calendar)
        bpdaylast = num2date(bptime[-1], bptime.units, calendar=calendar)

    #Added to correct for noleap calendar#########
    if calendar == 'noleap':
        bpdates = period_range(str(bpdayone), str(bpdaylast), without_leap = True)
    else:
        bpdates = period_range(str(bpdayone), str(bpdaylast), without_leap = False)
    ###########
    #if calendar=='360_day': bpdates = calendar360(bpdayone, bpdaylast)
    #else: 
    #    bpdates = pd.period_range(str(bpdayone), str(bpdaylast))
    dates_base = bpdates[(bpstart<=bpdates.year)&(bpdates.year<=bpend)]

    # Load all data
    try:
        tminnc = MFDataset(tminfile, 'r')
    except (IndexError, ValueError, OSError):
        tminnc = Dataset(tminfile, 'r')
    try:
        tmaxnc = MFDataset(tmaxfile, 'r')
    except (IndexError, ValueError, OSError):
        tmaxnc = Dataset(tmaxfile, 'r')
    tmax = tmaxnc.variables[tmaxvname][:]
    if len(tmax.shape)==4: tmax = tmax.squeeze()
    original_shape = tmax.shape
    tmin = tminnc.variables[tminvname][:]
    if len(tmin.shape)==4: tmin = tmin.squeeze()
    if maskfile:
        tmax = tmax[:,mask]
        tmin = tmin[:,mask]
    if tmaxnc.variables[tmaxvname].units=='K': tmax -= 273.15
    if tminnc.variables[tminvname].units=='K': tmin -= 273.15
    tave = (tmax + tmin)/2.

    # Remove leap days from tave
    if (calendar=='gregorian')|(calendar=='proleptic_gregorian')|\
                (calendar=='standard'):
        tave = tave[(dates.month!=2)|(dates.day!=29),...]
        tmax = tmax[(dates.month!=2)|(dates.day!=29),...]
        tmin = tmin[(dates.month!=2)|(dates.day!=29),...]

    # Remove incomplete starting year
    first_year = dayone.year
    if (dayone.month!=1)|(dayone.day!=1):
        first_year = dayone.year+1
        start = np.argmax(dates.year==first_year)
        tave = tave[start:,...]
        tmax = tmax[start:,...]
        tmin = tmin[start:,...]

    #Add synthetic data
    if syn:
        synnc = Dataset(syn, 'r')
        dmax = synnc.variables['DMAX'][:]
        dmin = synnc.variables['DMIN'][:]
        if len(dmax.shape)==4: dmax = dmax.squeeze()
        if len(dmin.shape)==4: dmin = dmin.squeeze()
        if maskfile:
            dmax = dmax[:,mask]
            dmin = dmin[:,mask]
        for i in range(0,tmax.shape[0]):
            idoy = i-daysinyear*int((i+1)/daysinyear)
            tmax[i,...] = tmax[i,...]+dmax[idoy,...]
            tmin[i,...] = tmin[i,...]+dmin[idoy,...]

    # Tx90pc exceedences
    txexceed = np.ones(tmax.shape)*np.nan
    tnexceed = txexceed.copy()
    for i in range(0,tmax.shape[0]):
        idoy = i-daysinyear*int((i+1)/daysinyear)
        txexceed[i,...] = tmax[i,...]>txpct[idoy,...]
        tnexceed[i,...] = tmin[i,...]>tnpct[idoy,...]
    txexceed[txexceed>0] = tmax[txexceed>0]
    tnexceed[tnexceed>0] = tmin[tnexceed>0]

    nyears = len(range(first_year,daylast.year+1))

    # Save to netCDF
    try:
        model = model
        realization = ensrn
        if realization==u'N/A': realization = u'r1i1p1'
    except AttributeError:
        model = ''
        realization = ''
    try:
        space = (tmaxnc.dimensions['lat'].__len__(),tmaxnc.dimensions['lon'].__len__())
        lonname = 'lon'
        latname = 'lat'
    except KeyError:
        lonname = 'longitude'
        latname = 'latitude'
        space = (tmaxnc.dimensions['latitude'].__len__(),tmaxnc.dimensions['longitude'].__len__())

    def identify_hw(ehfs, bmax, elength, e2length, dmin):
        """identify_hw locates heatwaves from EHF and returns duration indicators hw, chw, ahw.
        """
        
         # Agregate consecutive days with EHF>0, and consecutive days where EHF<0 (not heat wave)
        events = (ehfs>0.).astype(np.int)
        breaks = (events-1)/(-1)
        breake = breaks.copy()
        #event and break durations on first day of each:
        for i in range(ehfs.shape[0]-2,-1,-1):
             events[i,events[i,...]>0] = events[i+1,events[i,...]>0]+1
             breaks[i,breaks[i,...]>0] = breaks[i+1,breaks[i,...]>0]+1
        #event and break durations on last day of each:
        for i in range(1,ehfs.shape[0],1):
            breake[i,breake[i,...]>0] = breake[i-1,breake[i,...]>0]+1 #break duration on last day of break
        #Identify when heatwaves and breaks start and end with duration
        #Note that doesn't work for events that start on the first day (maybe should correct later)
        diffes = np.zeros(ehfs.shape)
        diffbs = np.zeros(ehfs.shape)
        diffbe = np.zeros(ehfs.shape)
        diffes[1:,...] = np.diff(events, axis=0).astype(np.int)
        diffbs[1:,...] = np.diff(breaks, axis=0).astype(np.int)
        diffbe[0:ehfs.shape[0]-1,...] = np.diff(breake, axis=0).astype(np.int)
        #Edit ends so breaks/events can end on last day, and start on first day.
        diffes[0,...]=events[0,...]
        diffbs[0,...]=breaks[0,...]
        diffbe[ehfs.shape[0]-1,...]=-1*breake[ehfs.shape[0]-1,...]
        endes = np.zeros(ehfs.shape,dtype=np.int)
        endbs = np.zeros(ehfs.shape,dtype=np.int)
        endbe = np.zeros(ehfs.shape,dtype=np.int)
        endes[diffes>0] = events[diffes>0]
        endbs[diffbs>0] = breaks[diffbs>0]
        endbe[diffbe<0] = breake[diffbe<0]
        del breaks
        del events
        del breake
        #del breake, breaks, diffes, diffbs, diffbe
        ##################
        #HW: Find non-compound heat waves of minimum length elength
        hw = np.zeros(ehfs.shape, dtype=np.int)
        hw[endes>=elength] = endes[endes>=elength]
        ##################
        #CHW: Find compound heat waves with first event >= elength, compounded events >= e2length, and total number of hot days >= dmin
        #AHW: Find compound and non-compound heat waves
        #Place in array break durations before and after events at same elements as event starts
        lasttime=ehfs.shape[0]-1
        breakbefore = np.roll(endbe,1, axis=0)
        breakafter=np.zeros(ehfs.shape, dtype=np.int)
        for i in range(0,lasttime,1):
            xy=endes[i,...]>0 #only evaluate for first days of events
            eduration=endes[i,xy]
            bsi=i+eduration
            bsi[bsi>lasttime]=0 #first day chw always=0, so use this index
            breakafter[i,xy]=endbs[bsi,xy]
        #Populate CHW output array with only events that might compound (break before or after <=bmax, length >= e2length), and AHW output array with only events >= e2length
        chw = np.zeros(ehfs.shape, dtype=np.int)
        bora = np.logical_or(breakafter<=bmax,breakbefore<=bmax)
        e2 = endes>=e2length
        boraande2 = np.logical_and(bora,e2)
        chw[boraande2] = endes[boraande2]
        #del bora, boraande2, e2
        #Add events with breaks<=bmax to ones following in time, iterated backwards in time so accumulate
        #chw counts only event (hot) days, while endlength counts days that are hot and days of breaks
        chw = np.append(chw,np.expand_dims(np.zeros(ehfs[0,...].shape),axis=0),0) #Append additional day
        endlength=chw.copy()
        size = ehfs.shape[0]
        for i in range(lasttime,-1,-1):
            xy=np.logical_and(chw[i,...]>0,breakafter[i,...]<=bmax) #only evaluation for first days of events part of compound events, where the break after is less than max break length
            eduration=endes[i,xy]
            bafter=breakafter[i,xy]
            eaddi=i+eduration+bafter
            eaddi[eaddi>lasttime]=size #first day endss always=0, so use this index
            chw[i,xy]=eduration+chw[eaddi,xy] #no. of event days in compound heat wave
            endlength[i,xy]=eduration+bafter+endlength[eaddi,xy] #no. of days (breaks and evnets) in compound heat wave
        chw = np.delete(chw,size,0) #Remove additional day with value 0
        endlength = np.delete(endlength,size,0) #Remove additional day with value 0
        chw[endes<elength]=0 #Turn compound events that can't be starts (original event length < elength) to zero
        chw[chw==endes] = 0 #Gets rid of heat waves at end that aren't compound
        chw[chw<dmin] = 0 #Get rid of heat waves that don't meet a minimum duration requirement
        #combine hw and chw to get preliminary ahw
        ahw=chw+hw
        chwhw = np.logical_and(chw>0,hw>0)
        ahw[chwhw]=chw[chwhw] #ensures don't double count hw that are summed into chw already
        del chwhw
        #Turn compound events that are not at beginning of event to zero for chw and ahw
        for i in range(0,lasttime,1):
            for index, j in np.ndenumerate(chw[i,...]):
                    if chw[i,index[0]]>0: #add ,index[1] after  index[0] if 3-D array
                        length=int(endlength[i,index[0]]) #add ,index[1] after  index[0] if 3-D array
                        elasti=i+length-1
                        chw[(i+1):elasti,index[0]]=0 #add ,index[1] after  index[0] if 3-D array
                        ahw[(i+1):elasti,index[0]]=0
        #secondary ahw (events that compound on to other events)
        ahw2 = chw-hw
        ahw2[ahw2<0] = 0
        #primary ahw (compound starting events, and events with no compounding)
        ahw1 = ahw-ahw2
        
        return hw, chw, ahw, ahw1, ahw2

    def hw_aspects(EHF, season, hemisphere, tmp_out_path, bmax, elength, e2length, dmin):
        """hw_aspects takes EHF values or temp 90pct exceedences identifies
        heatwaves and calculates seasonal aspects.
        """
        # Select indices depending on calendar season and hemisphere
        if season=='summer':
            if hemisphere=='south':
                startday = SHS[0]
                endday = SHS[1]
            else:
                startday = SHW[0]
                endday = SHW[1]
        elif season=='winter':
            if hemisphere=='south':
                startday = SHW[0]
                endday = SHW[1]
            else:
                startday = SHS[0]
                endday = SHS[1]
        

        def calculate_stats_for_year(out_path, uuid_, iyear_, year_):
            # Select this years season
            allowance = 0 # For including heatwave days after the end of the season
            ifrom = startday + daysinyear*iyear
            ito = endday + daysinyear*iyear + allowance
            EHF_i = EHF[ifrom:ito,...]
            hw_i, chw_i, ahw_i, ahw1_i, ahw2_i = identify_hw(EHF_i, bmax, elength, e2length, dmin)
            # Remove events that start after the end of the season and before start
            # EHF_i = EHF_i[0:,...]
            del EHF_i
            
            def calculate_metrics(stat_, labels_, out_path_, iyear__, uuid__):
                if len(labels_) == 3:
                    with open(f"{out_path}{iyear_}-{labels_[0]}-{uuid_}.npy", 'wb') as f:
                        np.save(f, (stat_>0).sum(axis=0))
                    with open(f"{out_path}{iyear_}-{labels_[1]}-{uuid_}.npy", 'wb') as f:
                        np.save(f, stat_.sum(axis=0))
                    with open(f"{out_path}{iyear_}-{labels_[2]}-{uuid_}.npy", 'wb') as f:
                        np.save(f, stat_.max(axis=0))
                else:
                    with open(f"{out_path_}{iyear__}-{labels_[0]}-{uuid__}.npy", 'wb') as f:
                        np.save(f, stat_.sum(axis=0))
                    with open(f"{out_path_}{iyear__}-{labels_[1]}-{uuid__}.npy", 'wb') as f:
                        np.save(f, stat_.max(axis=0))
            
            
            stat_labels = [(hw_i, ["HWN", "HWF", "HWD"]), 
                           (chw_i, ["CHWN", "CHWF", "CHWD"]), 
                           (ahw_i, ["AHWN", "AHWF", "AHWD"]), 
                           (ahw1_i, ["AHW1N", "AHW1F", "AHW1D"]), 
                           (ahw2_i, ["AHW2F", "AHW2D"])]
            
            threads = []
            
            for stat, labels in stat_labels:
                thread = threading.Thread(target=calculate_metrics, args=(stat, labels, out_path, iyear_, uuid_,), daemon=True)
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()
            
            
            # Calculate metrics
#             hw_i = hw_i[0:]
#             with open(f"{out_path}{iyear_}-HWN-{uuid_}.npy", 'wb') as f:
#                 np.save(f, (hw_i>0).sum(axis=0))
#             with open(f"{out_path}{iyear_}-HWF-{uuid_}.npy", 'wb') as f:
#                 np.save(f, hw_i.sum(axis=0))
#             with open(f"{out_path}{iyear_}-HWD-{uuid_}.npy", 'wb') as f:
#                 np.save(f, hw_i.max(axis=0))
#             del hw_i
#             chw_i = chw_i[0:]
#             with open(f"{out_path}{iyear_}-CHWN-{uuid_}.npy", 'wb') as f:
#                 np.save(f, (chw_i>0).sum(axis=0))
#             with open(f"{out_path}{iyear_}-CHWF-{uuid_}.npy", 'wb') as f:
#                 np.save(f, chw_i.sum(axis=0))
#             with open(f"{out_path}{iyear_}-CHWD-{uuid_}.npy", 'wb') as f:
#                 np.save(f, chw_i.max(axis=0))
#             del chw_i
#             ahw_i = ahw_i[0:]
#             with open(f"{out_path}{iyear_}-AHWN-{uuid_}.npy", 'wb') as f:
#                 np.save(f, (ahw_i>0).sum(axis=0))
#             with open(f"{out_path}{iyear_}-AHWF-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw_i.sum(axis=0))
#             with open(f"{out_path}{iyear_}-AHWD-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw_i.max(axis=0))
#             del ahw_i
#             ahw1_i = ahw1_i[0:]
#             with open(f"{out_path}{iyear_}-AHW1N-{uuid_}.npy", 'wb') as f:
#                 np.save(f, (ahw1_i>0).sum(axis=0))
#             with open(f"{out_path}{iyear_}-AHW1F-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw1_i.sum(axis=0))
#             with open(f"{out_path}{iyear_}-AHW1D-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw1_i.max(axis=0))
#             del ahw1_i
#             ahw2_i = ahw2_i[0:]
#             with open(f"{out_path}{iyear_}-AHW2F-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw2_i.sum(axis=0))
#             with open(f"{out_path}{iyear_}-AHW2D-{uuid_}.npy", 'wb') as f:
#                 np.save(f, ahw2_i.max(axis=0))
#             del ahw2_i

        uuid = str(uuid4())
        if not isdir(tmp_out_path):
            mkdir(tmp_out_path)

        processes = []
        num_procs = 0
        for iyear, year in enumerate(range(first_year, daylast.year)):
            print(f"{year} out of {daylast.year}")
            if year == daylast.year:
                continue  # Incomplete yr
            proc = Process(target=calculate_stats_for_year, args=(tmp_out_path, uuid, iyear, year,))
            proc.daemon = True
            proc.start()
            processes.append(proc)

        for process in processes:
            process.join()
        out_path = tmp_out_path
        uuid_ = uuid

        # Initialize arrays
        HWN = np.ones(((nyears,)+(EHF.shape[1],)))*np.nan
        
        # Loop over years
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-HWN-{uuid}.npy", 'rb')as f:
                HWN[iyear,...] = np.load(f)
        with open(f"{out_path}completed-HWN-{uuid_}.npy", 'wb') as f:
            np.save(f, HWN)

        HWF = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-HWF-{uuid}.npy", 'rb')as f:
                HWF[iyear,...] = np.load(f)
        with open(f"{out_path}completed-HWF-{uuid_}.npy", 'wb') as f:
            np.save(f, HWF)
        del HWF

        HWD = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-HWD-{uuid}.npy", 'rb')as f:
                HWD[iyear,...] = np.load(f)
        with open(f"{out_path}completed-HWD-{uuid_}.npy", 'wb') as f:
            np.save(f, HWD)
        del HWD

        CHWN = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-CHWN-{uuid}.npy", 'rb')as f:
                CHWN[iyear,...] = np.load(f)
        with open(f"{out_path}completed-CHWN-{uuid_}.npy", 'wb') as f:
            np.save(f, CHWN)
        del CHWN
        
        CHWF = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-CHWF-{uuid}.npy", 'rb')as f:
                CHWF[iyear,...] = np.load(f)
        with open(f"{out_path}completed-CHWF-{uuid_}.npy", 'wb') as f:
            np.save(f, CHWF)
        del CHWF

        CHWD = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-CHWD-{uuid}.npy", 'rb')as f:
                CHWD[iyear,...] = np.load(f)
        with open(f"{out_path}completed-CHWD-{uuid_}.npy", 'wb') as f:
            np.save(f, CHWD)
        del CHWD

        AHWN = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHWN-{uuid}.npy", 'rb')as f:
                AHWN[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHWN-{uuid_}.npy", 'wb') as f:
            np.save(f, AHWN)
        del AHWN

        AHWF = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHWF-{uuid}.npy", 'rb')as f:
                AHWF[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHWF-{uuid_}.npy", 'wb') as f:
            np.save(f, AHWF)
        del AHWF

        AHWD = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHWD-{uuid}.npy", 'rb')as f:
                AHWD[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHWD-{uuid_}.npy", 'wb') as f:
            np.save(f, AHWD)
        del AHWD

        AHW1N = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHW1N-{uuid}.npy", 'rb')as f:
                AHW1N[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHW1N-{uuid_}.npy", 'wb') as f:
            np.save(f, AHW1N)
        del AHW1N

        AHW1F = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHW1F-{uuid}.npy", 'rb')as f:
                AHW1F[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHW1F-{uuid_}.npy", 'wb') as f:
            np.save(f, AHW1F)
        del AHW1F

        AHW1D = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHW1D-{uuid}.npy", 'rb')as f:
                AHW1D[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHW1D-{uuid_}.npy", 'wb') as f:
            np.save(f, AHW1D)
        del AHW1D

        AHW2F = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHW2F-{uuid}.npy", 'rb')as f:
                AHW2F[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHW2F-{uuid_}.npy", 'wb') as f:
            np.save(f, AHW2F)
        del AHW2F

        AHW2D = HWN.copy()
        for iyear, year in enumerate(range(first_year, daylast.year)):
            with open(f"{tmp_out_path}{iyear}-AHW2D-{uuid}.npy", 'rb')as f:
                AHW2D[iyear,...] = np.load(f)
        with open(f"{out_path}completed-AHW2D-{uuid_}.npy", 'wb') as f:
            np.save(f, AHW2D)
        del AHW2D

        datasets = listdir(tmp_out_path)
        tmp_files = [tmp_out_path + name for name in datasets if uuid in name and "completed" not in name]
        for filename in set(tmp_files):
            remove(filename)

        return uuid


    # Calculate metrics year by year
    def split_hemispheres(EHF, bmax, elength, e2length, dmin, north, south, lats):
        """split_hemispheres splits the input data by hemispheres, and glues them
        back together after heatwave calculations.
        """
        tmp_out_path = "tmp/"
        if south:
            if maskfile:
                EHF_s = EHF[:,:(mask[lats<=0]>0).sum()]
            else:
                EHF_s = EHF[:,lats<=0,...]
            # Reshape to 2D
            space = EHF_s.shape[1:]
            if len(space)>1:
                EHF_s = EHF_s.reshape(EHF_s.shape[0],space[0]*space[1])
            # Southern hemisphere aspects
            uuid_s = hw_aspects(EHF_s, season, 'south', tmp_out_path, bmax, elength, e2length, dmin)
            # HWN_s, HWF_s, HWD_s, CHWN_s, CHWF_s, CHWD_s, AHWN_s, AHWF_s, AHWD_s, AHW1N_s, AHW1F_s, AHW1D_s, AHW2F_s, AHW2D_s  = \
            del EHF_s
        if north:
            if maskfile:
                EHF_n = EHF[:,(mask[lats<=0]>0).sum():]
            else:
                EHF_n = EHF[:,lats>0,...]
            # Reshape to 2D
            space = EHF_n.shape[1:]
            if len(space)>1:
                EHF_n = EHF_n.reshape(EHF_n.shape[0],space[0]*space[1])
            # Northern hemisphere aspects
            # HWN_n, HWF_n, HWD_n, CHWN_n, CHWF_n, CHWD_n, AHWN_n, AHWF_n, AHWD_n, AHW1N_n, AHW1F_n, AHW1D_n, AHW2F_n, AHW2D_n = \
            uuid_n = hw_aspects(EHF_n, season, 'north', tmp_out_path, bmax, elength, e2length, dmin)
            del EHF_n
        # Glue hemispheres back together
        if north and south:
            with open(f"{tmp_out_path}completed-HWN-{uuid_n}.npy", 'rb') as f:
                HWN_n = np.load(f)
            with open(f"{tmp_out_path}completed-HWN-{uuid_s}.npy", 'rb') as f:
                HWN_s = np.load(f)
            HWN = np.append(HWN_s, HWN_n, axis=1)
            del HWN_s
            del HWN_n

            with open(f"{tmp_out_path}completed-HWF-{uuid_n}.npy", 'rb') as f:
                HWF_n = np.load(f)
            with open(f"{tmp_out_path}completed-HWF-{uuid_s}.npy", 'rb') as f:
                HWF_s = np.load(f)
            HWF = np.append(HWF_s, HWF_n, axis=1)
            del HWF_s
            del HWF_n

            with open(f"{tmp_out_path}completed-HWD-{uuid_n}.npy", 'rb') as f:
                HWD_n = np.load(f)
            with open(f"{tmp_out_path}completed-HWD-{uuid_s}.npy", 'rb') as f:
                HWD_s = np.load(f)
            HWD = np.append(HWD_s, HWD_n, axis=1)
            del HWD_s
            del HWD_n

            with open(f"{tmp_out_path}completed-CHWN-{uuid_n}.npy", 'rb') as f:
                CHWN_n = np.load(f)
            with open(f"{tmp_out_path}completed-CHWN-{uuid_s}.npy", 'rb') as f:
                CHWN_s = np.load(f)
            CHWN = np.append(CHWN_s, CHWN_n, axis=1)
            del CHWN_s
            del CHWN_n

            with open(f"{tmp_out_path}completed-CHWF-{uuid_n}.npy", 'rb') as f:
                CHWF_n = np.load(f)
            with open(f"{tmp_out_path}completed-CHWF-{uuid_s}.npy", 'rb') as f:
                CHWF_s = np.load(f)
            CHWF = np.append(CHWF_s, CHWF_n, axis=1)
            del CHWF_s
            del CHWF_n

            with open(f"{tmp_out_path}completed-CHWD-{uuid_n}.npy", 'rb') as f:
                CHWD_n = np.load(f)
            with open(f"{tmp_out_path}completed-CHWD-{uuid_s}.npy", 'rb') as f:
                CHWD_s = np.load(f)
            CHWD = np.append(CHWD_s, CHWD_n, axis=1)
            del CHWD_s
            del CHWD_n

            with open(f"{tmp_out_path}completed-AHWN-{uuid_n}.npy", 'rb') as f:
                AHWN_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHWN-{uuid_s}.npy", 'rb') as f:
                AHWN_s = np.load(f)
            AHWN = np.append(AHWN_s, AHWN_n, axis=1)
            del AHWN_s
            del AHWN_n

            with open(f"{tmp_out_path}completed-AHWF-{uuid_n}.npy", 'rb') as f:
                AHWF_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHWF-{uuid_s}.npy", 'rb') as f:
                AHWF_s = np.load(f)
            AHWF = np.append(AHWF_s, AHWF_n, axis=1)
            del AHWF_s
            del AHWF_n

            with open(f"{tmp_out_path}completed-AHWD-{uuid_n}.npy", 'rb') as f:
                AHWD_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHWD-{uuid_s}.npy", 'rb') as f:
                AHWD_s = np.load(f)
            AHWD = np.append(AHWD_s, AHWD_n, axis=1)
            del AHWD_s
            del AHWD_n

            with open(f"{tmp_out_path}completed-AHW1N-{uuid_n}.npy", 'rb') as f:
                AHW1N_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHW1N-{uuid_s}.npy", 'rb') as f:
                AHW1N_s = np.load(f)
            AHW1N = np.append(AHW1N_s, AHW1N_n, axis=1)
            del AHW1N_s
            del AHW1N_n

            with open(f"{tmp_out_path}completed-AHW1F-{uuid_n}.npy", 'rb') as f:
                AHW1F_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHW1F-{uuid_s}.npy", 'rb') as f:
                AHW1F_s = np.load(f)
            AHW1F = np.append(AHW1F_s, AHW1F_n, axis=1)
            del AHW1F_s
            del AHW1F_n

            with open(f"{tmp_out_path}completed-AHW1D-{uuid_n}.npy", 'rb') as f:
                AHW1D_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHW1D-{uuid_s}.npy", 'rb') as f:
                AHW1D_s = np.load(f)
            AHW1D = np.append(AHW1D_s, AHW1D_n, axis=1)
            del AHW1D_s
            del AHW1D_n

            with open(f"{tmp_out_path}completed-AHW2F-{uuid_n}.npy", 'rb') as f:
                AHW2F_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHW2F-{uuid_s}.npy", 'rb') as f:
                AHW2F_s = np.load(f)
            AHW2F = np.append(AHW2F_s, AHW2F_n, axis=1)
            del AHW2F_s
            del AHW2F_n

            with open(f"{tmp_out_path}completed-AHW2D-{uuid_n}.npy", 'rb') as f:
                AHW2D_n = np.load(f)
            with open(f"{tmp_out_path}completed-AHW2D-{uuid_s}.npy", 'rb') as f:
                AHW2D_s = np.load(f)
            AHW2D = np.append(AHW2D_s, AHW2D_n, axis=1)
            del AHW2D_s
            del AHW2D_n

        elif north:
            with open(f"{tmp_out_path}completed-HWN-{uuid_n}.npy", 'rb') as f:
                HWN_n = np.load(f)
            HWN = HWN_n
            del HWN_n

            with open(f"{tmp_out_path}completed-HWF-{uuid_n}.npy", 'rb') as f:
                HWF_n = np.load(f)
            HWF = HWF_n
            del HWF_n

            with open(f"{tmp_out_path}completed-HWD-{uuid_n}.npy", 'rb') as f:
                HWD_n = np.load(f)
            HWD = HWD_n
            del HWD_n

            with open(f"{tmp_out_path}completed-CHWN-{uuid_n}.npy", 'rb') as f:
                CHWN_n = np.load(f)
            CHWN = CHWN_n
            del CHWN_n

            with open(f"{tmp_out_path}completed-CHWF-{uuid_n}.npy", 'rb') as f:
                CHWF_n = np.load(f)
            CHWF = CHWF_n
            del CHWF_n

            with open(f"{tmp_out_path}completed-CHWD-{uuid_n}.npy", 'rb') as f:
                CHWD_n = np.load(f)
            CHWD = CHWD_n
            del CHWD_n

            with open(f"{tmp_out_path}completed-AHWN-{uuid_n}.npy", 'rb') as f:
                AHWN_n = np.load(f)
            AHWN = AHWN_n
            del AHWN_n

            with open(f"{tmp_out_path}completed-AHWF-{uuid_n}.npy", 'rb') as f:
                AHWF_n = np.load(f)
            AHWF = AHWF_n
            del AHWF_n

            with open(f"{tmp_out_path}completed-AHWD-{uuid_n}.npy", 'rb') as f:
                AHWD_n = np.load(f)
            AHWD = AHWD_n
            del AHWD_n

            with open(f"{tmp_out_path}completed-AHW1N-{uuid_n}.npy", 'rb') as f:
                AHW1N_n = np.load(f)
            AHW1N = AHW1N_n
            del AHW1N_n

            with open(f"{tmp_out_path}completed-AHW1F-{uuid_n}.npy", 'rb') as f:
                AHW1F_n = np.load(f)
            AHW1F = AHW1F_n
            del AHW1F_n

            with open(f"{tmp_out_path}completed-AHW1D-{uuid_n}.npy", 'rb') as f:
                AHW1D_n = np.load(f)
            AHW1D = AHW1D_n
            del AHW1D_n

            with open(f"{tmp_out_path}completed-AHW2F-{uuid_n}.npy", 'rb') as f:
                AHW2F_n = np.load(f)
            AHW2F = AHW2F_n
            del AHW2F_n

            with open(f"{tmp_out_path}completed-AHW2D-{uuid_n}.npy", 'rb') as f:
                AHW2D_n = np.load(f)
            AHW2D = AHW2D_n
            del AHW2D_n
        elif south:
            with open(f"{tmp_out_path}completed-HWN-{uuid_s}.npy", 'rb') as f:
                HWN_s = np.load(f)
            HWN = HWN_s
            del HWN_s

            with open(f"{tmp_out_path}completed-HWF-{uuid_s}.npy", 'rb') as f:
                HWF_s = np.load(f)
            HWF = HWF_s
            del HWF_s

            with open(f"{tmp_out_path}completed-HWD-{uuid_s}.npy", 'rb') as f:
                HWD_s = np.load(f)
            HWD = HWD_s
            del HWD_s

            with open(f"{tmp_out_path}completed-CHWN-{uuid_s}.npy", 'rb') as f:
                CHWN_s = np.load(f)
            CHWN = CHWN_s
            del CHWN_s

            with open(f"{tmp_out_path}completed-CHWF-{uuid_s}.npy", 'rb') as f:
                CHWF_s = np.load(f)
            CHWF = CHWF_s
            del CHWF_s

            with open(f"{tmp_out_path}completed-CHWD-{uuid_s}.npy", 'rb') as f:
                CHWD_s = np.load(f)
            CHWD = CHWD_s
            del CHWD_s

            with open(f"{tmp_out_path}completed-AHWN-{uuid_s}.npy", 'rb') as f:
                AHWN_s = np.load(f)
            AHWN = AHWN_s
            del AHWN_s

            with open(f"{tmp_out_path}completed-AHWF-{uuid_s}.npy", 'rb') as f:
                AHWF_s = np.load(f)
            AHWF = AHWF_s
            del AHWF_s

            with open(f"{tmp_out_path}completed-AHWD-{uuid_s}.npy", 'rb') as f:
                AHWD_s = np.load(f)
            AHWD = AHWD_s
            del AHWD_s

            with open(f"{tmp_out_path}completed-AHW1N-{uuid_s}.npy", 'rb') as f:
                AHW1N_s = np.load(f)
            AHW1N = AHW1N_s
            del AHW1N_s

            with open(f"{tmp_out_path}completed-AHW1F-{uuid_s}.npy", 'rb') as f:
                AHW1F_s = np.load(f)
            AHW1F = AHW1F_s
            del AHW1F_s

            with open(f"{tmp_out_path}completed-AHW1D-{uuid_s}.npy", 'rb') as f:
                AHW1D_s = np.load(f)
            AHW1D = AHW1D_s
            del AHW1D_s

            with open(f"{tmp_out_path}completed-AHW2F-{uuid_s}.npy", 'rb') as f:
                AHW2F_s = np.load(f)
            AHW2F = AHW2F_s
            del AHW2F_s

            with open(f"{tmp_out_path}completed-AHW2D-{uuid_s}.npy", 'rb') as f:
                AHW2D_s = np.load(f)
            AHW2D = AHW2D_s
            del AHW2D_s
        datasets = listdir(tmp_out_path)
        tmp_files = [tmp_out_path + name for name in datasets if uuid_s in name or uuid_n in name]
        for filename in set(tmp_files):
            remove(filename)

        return HWN, HWF, HWD, CHWN, CHWF, CHWD, AHWN, AHWF, AHWD, AHW1N, AHW1F, AHW1D, AHW2F, AHW2D


    def save_yearly(HWN, HWF, HWD, CHWN, CHWF, CHWD, AHWN, AHWF, AHWD, AHW1N, AHW1F, AHW1D, AHW2F, AHW2D, definition, basedat, experiment, label):
        yearlyout = Dataset(output_path + definition + ".nc", 'w')
        yearlyout.createDimension('time', len(range(first_year,
                daylast.year+1)))
        yearlyout.createDimension('lon', tmaxnc.dimensions[lonname].__len__())
        yearlyout.createDimension('lat', tmaxnc.dimensions[latname].__len__())
        yearlyout.createDimension('day', daysinyear)
        setattr(yearlyout, "author", "Tammas Loughran")
        setattr(yearlyout, "contact", "t.loughran@student.unsw.edu.au")
        setattr(yearlyout, "source", "https://github.com/tammasloughran/ehfheatwaves")
        setattr(yearlyout, "date", dt.datetime.today().strftime('%Y-%m-%d'))
        setattr(yearlyout, "script", sys.argv[0])
        if model:
            setattr(yearlyout, "model_id", model)
            setattr(yearlyout, "experiment", experiment)
            setattr(yearlyout, "realization", "%s"%(realization))
        setattr(yearlyout, "period", "%s-%s"%(str(first_year),str(daylast.year)))
        setattr(yearlyout, "base_period", "%s-%s"%(str(bpstart),str(bpend)))
        setattr(yearlyout, "percentile", "%sth"%(str(pcntl)))
        setattr(yearlyout, "definition", definition)
        setattr(yearlyout, "frequency", "yearly")
        setattr(yearlyout, "season", season)
        setattr(yearlyout, "season_note", ("The year of a season is the year it starts"
                "in. SH summer: Nov-Mar. NH summer: May-Sep."))
        try:
            file = open('version', 'r')
            commit = file.read()[:-2]
        except IOError:
            commit = "Unknown. Check date for latest version."
        setattr(yearlyout, "git_commit", commit)
        setattr(yearlyout, "tmax_file", tmaxfile)
        setattr(yearlyout, "tmin_file", tminfile)
        if maskfile:
            setattr(yearlyout, "mask_file", maskfile)
        setattr(yearlyout, "quantile_method", "climpact")
        otime = yearlyout.createVariable('time', 'f8', 'time', 
                fill_value=-999.99)
        setattr(otime, 'units', 'year')
        olat = yearlyout.createVariable('lat', 'f8', 'lat')
        setattr(olat, 'standard_name', 'latitude')
        setattr(olat, 'long_name', 'Latitude')
        setattr(olat, 'units', 'degrees_north')
        setattr(olat, 'axis', 'Y')
        olon = yearlyout.createVariable('lon', 'f8', 'lon')
        setattr(olon, 'standard_name', 'longiitude')
        setattr(olon, 'long_name', 'Longitude')
        setattr(olon, 'units', 'degrees_east')
        setattr(olon, 'axis', 'X')
        otpct = yearlyout.createVariable(label, 'f8',
            ('day','lat','lon'), fill_value=-999.99)
        setattr(otpct, 'long_name', '%s %s percentile'%(label,str(pcntl).rstrip('.')))
        setattr(otpct, 'units', 'degC')
        setattr(otpct, 'description', 
                '%s %sth percentile of %s-%s'%(label,f"tx{str(pcntl*100)}",str(bpstart),str(bpend)))
        HWNout = yearlyout.createVariable('HWN_%s'%(definition), 'f8', ('time', 'lat', 'lon'), 
                fill_value=-999.99)
        setattr(HWNout, 'long_name', 'Non-compound Heatwave Number')
        setattr(HWNout, 'units','heatwaves')
        setattr(HWNout, 'description', 'Number of heatwaves per year')
        HWFout = yearlyout.createVariable('HWF_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(HWFout, 'long_name','Non-compound Heatwave Frequency')
        setattr(HWFout, 'units', 'days')
        setattr(HWFout, 'description', 'Proportion of non-compound heatwave days per season')
        HWDout = yearlyout.createVariable('HWD_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(HWDout, 'long_name', 'Non-compound Heatwave Duration')
        setattr(HWDout, 'units', 'days')
        setattr(HWDout, 'description', 'Duration of the longest non-compound heatwave per year')
        CHWNout = yearlyout.createVariable('CHWN_%s'%(definition), 'f8', ('time', 'lat', 'lon'), 
                fill_value=-999.99)
        setattr(CHWNout, 'long_name', 'Compound Heatwave Number')
        setattr(CHWNout, 'units','heatwaves')
        setattr(CHWNout, 'description', 'Number of compound heatwaves per year')
        CHWFout = yearlyout.createVariable('CHWF_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(CHWFout, 'long_name','Compound Heatwave Frequency')
        setattr(CHWFout, 'units', 'days')
        setattr(CHWFout, 'description', 'Proportion of compound heatwave days per season')
        CHWDout = yearlyout.createVariable('CHWD_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(CHWDout, 'long_name', 'Compound Heatwave Duration')
        setattr(CHWDout, 'units', 'days')
        setattr(CHWDout, 'description', 'Duration of the longest compound heatwave per year')
        AHWNout = yearlyout.createVariable('AHWN_%s'%(definition), 'f8', ('time', 'lat', 'lon'), 
                fill_value=-999.99)
        setattr(AHWNout, 'long_name', 'All Heatwave Number')
        setattr(AHWNout, 'units','heatwaves')
        setattr(AHWNout, 'description', 'Number of compound and non-compound heatwaves per year')
        AHWFout = yearlyout.createVariable('AHWF_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHWFout, 'long_name','All Heatwave Frequency')
        setattr(AHWFout, 'units', 'days')
        setattr(AHWFout, 'description', 'Proportion of compound and non-compound heatwave days per season')
        AHWDout = yearlyout.createVariable('AHWD_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHWDout, 'long_name', 'All Heatwave Duration')
        setattr(AHWDout, 'units', 'days')
        setattr(AHWDout, 'description', 'Duration of the longest of any heatwave (compound or non-compound) per year')
        AHW1Nout = yearlyout.createVariable('AHW1N_%s'%(definition), 'f8', ('time', 'lat', 'lon'), 
                fill_value=-999.99)
        setattr(AHW1Nout, 'long_name', 'Primary Heatwave Number')
        setattr(AHW1Nout, 'units','heatwaves')
        setattr(AHW1Nout, 'description', 'Number of heatwave starting events per year')
        AHW1Fout = yearlyout.createVariable('AHW1F_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHW1Fout, 'long_name','Primary Heatwave Frequency')
        setattr(AHW1Fout, 'units', 'days')
        setattr(AHW1Fout, 'description', 'Proportion of heatwave days in starting heatwave per season')
        AHW1Dout = yearlyout.createVariable('AHW1D_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHW1Dout, 'long_name', 'Primary Heatwave Duration')
        setattr(AHW1Dout, 'units', 'days')
        setattr(AHW1Dout, 'description', 'Duration of the longest of any starting heatwave (compound or non-compound) per year')
        AHW2Fout = yearlyout.createVariable('AHW2F_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHW2Fout, 'long_name','Secondary Heatwave Frequency')
        setattr(AHW2Fout, 'units', 'days')
        setattr(AHW2Fout, 'description', 'Proportion of secondary heatwave days per season')
        AHW2Dout = yearlyout.createVariable('AHW2D_%s'%(definition), 'f8', ('time','lat','lon'), 
                fill_value=-999.99)
        setattr(AHW2Dout, 'long_name', 'Secondary Heatwave Duration')
        setattr(AHW2Dout, 'units', 'days')
        setattr(AHW2Dout, 'description', 'Duration of the longest number of secondary heat wave days for an individual heat wave (compound or non-compound) per year')
        otime[:] = range(first_year, daylast.year+1)
        olat[:] = tmaxnc.variables[latname][:]
        olon[:] = tmaxnc.variables[lonname][:]
        dummy_array = np.ones((daysinyear,)+original_shape[1:])*np.nan
        if maskfile:
            dummy_array[:,mask] = basedat
            dummy_array[np.isnan(dummy_array)] = -999.99
            otpct[:] = dummy_array.copy()
            dummy_array = np.ones((nyears,)+original_shape[1:])*np.nan
            dummy_array[:,mask] = HWN
            dummy_array[np.isnan(dummy_array)] = -999.99
            HWNout[:] = dummy_array.copy()
            dummy_array[:,mask] = HWF
            dummy_array[np.isnan(dummy_array)] = -999.99
            HWFout[:] = dummy_array.copy()
            dummy_array[:,mask] = HWD
            dummy_array[np.isnan(dummy_array)] = -999.99
            HWDout[:] = dummy_array.copy()
            dummy_array[:,mask] = CHWN
            dummy_array[np.isnan(dummy_array)] = -999.99
            CHWNout[:] = dummy_array.copy()
            dummy_array[:,mask] = CHWF
            dummy_array[np.isnan(dummy_array)] = -999.99
            CHWFout[:] = dummy_array.copy()
            dummy_array[:,mask] = CHWD
            dummy_array[np.isnan(dummy_array)] = -999.99
            CHWDout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHWN
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHWNout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHWF
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHWFout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHWD
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHWDout[:] = dummy_array.copy() 
            dummy_array[:,mask] = AHW1N
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHW1Nout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHW1F
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHW1Fout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHW1D
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHW1Dout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHW2F
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHW2Fout[:] = dummy_array.copy()
            dummy_array[:,mask] = AHW2D
            dummy_array[np.isnan(dummy_array)] = -999.99
            AHW2Dout[:] = dummy_array.copy()
        else:
            otpct[:] = basedat
            HWNout[:] = HWN.reshape((nyears,)+space)
            HWFout[:] = HWF.reshape((nyears,)+space)
            HWDout[:] = HWD.reshape((nyears,)+space)
            CHWNout[:] = CHWN.reshape((nyears,)+space)
            CHWFout[:] = CHWF.reshape((nyears,)+space)
            CHWDout[:] = CHWD.reshape((nyears,)+space)
            AHWNout[:] = AHWN.reshape((nyears,)+space)
            AHWFout[:] = AHWF.reshape((nyears,)+space)
            AHWDout[:] = AHWD.reshape((nyears,)+space)
            AHW1Nout[:] = AHW1N.reshape((nyears,)+space)
            AHW1Fout[:] = AHW1F.reshape((nyears,)+space)
            AHW1Dout[:] = AHW1D.reshape((nyears,)+space)
            AHW2Fout[:] = AHW2F.reshape((nyears,)+space)
            AHW2Dout[:] = AHW2D.reshape((nyears,)+space)
        yearlyout.close()


    # Split by latitude
    try:
        lats = tmaxnc.variables['lat'][:]
    except KeyError:
        lats = tmaxnc.variables['latitude'][:]
    north = (lats > 0).any()
    south = (lats <= 0).any()

    experiment = f"{elength}{bmax}{e2length}{dmin}"

    HWN_tx, HWF_tx, HWD_tx, CHWN_tx, CHWF_tx, CHWD_tx, AHWN_tx, AHWF_tx, AHWD_tx, AHW1N_tx, AHW1F_tx, AHW1D_tx, AHW2F_tx, AHW2D_tx = \
        split_hemispheres(txexceed, int(bmax), int(elength), int(e2length), int(dmin), north, south, lats)
    HWN_tn, HWF_tn, HWD_tn, CHWN_tn, CHWF_tn, CHWD_tn, AHWN_tn, AHWF_tn, AHWD_tn, AHW1N_tn, AHW1F_tn, AHW1D_tn, AHW2F_tn, AHW2D_tn = \
        split_hemispheres(tnexceed, int(bmax), int(elength), int(e2length), int(dmin), north, south, lats)

    save_yearly(HWN_tx, HWF_tx, HWD_tx, CHWN_tx, CHWF_tx, CHWD_tx, AHWN_tx, AHWF_tx, AHWD_tx, AHW1N_tx, AHW1F_tx, AHW1D_tx, AHW2F_tx, AHW2D_tx, f"tx{str(pcntl*100)}", txpct, experiment, 'tx')
    save_yearly(HWN_tn, HWF_tn, HWD_tn, CHWN_tn, CHWF_tn, CHWD_tn, AHWN_tn, AHWF_tn, AHWD_tn, AHW1N_tn, AHW1F_tn, AHW1D_tn, AHW2F_tn, AHW2D_tn, f"tx{str(pcntl*100)}", tnpct, experiment, 'tn')
    
    return True
