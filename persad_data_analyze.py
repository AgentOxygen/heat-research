# -*- coding: utf-8 -*-
"""
Updated 4/7/21

@author: Cameron Cummins

Data import, export, and analysis functions for Dr. Persad's Hydroclimate Research.
"""
import xarray
import numpy
import geopandas
import regionmask
import matplotlib.pyplot as pyplot

def cropRegionLatLon(array:xarray.DataArray, lat_min:int, lat_max:int, lon_min:int, lon_max:int) -> xarray.DataArray:
    """
    Parameters
    ----------
    array : xarray.DataArray
        array to crop
    lat_min : int
        minimum latitude
    lat_max : int
        maximum latitude
    lon_min : int
        minimum longitude
    lon_max : int
        maximum longitude

    Returns
    -------
    array : xarray.DataArray()
        list of models
    """
    # Create mask for lattitude values (between min and max)
    lat_mask = (array.lat >= lat_min) & (array.lat <= lat_max)
    # Create mask for longitude values (between min and max)
    lon_mask = (array.lon >= lon_min) & (array.lon <= lon_max)
    return array.where(lat_mask & lon_mask, drop=True)

def getModelsFromNetCDF(path:str, names:list=['']) -> list:
    """
    Parameters
    ----------
    path : string
        path to netCDF to extract models from
    names : list of strings, optional
        list of names of models to extract, by default all found will be extracted. The default is [''].

    Returns
    -------
    models : list of data arrays
        list of models
    """
    # Open metric dataset
    dataset = xarray.open_dataset(path)
    models = []
    
    # Cycle through models in dataset, adding each one to a list
    for model in dataset:
        # Cycle through names and if a model name is found, add it to the list
        for name in names:
            if (name in model):
                models.append(dataset[model])
    
    return models


def getMeanModel(models:xarray.DataArray) -> xarray.DataArray:
    """
    Parameters
    ----------
    models : list of data arrays
        Models to average
        
    Returns
    -------
    mean_model : data array
        Average model of the same structure
    """
    # Create empty structure to sum all of the models over
    models_sum = models[0] * 0
    for model in models:
        models_sum = models_sum + model
    
    # Divide the sum by the number of models to get hte size
    mean_model = models_sum / len(models)
    return mean_model

def getModelsAgreement(mean_model:xarray.DataArray(), models:list) -> xarray.DataArray:
    """
    Parameters
    ----------
    mean_model : data array
        The average of all the models, what each model will be compared to
    models : list of data arrays
        list of models to compare against the mean_model

    Returns
    -------
    model_agreement : data array
        array with the same structure as mean_model containing integer values indicating the number of models 
        that agreed on the sign of the values in mean_model
    """
    # To keep track of how many models agree for each point, create empty array preserving structure
    model_agreement = mean_model * 0
    
    for model in models:
        # Indicating a positive value in the larger, mean model
        mean_pos = (mean_model > 0)
        # Indiciating a negative value in the larger, mean model
        mean_neg = (mean_model < 0)
        # Indiciating a positive value in the individual model in question
        model_pos = (model > 0)
        # Indiciating a negative value in the individual model in question
        model_neg = (model < 0)
        # Count where this model if it agrees with the mean model over the structure of the array (adds 1 to model_agreement')
        agree = model * 0 + 1
        # Don't count it if it disagrees (adds 0 to 'model_agreement')
        disagree = model * 0
        # The individual model and mean model agree where both are positive in change or both are negative in change
        models_agree = numpy.logical_or(numpy.logical_and(model_pos, mean_pos),numpy.logical_and(model_neg, mean_neg))
        # The agreement array has the same structure as the models, but only keeps count of how many models agree for each point
        model_agreement = model_agreement + xarray.where(models_agree, agree, disagree)
        
    return model_agreement

def getPercentChangeModels(perc_models:list, perc_historical_models:list) -> list:
    """
    Parameters
    ----------
    perc_models : list of data arrays
        projection models to get percentage changes for
    perc_historical_models : list of data arrays
        historical models to compare 'models' to index by index
        
    Returns
    -------
    percent_change_ret : list of data arrays
       each model with their respective percentage difference
    """
    percent_change_ret = [];
    
    for index, model in enumerate(perc_models):
        hist_model = perc_historical_models[index]
        percent_change_ret.append((model.where(hist_model != 0) - hist_model.where(hist_model != 0)) / hist_model * 100)
            
    return percent_change_ret

def getDifferenceModels(perc_models:list, perc_historical_models:list) -> list:
    """
    Parameters
    ----------
    perc_models : list of data arrays
        projection models to get ratios for
    perc_historical_models : list of data arrays
        historical models to compare 'models' to index by index
    Returns
    -------
    relative_ratio_ret : list of data arrays
       each model with their respetive relative ratios
    """
    difference_ret = [];
    
    for index, model in enumerate(perc_models):
        hist_model = perc_historical_models[index]
        difference_ret.append(model - hist_model)
            
    return difference_ret

def getRelativeRatioModels(perc_models:list, perc_historical_models:list) -> list:
    """
    Parameters
    ----------
    perc_models : list of data arrays
        projection models to get differences for
    perc_historical_models : list of data arrays
        historical models to compare 'models' to index by index
    Returns
    -------
    relative_ratio_ret : list of data arrays
       each model with their respetive differences
    """
    relative_ratio_ret = [];
    
    for index, model in enumerate(perc_models):
        hist_model = perc_historical_models[index]
        relative_ratio_ret.append(model.where(hist_model != 0) / hist_model * 100)
            
    return relative_ratio_ret

def getDataOnRegion(data:xarray.DataArray, shapefile:geopandas.GeoDataFrame):
    """
    sourced from https://www.guillaumedueymes.com/post/shapefiles_country/
   
    Parameters
    ----------
    data : xarray.DataArray
        array containing complete geographical data
    shapefile : geopandas.GeoDataFrame
        shapefile data defining specific sub-region to the full data

    Returns
    -------
    masked_data : xarray.DataArray
        array containing data specific to the region defined by the shapefile

    """
    my_list = list(shapefile['OBJECTID'])
    shapefile = shapefile[(numpy.logical_not(numpy.isnan(my_list)))]
    my_list = list(shapefile['OBJECTID'])
    my_list_unique = set(list(shapefile['OBJECTID']))
    indexes = [my_list.index(x) for x in my_list_unique]
    if len(my_list) == 1:
        indexes = [x - 1 for x in my_list_unique]
    basin_regions = regionmask.Regions(name = 'Basin_Name', numbers = indexes, names = shapefile.Basin_Name[indexes], abbrevs = shapefile.Basin_Name[indexes], outlines = list(shapefile.geometry.values[i] for i in range(0, shapefile.shape[0])))
    masked_data = basin_regions.mask(data, lat_name='lat', lon_name='lon')
    
    return masked_data

def outputFigureOfRegion(data:xarray.DataArray, shapefile:geopandas.GeoDataFrame, output_path:str) -> None:
    """
    Outputs image of figure with shapefile overlay (used for debugging)
   
    Parameters
    ----------
    data : xarray.DataArray
        array containing complete geographical data
    shapefile : geopandas.GeoDataFrame
        shapefile data defining specific sub-region to the full data
    output_path : string
        Path to output file to

    Returns
    -------
    None
    """
    masked_data = getDataOnRegion(data, shapefile)
    pyplot.plot()
    ax = pyplot.axes()
    shapefile.plot(ax = ax, alpha = 0.8, facecolor = 'none')
    
    pyplot.savefig(output_path)
