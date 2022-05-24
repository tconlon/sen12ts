import numpy as np
import logging; logging.getLogger().setLevel(logging.INFO); logging.captureWarnings(True)
from contextlib import contextmanager
import sys, os
import geopandas as gpd
from glob import glob

import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
# from appsci_utils.regularization.spatiotemporal_denoise_stack import spatiotemporally_denoise
from appsci_utils.file_io.geotiff import write_geotiff
# from appsci_utils.image_processing.coregistration import coregister_stack
from descarteslabs.catalog import Product, Image, OverviewResampler
from descarteslabs.scenes.geocontext import DLTile
# from sen12ts.data_collection.utils import write_gdal_file
from osgeo import gdal
from osgeo import gdalconst
import shapely

import random
import datetime
from tqdm import tqdm


def write_gdal_file(filename, geotransform, geoproj, data, data_type, format="GTiff", nodata=0, options=None,):
    '''
    This function writes a GeoTIFF file from the numpy array.
    Function provided by Scott Arko at Descartes Labs.
    '''
    # Get dimensions
    if options is None:
        options = []
    if len(data.shape) == 2:
        (x, y) = data.shape
        numbands = 1
    elif len(data.shape) == 3:
        (numbands, x, y) = data.shape
    # Initialize file
    driver = gdal.GetDriverByName(format)
    dst_datatype = data_type
    dst_ds = driver.Create(filename, y, x, numbands, dst_datatype, options)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoproj)

    # Save each band
    if numbands == 1 and len(data.shape) == 2:
        if nodata != "":
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(0, numbands):
            d2 = data[i, :, :]
            if nodata != "":
                dst_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
            dst_ds.GetRasterBand(i + 1).WriteArray(d2)
    # Clear variables
    data = None
    dst_ds = None


def check_catalog_scenes():
    region = 'ca_central_valley_cropland'

    pid = f'sen12ts_{region}_s1'

    auth = dl.Auth()
    product = f"{auth.payload['org']}:{pid}"


    inp_file = gpd.read_file(f'../../data/region_shapefiles/{region}.geojson')
    aoi = inp_file['geometry'].iloc[0]

    years = [2020]
    year_start = np.min(years)
    year_end = np.max(years)

    start_date = f'{year_start}-01-01'
    end_date = f'{year_end}-12-31'

    print(product)

    scenes, ctx = dl.scenes.search(aoi, products=product,
                                   start_datetime=start_date,
                                   end_datetime=end_date,
                                   limit=None)

    dltiles = [i.properties.key.split('_lat')[0].replace('dltile_', '').replace('_', ':') for i in scenes]
    unique_dltiles = np.unique(dltiles)

    print(dltiles)
    print(unique_dltiles)

    unique_dltiles = [DLTile.from_key(i) for i in unique_dltiles]
    print(unique_dltiles)



if __name__ == '__main__':
    check_catalog_scenes()