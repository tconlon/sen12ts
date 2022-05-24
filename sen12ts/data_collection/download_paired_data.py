import numpy as np
import logging; logging.getLogger().setLevel(logging.INFO); logging.captureWarnings(True)
import os
import geopandas as gpd
from glob import glob
from tqdm import tqdm
import descarteslabs as dl
from descarteslabs.scenes import SceneCollection
from descarteslabs.catalog import Product, Image, OverviewResampler
import datetime
import time
from osgeo import gdal
from google.cloud import storage as gcp_storage
from descarteslabs import Auth

auth = Auth()
print(f'Signed into DL account: {auth.payload["email"]}')

class S1_S2_LULC_Generator():
    '''
    This data generator creates paired  sets containing S1, S2, and LULC imagery.
    
    The input s1 data contains a timeseries of vh and vv S1 backscatter, and 
    insar coherence layers. SRTM altitude and slope layers are  appended to the
    end of the s1 input image, along with a shadow/layover mask. The S1 SAR input data
    is coregistered and presented to log space.
    
    The S2 data contains the Sentinel-2 L2A spectral bands

    The LULC layer contains the land-use/land cover classification from the location-specific source.
        - USDA CDL: Iowa and California
        - SIGPAC: Catalonia
        - ESA WorldCover: Ethiopia, Uganda, Sumatra

    Below is a list of the input and output layers that result from the
    current object configuration. 
    
    S1 layers:
    0: image t, band vh
    1: image t, band vv
    2: image t, band coherence
    3: image t, band phase
    4: image t-1, band vh
    5: image t-1, band vv
    6: image t-1, band coherence
    7: image t-1, band phase
    8: image t-2, band vh
    9: image t-2, band vv
    10: image t-2, band coherence
    11: image t-2, band phase
    12: image t-3, band vh
    13: image t-3,band vv
    14: image t-3, band coherence
    15: image t-3, band phase
    16: local incidence angle
    17: srtm slope
    18: shadow/layover mask
    
    S2 layers

    0: image t, coastal-aerosol
    1: image t, blue
    2: image t, green
    3: image t, red
    4: image t, red-edge
    5: image t, red-edge-2
    6: image t, red-edge-3
    7: image t, nir
    8: image t, red-edge-4
    9: image t, water-vapor
    10: image t, swir1
    11: image t, swir2
    12: cloud mask

    LULC layers

    Paired s1, s2, and LULC images are saved with matching file names in s1/, s2/,
    and labels/ directories within the basedir.
    
    To download a new set of imagery, run this .py file from terminal. 

    Images are only downloaded if there are less than 1% invalid (masked, zero-valued,
    or NaN) pixels within the S1 and S2 stacks.
    '''
    
    def __init__(self, deploy_virtually=False):
        # Define parameters for download tile
        self.resolution = 10 # Spatial resolution in meters
        self.tilesize = 256 # Number of pixels in spatial dimensions for imagery
        self.pad = 0 # Number of pixels for spatial padding of imagery
        self.close_day_threshold = 3 # Number of days apart S1 and S2 imagery can be to both be cconsidered at time t'=t
        self.num_total_s1_timesteps = 4 # Number of S1 timesteps in S1 timeseries (12 days apart)
        self.seasons = ['spring', 'summer', 'fall', 'winter'] # Define seasons for imagery collection
        self.years = [2020] # Year for imagery collection
        self.max_images_per_tile_season = 4 # Max number of images per tile per season
        self.max_invalid_count_per_tile_season = 12 # Max number of imvalid images per tile before skipping tile
        self.saved_dltiles = 0 # Hard coded value for # of previously saved tiles
        self.dltiles_to_download = 5 # Number of tiles over which to download imagery.

        # Define parameters for filtering S2 images
        self.max_s2_cloud_frac = 0.05
        self.acceptible_missing_pixels = (self.tilesize * self.tilesize) * 0.01

        # Define strings for saving
        self.area_name = 'california'
        self.base_dir = f'/Volumes/sel_external/sen12ts/paired_imagery/{self.area_name}_{self.max_s2_cloud_frac}'
        self.file_ext = 'lat_{}_lon_{}_s1date_{}_s2date_{}_s1track_{}_s1lookpass_{}.tif'
        self.s1_pid = f'sen12ts_{self.area_name}_s1'
        self.s2_pid = f'sen12ts_{self.area_name}_s2'

        # Specify whether saved images should be saved locally, uploaded to the DL catalog, or uploaded to GCP.
        self.download_locally = False
        self.upload_to_catalog = False
        self.upload_to_gcp = True

        # Define bands for download
        self.s2_bands = ['coastal-aerosol', 'blue', 'green', 'red', 'red-edge', 'red-edge-2', 'red-edge-3',
                         'nir', 'red-edge-4', 'water-vapor', 'swir1', 'swir2', 'scl']
        self.insar_bands = 'coherence phase alpha'
        self.rtc_bands = 'factor localinc mask'
        self.s1_bands = 'vh vv alpha'

        # Whether script is being deployed virtually or not.
        self.deploy_virtually = deploy_virtually

        # Create dirs if they DNE
        if not self.deploy_virtually:
            # Collect the aoi, count the previously downloaded tiles, find the new dtiles,
            # and find the corresponding imagery

            if self.download_locally:
                save_dir = self.base_dir
            elif self.upload_to_gcp:
                save_dir = 'tmp'

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(f"{save_dir}/s1/"):
                os.makedirs(f"{save_dir}/s1")
            if not os.path.exists(f"{save_dir}/s2/"):
                os.makedirs(f"{save_dir}/s2")

            self.define_aoi()
        
    def define_aoi(self):
        '''
        This function defines the shapefile with an saved geojson file.
        ''' 
        inp_file = gpd.read_file(f'../../data/shapefiles/final_shapefiles_renamed/{self.area_name}.geojson')
        self.aoi = inp_file['geometry'].iloc[0]
        
    def upload_to_catalog_func(self, dltile, image_type, file_ext, image_array):
        '''
        This function uploads the saved imagery to the DL catalog.
        '''

        if image_type == 's1':
            pid = self.s1_pid
        elif image_type == 's2':
            pid = self.s2_pid
        
        auth = dl.Auth()
        pid = f"{auth.payload['org']}:{pid}"
        product = Product.get(pid)

        s2_date = file_ext.split('_')[-5]

        img_id = f"{pid}:{file_ext}"
        print(f'Image id for the catalog: {img_id}')

        image = Image(product=product, name=file_ext, id=img_id)
        image.acquired = s2_date
        image.geotrans = dltile.geotrans
        image.cs_code = dltile.crs

        upload = image.upload_ndarray(
            image_array,
            overviews=[2, 4],
            overview_resampler=OverviewResampler.MODE,
            overwrite=True
        )

        upload.wait_for_completion()

    def last_nonmask(self, arr, axis, invalid_val=-1):
        '''
        This function gets the last non-masked pixel in a 4 dimensional imagery stack with dimensions:
        (timesteps, cols, rows, bands). The last non-masked pixel is selected along the first (i.e. temporal) axis.
        '''
        # Get non-masked values
        nonmask = ~np.ma.getmaskarray(arr)

        # Find last indices along axis=0 with non-masked value
        val_ix = arr.shape[axis] - np.flip(nonmask, axis=axis).argmax(axis=axis) - 1

        # Extract values of these last indices
        test = np.take_along_axis(arr, val_ix[None], axis=0)[0]

        # Return values of these last indices + invalid val wherever theres no non-masked pixels
        return np.where(nonmask.any(axis=axis), test, invalid_val)

    def upload_to_gcp_func(self, save_dir, file_ext, output_stack):
        '''
        This functions uploads the saved imagery sets to a GCP bucket called 'sen12ts'
        '''
        tmp_dir = 'tmp'

        # Create dirs for local download if needed
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(f"{tmp_dir}/{save_dir}/"):
            os.makedirs(f"{tmp_dir}/{save_dir}")

        save_str_gcp = f'{tmp_dir}/{save_dir}/{file_ext}'

        ## Note this will only work for S2 + LULC as the
        if save_dir == 's2' or save_dir == 'labels':
            data_type = gdal.GDT_Int16
        else:
            data_Type = gdal.GDT_Float32

        # Temporarily save imagery to local
        self.write_gdal_file(
            filename=save_str_gcp,
            geotransform=self.dltile.geotrans,
            geoproj=self.dltile.wkt,
            data=output_stack,
            data_type=data_type,
            format="GTiff",
            nodata=-3000,
            options=None,
        )

        # Extract permissions from DL Storage and specify GCP save location
        storage = dl.Storage()
        storage.get_file("my_creds_sen12ts.json", "creds.json")
        storage_client = gcp_storage.Client.from_service_account_json("creds.json")
        bucket_name = 'sen12ts'
        destination_blob_name = f'paired_imagery/{self.area_name}/{save_dir}/{file_ext}'

        # Transfer local file to GCP
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(save_str_gcp)

        print(f"File {file_ext} uploaded to GCP")

        ## Remove local file
        os.remove(save_str_gcp)
        os.remove("creds.json")

    def write_gdal_file(self, filename, geotransform, geoproj, data, data_type,
                        format="GTiff", nodata=0, options=None, ):
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

        print('GDAL projection')
        print(dst_ds.GetProjection())

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


        dst_ds.FlushCache()
        data = None
        dst_ds = None

    def return_dates_for_season(self, season):
        '''
        This function specifies the dates that define the season
        '''
        
        self.start_date = []
        self.end_date = []
        
        for year in self.years:
            if season == 'spring':
                self.start_date.extend([f'{year}-03-01'])
                self.end_date.extend([f'{year}-05-31'])

            elif season == 'summer':
                self.start_date.extend([f'{year}-06-01'])
                self.end_date.extend([f'{year}-08-31'])

            elif season == 'fall':
                self.start_date.extend([f'{year}-09-01'])
                self.end_date.extend([f'{year}-11-30'])

            elif season == 'winter': 
                self.start_date.extend([f'{year}-01-01', f'{year}-12-01'])
                self.end_date.extend([f'{year}-02-28', f'{year}-12-31'])
        
    def invalid_px_ct(self, array):
        '''
        This function counts the invalid pixels in an array (either NaN, zero valued, or masked)
        '''
        # Helper function to determine how many invalid pixels are within an array
        # Array is derived:visual_cloud_mask band
        nan_mask = np.isnan(array).astype(np.int16)
        zero_mask = (array == 0).astype(np.int16)
        ma_mask = np.ma.getmaskarray(array).astype(np.int16)

        nan = np.count_nonzero(nan_mask)
        zero_val = np.count_nonzero(zero_mask)
        masked = np.ma.count_masked(array)

        invalid_mask = np.stack((nan_mask, zero_mask, ma_mask), axis=-1)


        return nan + zero_val + masked, invalid_mask

    def load_dltiles(self):
        '''
        This function loads the DL tiles for an specified AOI
        It also checks to see which tiles have imagery downloaded for them, and removes those tiles
        from the list of tiles to be used for imagery collection.
        '''
        print('Loading tiles for AOI')

        # Load the DLTiles to download from a local text file
        load_from_txt_file = False

        # Determine which DLTiles already have imagery downloaded for them, either catalog or local
        find_dltiles_in_catalog = False
        find_dltiles_in_local = False
        find_imgs_in_catalog = False

        # Load saved dltiles to limit future download DLTiles
        load_saved_dltiles = False

        saved_dltiles_loc = 'local'# or 'catalog'

        dltiles_dir = '../../data/dltile_txt_files'

        # If statement for loading already-downloaded tiles from a txt file
        if load_from_txt_file:

            txt_file = glob(f'{dltiles_dir}/all_dltiles/{self.area_name}*.txt')[0]
            all_dltiles = []

            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    all_dltiles.append(line.replace('\n', ''))
        else:
            all_dltiles = dl.scenes.DLTile.from_shape(shape=self.aoi, resolution=self.resolution,
                                                      tilesize=self.tilesize, pad=self.pad,
                                                      keys_only=True)

        # If statement for loading already-downloaded tiles from the DL catalog
        if find_dltiles_in_catalog:

            print('Check for existing uploaded DLTiles')
            pid = f'sen12ts_{self.area_name}_s1'

            auth = dl.Auth()
            product = f"{auth.payload['org']}:{pid}"

            year_start = np.min(self.years)
            year_end = np.max(self.years)
            start_date = f'{year_start}-01-01'
            end_date = f'{year_end}-12-31'

            scenes, ctx = dl.scenes.search(self.aoi, products=product,
                                           start_datetime=start_date,
                                           end_datetime=end_date,
                                           limit=None)

            dltiles = [i.properties.key.split('_lat')[0].replace('dltile_', '').replace('_', ':') for i in scenes]
            unique_dltiles = np.unique(dltiles)

            print(f'Number of saved dltiles: {len(unique_dltiles)}')
            self.saved_dltiles = len(dltiles)
            dirtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            txt_outfile = f'{dltiles_dir}/downloaded_dltiles/' \
                          f'{self.area_name}_{dirtime}_catalog_saved_tiles.txt'


            with open(txt_outfile, 'wt', encoding='utf-8') as myfile:
                myfile.write('\n'.join(unique_dltiles))

        # If statement for finding already-downloaded images in the DL catalog
        if find_imgs_in_catalog:

            print('Check for existing uploaded imgs')
            pid = f'sen12ts_{self.area_name}_s1'

            auth = dl.Auth()
            product = f"{auth.payload['org']}:{pid}"

            year_start = np.min(self.years)
            year_end = np.max(self.years)

            start_date = f'{year_start}-01-01T00:00:00'
            end_date = f'{year_end}-12-31T23:59:59'

            scenes, ctx = dl.scenes.search(self.aoi, products=product,
                                           start_datetime=start_date,
                                           end_datetime=end_date,
                                           limit=None)

            dltiles = [i.properties.key for i in scenes]

            print(f'Number of saved images: {len(dltiles)}')

            dirtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            txt_outfile = f'../../data/imgs_on_catalog/' \
                          f'{self.area_name}_catalog_stored_s1_imgs_{dirtime}.txt'

            with open(txt_outfile, 'wt', encoding='utf-8') as myfile:
                myfile.write('\n'.join(dltiles))

        # If statement for finding already-downloaded dltiles in a folder contains saved .tif files
        if find_dltiles_in_local:

            s1_dir = f"{self.base_dir}/s1"
            saved_files = glob(f'{s1_dir}/*.tif')
            dltiles = np.unique([i.split('/')[-1].split('_lat')[0].replace('dltile_', '').replace('_', ':')
                                       for i in saved_files])


            unique_dltiles = np.unique(dltiles)

            print(f'Number of saved dltiles: {len(unique_dltiles)}')
            self.saved_dltiles = len(dltiles)


            dirtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            txt_outfile = f'{dltiles_dir}/downloaded_dltiles/' \
                          f'{self.area_name}_{dirtime}_local_saved_tiles.txt'

            with open(txt_outfile, 'wt', encoding='utf-8') as myfile:
                myfile.write('\n'.join(unique_dltiles))


        prev_saved_files = []

        # If statement for whether already-downloaded DLTiles should be removed from list of DLTiles for download
        if load_saved_dltiles:
            txt_file = sorted(glob(f'{dltiles_dir}/downloaded_dltiles/{self.area_name}*{saved_dltiles_loc}*.txt'))[-1]
            print(f"Loading already downloaded DLTiles from {txt_file}")
            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    prev_saved_files.append(line.replace('\n', ''))

            valid_dltiles = [i for i in all_dltiles if i not in prev_saved_files]

        else:
            valid_dltiles = all_dltiles

        print(f'Total number of dltiles in the region for download: {len(all_dltiles)}')
        print(f'Valid (non-downloaded) dltiles in the region: {len(valid_dltiles)}')

        # Shuffle and return DLTiles
        np.random.seed(7)
        np.random.shuffle(valid_dltiles)

        print(f'Downloading {len(valid_dltiles)} dltiles in the region')

        return valid_dltiles

    def find_seasonal_scene_collections(self, dltile, season):
        '''
        Function for returning imagery collections by season
        '''

        self.return_dates_for_season(season)

        # Define empty SceneCollection objects to populate
        s1_scenes_sc        = SceneCollection()
        s1_rtc_scenes_sc    = SceneCollection()
        insar_scenes_sc     = SceneCollection()
        s2_scenes_sc        = SceneCollection()
        s2_cloud_scenes_sc  = SceneCollection()

        # Define product IDs
        s1_product = 'sentinel-1:sar:sigma0v:v1'
        insar_product = 'sentinel-1:insar:ifg:v0'
        rtc_product = 'sentinel-1:sar:rtc:v0'
        s2_product = 'esa:sentinel-2:l2a:v1'
        s2_cloud_product = 'sentinel-2:L1C:dlcloud:v1'

        # Collect imagery scenes by season
        for ix, start_date in enumerate(self.start_date):

            end_date = self.end_date[ix]
        
            # Adjust S1 start date
            s2_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            s1_start_date = s2_start_date - datetime.timedelta(days=self.num_total_s1_timesteps * 12)

            # Get all available Sentinel-1 scenes
            s1_scenes, ctx = dl.scenes.search(
                dltile,
                products=s1_product,
                start_datetime=s1_start_date,
                end_datetime=end_date,                             
                limit=None,
            )
            
            # Get radiometric corrections for S1 backscatter
            s1_rtc_scenes, ctx = dl.scenes.search(
                dltile,
                products=rtc_product,
                limit=None,
            )

            # Get insar interferogram scenes
            insar_scenes, ctx = dl.scenes.search(
                dltile,
                products=insar_product,
                start_datetime=s1_start_date,
                end_datetime=end_date,                                                         
                limit=None,
            )

            # Get clear Sentinel-2 scenes by filtering by a low cloud fraction
            s2_scenes, ctx = dl.scenes.search(
                dltile,
                products=s2_product,
                start_datetime=start_date,
                end_datetime=end_date,
                cloud_fraction=self.max_s2_cloud_frac,
                limit=None,
            )

            s2_cloud_scenes, ctx = dl.scenes.search(
                dltile,
                products=s2_cloud_product,
                start_datetime=start_date,
                end_datetime=end_date,
                limit=None,
            )

            # Extend the SceneCollection objects
            s1_scenes_sc.extend(s1_scenes)
            s1_rtc_scenes_sc.extend(s1_rtc_scenes)
            insar_scenes_sc.extend(insar_scenes)
            s2_scenes_sc.extend(s2_scenes)
            s2_cloud_scenes_sc.extend(s2_cloud_scenes)


        print(f'Maximum number of S1 scenes: {len(s1_scenes_sc)}')
        print(f'Maximum number of InSAR scenes: {len(insar_scenes)}')
        print(f'Maximum number of S2 scenes: {len(s2_scenes_sc)}')

        ## Apply groupby functions and convert to lists

        # Group by track (orbit #) and date
        sar_grouping = lambda x: (x.properties["id"].split(":")[-1].split("-")[4], 
                                  x.properties["id"].split(":")[-1][:10])

        # Group by track (orbit #)
        rtc_grouping = lambda x: x.properties["id"].split(":")[-1].split("-")[0]

        # Extract days between SAR passes for InSAR creation
        time_delta = lambda x: x.properties["id"].split(":")[-1].split("-")[-1]

        # S2 group by date
        s2_grouping = lambda x: x.properties["id"].split(":")[-1].split("_")[2].split('T')[0]

        # S2 cloud group by date
        s2_cloud_grouping = lambda x: x.properties["id"].split(":")[-1].split("_")[0]

        # Group by track (orbit #), date (second pass), time between first and second passes
        insar_grouping = lambda x: (x.properties["id"].split(":")[-1].split("-")[4],
                                    x.properties["id"].split(":")[-1][:10],
                                    time_delta(x))

        # Convert to lists
        s1_scenes_sc       = list(s1_scenes_sc.groupby(sar_grouping))
        s1_rtc_scenes_sc   = list(s1_rtc_scenes_sc.groupby(rtc_grouping))
        insar_scenes_sc    = list(insar_scenes_sc.groupby(insar_grouping))                  
        s2_scenes_sc       = list(s2_scenes_sc.groupby(s2_grouping))
        s2_cloud_scenes_sc = list(s2_cloud_scenes_sc.groupby(s2_cloud_grouping))


        return s1_scenes_sc, s1_rtc_scenes_sc, insar_scenes_sc, s2_scenes_sc, s2_cloud_scenes_sc

    def find_imagery(self, dltile, season):
        '''
        This function organizes the S1 and S2 imagery scenes into valid pairs for the SEN12TS dataset
        '''

        print(f'Finding valid paired imagery for dltile: {dltile.key}, season: {season}')
        s1_scenes, s1_rtc_scenes, insar_scenes, s2_scenes, s2_cloud_scenes = \
        self.find_seasonal_scene_collections(dltile, season)
        
        base_date = datetime.datetime(2010, 1, 1) 
        unique_tracks = np.unique([track for track, _ in s1_rtc_scenes])
        
        s1_scenes_dict = {}
        s1_series_dict = {}
        insar_scenes_dict = {}
        insar_series_dict = {}
        
        ## Determine sets of S1 and inSAR data collected in valid series of self.num_total_s1_timesteps images 12
        # days apart
        for track in unique_tracks:
                        
            s1_scenes_dict[track] = [(date, (datetime.datetime.strptime(date, "%Y-%m-%d") - base_date).days, scenes) for 
                                    (track_d, date), scenes in s1_scenes if track_d == track]
            insar_scenes_dict[track] = [(second_date, (datetime.datetime.strptime(second_date, "%Y-%m-%d") - base_date).days, scenes)
                                        for (track_d, second_date, time_delta), scenes 
                                        in insar_scenes if (track_d == track and int(time_delta) == 12)]
        
            s1_date_nos_list = [day_no for ix, (date, day_no, scenes) in enumerate(s1_scenes_dict[track])]
            insar_date_nos_list = [day_no for ix, (date, day_no, scenes) in enumerate(insar_scenes_dict[track])]

            for (date, day_no, scenes) in s1_scenes_dict[track]:
                series_day_nos = [day_no + 12*i for i in range(self.num_total_s1_timesteps)]

                if set(series_day_nos) <= set(s1_date_nos_list) and set(series_day_nos) <= set(insar_date_nos_list):
                    # Enough valid S1 images and inSAR images to save the series
                    s1_scene_ixs = [i for i, j in enumerate(s1_date_nos_list) if j in series_day_nos]
                    insar_scene_ixs = [i for i, j in enumerate(insar_date_nos_list) if j in series_day_nos]
                    
                    
                    if track not in s1_series_dict.keys():
                        s1_series_dict[track] = [[s1_scenes_dict[track][ix] for ix in reversed(s1_scene_ixs)]]
                        insar_series_dict[track] = [[insar_scenes_dict[track][ix] for ix in reversed(insar_scene_ixs)]]
                    else:
                        s1_series_dict[track].append([s1_scenes_dict[track][ix] for ix in reversed(s1_scene_ixs)])
                        insar_series_dict[track].append([insar_scenes_dict[track][ix] for ix in reversed(insar_scene_ixs)])
            
        # Define lists to add valid S1 and S2 imagery scenes to
        self.s1_scenes_list = []
        self.s1_rtc_scenes_list = []
        self.insar_scenes_list = []
        self.s2_scenes_list = []
        self.s2_cloud_scenes_list = []

        
        s2_day_nos = np.array([(datetime.datetime.strptime(date, "%Y%m%d") - base_date).days for
                               date, scenes in s2_scenes])

        # Pair S1 and S2 imagery scenes if they're <=3 days apart
        for key in s1_series_dict.keys():
            for ix, series in enumerate(s1_series_dict[key]):                
                s1_final_day_no = series[0][1]
                s2_day_diff = np.abs(s2_day_nos - s1_final_day_no)
                
                s2_near_images_ix = np.argwhere(s2_day_diff <= self.close_day_threshold)                
                if len(s2_near_images_ix) > 0:
                    for s2_ix in s2_near_images_ix[0]:
                        s2_date = datetime.datetime.strptime(s2_scenes[s2_ix][0], "%Y%m%d").strftime("%Y-%m-%d")
                        print(f's2_date: {s2_date}')

                        self.s1_scenes_list.append(s1_series_dict[key][ix])
                        self.s1_rtc_scenes_list.extend([(track, scenes) for track, scenes in s1_rtc_scenes if track==key])
                        self.insar_scenes_list.append(insar_series_dict[key][ix])
                        self.s2_scenes_list.append(s2_scenes[s2_ix])

                        s2_cloud_mask = [s2_cloud_collection for s2_cloud_date, s2_cloud_collection in \
                                         s2_cloud_scenes if s2_cloud_date == s2_date]
                        if len(s2_cloud_mask) > 0:
                            self.s2_cloud_scenes_list.append(s2_cloud_mask[0])
                        else:
                            self.s2_cloud_scenes_list.append(None)

    def download_imagery_for_dltile(self, dltile, season):
        '''
        This function downloads the S1, S2, and LULC imagery for a valid SEN12TS imagery set.
        '''

        print(f'Downloading paired imagery for dltile: {dltile.key}')
        # Shuffle the lists of scenes so that we're not taking imagery from the same date first
        # This matters when we limit the number of paired images to download per tile
        self.tile_download = False

        if len(self.s1_scenes_list) > 0 and len(self.s2_scenes_list) > 0:
            list_zip = list(zip(self.s1_scenes_list, self.s1_rtc_scenes_list, self.insar_scenes_list,
                                self.s2_scenes_list, self.s2_cloud_scenes_list))
            np.random.shuffle(list_zip)
            self.s1_scenes_list, self.s1_rtc_scenes_list, self.insar_scenes_list, self.s2_scenes_list,  \
            self.s2_cloud_scenes_list = zip(*list_zip)

            s1_dates_for_download = []
            s2_dates_for_download = []


            downloaded_imgs_for_tile = 0
            invalid_count_per_tile_season = 0

            print(f'Max potential number of images to download for {season}: {len(self.s1_scenes_list)}')
            try:
                for ix, s1_series in enumerate(self.s1_scenes_list):
                    print(f'Total number of images downloaded for tile {dltile.key} and season {season}:'
                          f' {downloaded_imgs_for_tile}')

                    print(f'S1 scenes: {self.s1_scenes_list[ix]}')
                    print(f'InSAR scenes: {self.insar_scenes_list[ix]}')
                    print(f'S2 scenes: {self.s2_scenes_list[ix]}')
                    print(f'S2 cloud scenes: {self.s2_cloud_scenes_list[ix]}')

                    if downloaded_imgs_for_tile == self.max_images_per_tile_season:
                        print('Maximum images per seasons reached')
                        break

                    if invalid_count_per_tile_season == self.max_invalid_count_per_tile_season:
                        print('Maximum invalid count per tile reached')
                        break

                    # Define empty stacks to save imagery into
                    s1_stack = np.full([self.num_total_s1_timesteps, dltile.tilesize,
                                                dltile.tilesize, 3], np.nan)
                    insar_stack = np.full([self.num_total_s1_timesteps, dltile.tilesize,
                                                dltile.tilesize, 3], np.nan)


                    ## Mosaic images and check all S2 pixels are valid

                    # Check S2 imagery to make sure no pixels are masked out
                    s2_scenes = self.s2_scenes_list[ix][-1].sorted(lambda s: s.properties.cloud_fraction,
                                                 reverse=True)
                    s2_cloud_scenes = self.s2_cloud_scenes_list[ix].sorted(lambda s: s.properties.cloud_fraction,
                                                reverse=True)



                    print(f'Stacking S2 bands, {len(s2_scenes)}')
                    t = time.time()
                    s2_stack, s2_raster_info = s2_scenes.mosaic(self.s2_bands,
                                                                ctx=self.dltile,
                                                                bands_axis=0,
                                                                raster_info=True,
                                                                mask_nodata=False,
                                                                mask_alpha=False,
                                                                data_type='Float64')
                    elapsed = time.time() - t
                    print(f'time for s2 stack: {elapsed}')

                    # Convert to ints and clip
                    s2_stack[0:-1] = s2_stack[0:-1] * 10000
                    s2_stack = np.clip(s2_stack, 0, 10000).astype(np.int16)

                    # Find matching cloud cover layers
                    full_cloud_mask_list = []
                    if len(s2_cloud_scenes) > 0:
                        print('Existing cloud mask')

                        # Iterate through the S2 L2A scenes and extract the cloud cover scenes that match based on the S2 key
                        for l2a_ix in range(len(self.s2_scenes)):
                            l2a_granule_key = self.s2_scenes[l2a_ix].properties.key

                            matching_dlcloud_scene = [sc for sc in self.s2_cloud_scenes if
                                                      (sc.properties.key.split('_')[1] in l2a_granule_key and
                                                       sc.properties.key.split('_')[-2] in l2a_granule_key)]
                            if len(matching_dlcloud_scene) > 0:
                                print('Matching cloud cover scene')
                                print(matching_dlcloud_scene[0].properties.key)
                                cloud_mask = matching_dlcloud_scene[0].ndarray(bands='valid_cloudfree', ctx=self.dltile,
                                                                               bands_axis=0) / 255
                                full_cloud_mask_list.append(cloud_mask)
                            else:
                                print('No match')
                                full_cloud_mask_list.append(np.ones((1, self.tilesize, self.tilesize)))

                        # Stack the rasterized cloud cover scenes, and take the last one in the stack, similar to mosaicking
                        full_cloud_mask_stack = np.ma.concatenate(full_cloud_mask_list, axis=0)
                        s2_cloud_mosaic = self.last_nonmask(arr=full_cloud_mask_stack, axis=0, invalid_val=-3000)[None]

                        print(f'Fraction of image covered by DL cloud mask: {np.mean(s2_cloud_mosaic)}')



                    else:
                        # If no cloud cover scenes, all pixels are valid
                        s2_cloud_mosaic = np.ones((1, self.tilesize, self.tilesize))

                    # Add cloud layer/pixel validity mask to back of s2 stack
                    s2_output_stack = np.concatenate((s2_stack, s2_cloud_mosaic), axis=0)

                    s2_invalid_px_ct = np.ma.count_masked(s2_stack[0]) + np.count_nonzero(s2_stack!=1)
                    print(f'Total invalid pixel count by band: {s2_invalid_px_ct}')

                    # At them moment, just use invalid pixel count of the last band. Shold be identical across bands
                    if s2_invalid_px_ct[-1] <= self.acceptible_missing_pixels:

                        print('S2 is valid. Collect S2 stack and check S1-based imagery')
                        final_s2_mask = np.ma.getmask(s2_output_stack).astype(np.int16)
                        # Fill invalid, masked pixels with -3000s
                        s2_output_stack[np.where(final_s2_mask)] = -3000
                        print(f'Num masked S2 pixels, bandwise: {np.count_nonzero(s2_stack==-3000, axis=(1,2))}')

                        # Retrieve S1 backscatter images
                        for jx in range(self.num_total_s1_timesteps):
                            s1_stack[jx], s1_raster_info = self.s1_scenes_list[ix][jx][-1].mosaic(bands=self.s1_bands,
                                                                                                  ctx=dltile,
                                                                                                  bands_axis=-1,
                                                                                                  raster_info=True,
                                                                                                  mask_alpha=True,
                                                                                                  scaling="physical")

                        # Find radiometric correction factor
                        rtc_stack = self.s1_rtc_scenes_list[ix][-1].mosaic(bands=self.rtc_bands,
                                                                           ctx=dltile,
                                                                           bands_axis=-1,
                                                                           scaling="physical")

                        # Check all pixels are valid, 7 is 'valid' for RTC alpha band
                        # print(f'Check s1 invalid pixels ')
                        invalid_s1, invalid_s1_mask = self.invalid_px_ct(s1_stack[..., -1])
                        invalid_rtc, invalid_rtc_mask = self.invalid_px_ct(rtc_stack[..., -1] == 7)

                        s1_invalid_px_ct = invalid_s1 + invalid_rtc

                        print(f'Invalid S1 px: {invalid_s1}')
                        print(f'Invalid RTC px: {invalid_rtc}')

                        # Flatten S1 mask along last, bandwise axis
                        invalid_s1_mask = np.max(invalid_s1_mask, axis=(0,-1))
                        invalid_rtc_mask = np.max(invalid_rtc_mask, axis=-1)
                        print(f'invalid RTC mask shape: {invalid_rtc_mask.shape}')

                        invalid_s1_mask_combined = np.max(np.stack((invalid_s1_mask, invalid_rtc_mask), axis=-1), axis=-1)


                        if s1_invalid_px_ct <= self.acceptible_missing_pixels:

                            print('S1 is valid. Drop backscatter alpha and add radiometric correction. Check insar imagery.')
                            s1_stack_rtc = s1_stack[..., 0:2] + np.tile(rtc_stack[None, ..., 0:1], (4,1,1,2))
                            ## Convert to decibels
                            # Integer values between 1-4095 scale linearly to -40 to 30 dB
                            # s1_stack_rtc = (70 * (s1_stack_rtc - 1)/4094) - 40

                            # Fill masked values with -3000
                            np.ma.set_fill_value(s1_stack_rtc, -3000)
                            s1_stack_rtc = s1_stack_rtc.filled()

                            print(f'Num masked values in S1 rtc stack: {np.count_nonzero(s1_stack_rtc==-3000)}')

                            # S1 backscatter is valid. Check insar imagery
                            for jx in range(self.num_total_s1_timesteps):
                                insar_stack[jx], insar_info = self.insar_scenes_list[ix][jx][-1].mosaic(bands=self.insar_bands,
                                                                                            ctx=dltile,
                                                                                            bands_axis=-1,
                                                                                            raster_info=True,
                                                                                            mask_alpha=True,
                                                                                            scaling="physical")

                            insar_invalid_px_ct, invalid_insar_mask = self.invalid_px_ct(insar_stack[..., -1])

                            invalid_insar_mask = np.max(invalid_insar_mask, axis=(0, -1))
                            invalid_radar_mask_combined = np.stack((invalid_s1_mask_combined, invalid_insar_mask), axis=-1)
                            valid_radar_mask_combined = 1 - np.max(invalid_radar_mask_combined, axis=-1)

                            print(f'Total invalid radar pixels: {np.count_nonzero(valid_radar_mask_combined==0)}')


                            if insar_invalid_px_ct <= self.acceptible_missing_pixels:
                                ## All layers are valid! Save image triplets

                                # Fill masked values with -3000
                                np.ma.set_fill_value(insar_stack, -3000)
                                if np.ma.count_masked(insar_stack) > 0:
                                    insar_stack = insar_stack.filled()


                                # Get #SRTM layer
                                srtm_scenes, _ = dl.scenes.search(
                                    dltile,
                                    products = 'srtm:GL1003',
                                    limit=None,
                                )

                                print('Download is valid!')

                                srtm_raster = srtm_scenes.mosaic(bands= 'slope', ctx=dltile,
                                                                 scaling="physical")

                                # Get USDA Cropland Layer
                                if (self.area_name == 'ca_central_valley_cropland' or self.area_name == 'california' or
                                    self.area_name == 'iowa'):
                                    lulc_product = 'usda:cdl:v1'
                                    lulc_band_name = 'class'
                                elif self.area_name == 'catalonia':
                                    lulc_product = 'dl-interns:sigpac_2020_crop_classifications'
                                    lulc_band_name = 'crop_class'
                                else:
                                    lulc_product = 'dl-interns:esa_2020_worldcover'
                                    lulc_band_name = 'lulc_class'

                                year = int(self.start_date[0][0:4])
                                start_lulc_dt = f'{year}-01-01T00:00:00'
                                end_lulc_dt = f'{year}-12-31T23:59:59'

                                lulc_scenes, _ = dl.scenes.search(
                                    dltile,
                                    products=lulc_product,
                                    start_datetime=start_lulc_dt,
                                    end_datetime=end_lulc_dt,
                                    limit=None)

                                lulc_raster = lulc_scenes.mosaic(bands=lulc_band_name, ctx=dltile)

                                # Begin populating S1 output array. Organize by timestep
                                s1_output_stack = np.zeros((16, 256, 256))
                                for jx in range(self.num_total_s1_timesteps):

                                    s1_output_stack[jx*4:(jx*4)+2] = np.transpose(s1_stack_rtc[jx], (2,0,1))
                                    s1_output_stack[(jx*4)+2:(jx+1)*4] = np.transpose(insar_stack[jx,...,0:2], (2,0,1))

                                # Add local incidence angle, SRTM slope, and valid radar px band
                                s1_output_stack = np.concatenate((s1_output_stack, rtc_stack[...,1][None,...],
                                                                  srtm_raster, valid_radar_mask_combined),
                                                                axis=0).astype(np.float32)



                                # Define fields for file extension
                                s1_date = self.s1_scenes_list[ix][0][0]
                                s2_date = self.s2_scenes_list[ix][0]
                                s2_date = datetime.datetime.strptime(s2_date, "%Y%m%d").strftime("%Y-%m-%d")
                                s1_props = self.s1_scenes_list[ix][0][-1][0].properties['id'].split(":")[-1]
                                s1_track = s1_props.split('-')[4]
                                look_pass = s1_props.split('-')[-2]

                                lon = str(np.round(dltile.geometry.centroid.xy[0][0], decimals=4))
                                lat = str(np.round(dltile.geometry.centroid.xy[1][0], decimals=4))
                                file_ext = self.file_ext.format(lat, lon, s1_date, s2_date,
                                                                s1_track, look_pass)
                                lulc_file_ext = f'lat_{lat}_lon_{lon}_date_{year}_lulc_label.tif'

                                s1_save_str = f'{self.base_dir}/s1/{file_ext}'
                                s2_save_str = f'{self.base_dir}/s2/{file_ext}'
                                lulc_save_str = f'{self.base_dir}/labels/{lulc_file_ext}'

                                # Download tiles!
                                print(f'Downloading {file_ext}')
                                downloaded_imgs_for_tile += 1

                                if self.download_locally:
                                    print(f'Write tif to local')

                                    self.write_gdal_file(
                                        filename=s1_save_str,
                                        geotransform=dltile.geotrans,
                                        geoproj=dltile.wkt,
                                        data=s1_output_stack,
                                        data_type=gdal.GDT_Float32,
                                        format="GTiff",
                                        nodata=-3000,
                                        options=None,
                                    )

                                    self.write_gdal_file(
                                        filename=s2_save_str,
                                        geotransform=dltile.geotrans,
                                        geoproj=dltile.wkt,
                                        data=s2_output_stack,
                                        data_type=gdal.GDT_Int16,
                                        format="GTiff",
                                        nodata=-3000,
                                        options=None,
                                    )

                                    if not self.tile_download:
                                        self.write_gdal_file(
                                            filename=lulc_save_str,
                                            geotransform=dltile.geotrans,
                                            geoproj=dltile.wkt,
                                            data=lulc_raster,
                                            data_type=gdal.GDT_Int16,
                                            format="GTiff",
                                            nodata=-3000,
                                            options=None,
                                        )

                                if self.upload_to_gcp:
                                    self.upload_to_gcp_func('s1', s1_save_str, s1_output_stack)
                                    self.upload_to_gcp_func('s2', s2_save_str, s2_output_stack)

                                    if not self.tile_download:
                                        self.upload_to_gcp_func('labels', lulc_save_str, lulc_raster)


                                self.tile_download = True
                                invalid_count_per_tile_season = 0

                            else:
                                print('invalid insar')
                        else:
                            print('invalid s1')
                    else:
                        print('invalid s2')
            except Exception as e:
                print(e)
        else:
            print('No available S2/S1 imagery')


if __name__ == '__main__':
    '''
     __main__ module for downloading sets of S1, S2, and LULC imaery in SEN12TS dataset 
    '''

    generator = S1_S2_LULC_Generator()
    valid_tiles = generator.load_dltiles()
    print(f'Number of valid tiles for download: {len(valid_tiles)}')
    
    dltiles = valid_tiles
    seasons = ['spring', 'summer', 'fall', 'winter']

    generator.pbar = tqdm(total=generator.dltiles_to_download)
    generator.pbar.update(generator.saved_dltiles)
    generator.pbar.refresh()

    
    download = True

    if download:
        print('Downloading')

        tile_ix = 0
        while generator.saved_dltiles < generator.dltiles_to_download:
            dltile = dltiles[tile_ix]
            dltile = dl.scenes.DLTile.from_key(dltile)

            dl_imgs_for_tile = 0

            for season in seasons:
                generator.find_imagery(dltile, season)
                generator.download_imagery_for_dltile(dltile, season)
                dl_imgs_for_tile += int(generator.tile_download)

            if dl_imgs_for_tile > 0:
                generator.saved_dltiles += 1
                generator.pbar.update(1)


            tile_ix += 1