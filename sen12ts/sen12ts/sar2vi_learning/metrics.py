import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
pd.options.mode.chained_assignment = None
from sen12ts.sar2vi_learning.datagenerator import *
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def return_valid_ndvi_range(gen_output, target):
    '''
    Clip NDVI output range to 0 to 1
    '''
    gen_output =  (tf.clip_by_value(gen_output, -1, 1) + 1)/2
    
    target_ndvi = (target[...,0] + 1) / 2
    target =  tf.concat([target_ndvi[...,None], target[..., 1::]],
                        axis = -1)
    
    return gen_output, target
    
    
def calculate_mse(args, gen_output, target):
    '''
    Calculate MSE for all pixels in the image, and also for only cropped pixels
    '''
    
    # Clip + fit to native NDVI range
    gen_output, target = return_valid_ndvi_range(gen_output, target)
    
    
    valid_pixels = return_cropped_pixels(target)
    val_pix_ix_tf = tf.where(valid_pixels)
    
    # Extract valid pixel locations from NDVI layer
    gen_output_flattened = tf.gather_nd(gen_output[..., 0], val_pix_ix_tf)
    target_flattened = tf.gather_nd(target[..., 0], val_pix_ix_tf)
    
    # Mean NDVI for crops
    crop_mean_ndvi = tf.math.reduce_mean(target_flattened)
    all_pixel_mean_ndvi = tf.math.reduce_mean(target[...,0])
    
    # MSE for crops + all pixels
    crop_mse = tf.math.reduce_mean(tf.math.squared_difference(gen_output_flattened,
                                                              target_flattened))
    all_pixel_mse = tf.math.reduce_mean(tf.math.squared_difference(gen_output, target[...,0][...,None]))
    
    # Also return the number of cropped pixels per image
    return all_pixel_mse, crop_mse, tf.size(gen_output_flattened), all_pixel_mean_ndvi, crop_mean_ndvi
    

def calculate_ndvi_correlation(args, gen_output, target):
    ''' 
    Flatten only the first layer of the output/target tensor 
    This layer corresponds to NDVI (Not an issue when predicting only NDVI)
    '''
    
    # Find valid pixels 
    if args.CROPPED_PIXEL_LOSS:
        valid_pixels = return_cropped_pixels(target)
    else:
        valid_pixels = tf.ones_like(target[...,0])
    
    # Valid pixels are for a 2-D tensor         
    val_pix_ix_tf = tf.where(valid_pixels)
        
    # Extract valid pixel locations from NDVI layer
    gen_output_flattened = tf.gather_nd(gen_output[..., 0], val_pix_ix_tf)
    target_flattened = tf.gather_nd(target[..., 0], val_pix_ix_tf)

    # Calculate the correlation
    correlation = tfp.stats.correlation(gen_output_flattened, target_flattened,
                          sample_axis=0, event_axis=None)
        
    # Return the correlation and # of cropped pixels (used for calculation)
    return correlation, tf.size(gen_output_flattened)




def return_cropped_pixels(target):
    '''
    USDA NASS CDL cropland layer is set to the third band of the target data.
    Find all the pixel locations that correspond to cropland (or whatever distinction
    is desired to further tailor training)
    
    See: https://docs.google.com/spreadsheets/d/1U-YntdlXMDp17ZCc6-LS93XnYztVGoqsKNIk7EmsPRs/edit?usp=sharing
    
    Returns a tensor of shape = (target.shape) with valid pixels denoted True
    
    This function is used during training, so all data types must be tensors.
    
    Change the following cropland_values variable (it must remain a list of a list) to apply the L1 crop 
    loss to a select set of cropped land cover classes. Any change to this variable will also return 
    different aggregate crop metrics from calculate_metrics_per_lc_type.  
    
    '''
    # All cropped land cover types included in for crop L1 loss (and metrics) calculations.
    cropland_values = [range(1,7), range(10, 15), range(21, 40),
                       range(41, 59), range(66, 70), range(71, 73),
                       range(74, 78), range(204, 251)]
    
    # A single land cover class specified for crop L1 loss (and metrics) calculations -- here 5 = Soybeans
#     cropland_values = [[5]]
                       
    cropland_values_flat = tf.convert_to_tensor([item for sublist in \
                                                 cropland_values for item in sublist],
                                               dtype = tf.int32)
    
    
    cdl_layer = tf.cast(target[...,-1], dtype = tf.int32)

    ## Flatten array for comparison to CDL crop values
    cdl_layer_flat = tf.reshape(cdl_layer, shape=[-1])


  
    # Equivalent to np.isin(cdl_layer, cropland_values) -- Subtract every value of the CDL layer from 
    # the cropland values, then find the indices of the zeros. 
    find_match_flat = tf.reduce_min(tf.math.abs(cropland_values_flat[..., None] - 
                                cdl_layer_flat[None,...]), 0)
    
    # Reshape the array to match cdl_layer
    find_match = tf.reshape(find_match_flat, shape = cdl_layer.shape)
    
    # Return valid pixel array of shape: (None, IMG_HEIGHT, IMG_WIDTH)
    # Unclear if this will work with batches
    valid_pixels = tf.equal(find_match, tf.zeros_like(find_match))
    
    
    return valid_pixels


def return_crop_names():
    '''
    Create and return a dictionary containing the paired CDL raster values and crop names
    '''

    crop_dict = {1: 'Corn', 2 : 'Cotton',3 : 'Rice', 4 : 'Sorghum', 5 : 'Soybeans', 6 : 'Sunflower',
                   10 : 'Peanuts', 11 : 'Tobacco', 12 : 'Sweet Corn', 13 : 'Pop or Orn Corn', 
                   14 : 'Mint', 21 : 'Barley', 22 :'Durum Wheat', 23 : 'Spring Wheat', 24 : 'Winter Wheat',
                   25 : 'Other Small Grains', 26 : 'Dbl Crop WinWht/Soybeans', 27 : 'Rye', 28 : 'Oats',
                   29 : 'Millet', 30 : 'Speltz', 31 : 'Canola', 32 : 'Flaxseed', 33 : 'Safflower', 
                   34 : 'Rape Seed', 35 : 'Mustard', 36 : 'Alfalfa', 37 : 'Other Hay/Non Alfalfa',
                   38 : 'Camelina', 39 : 'Buckwheat', 41 : 'Sugarbeets', 42 : 'Dry Beans', 43 : 'Potatoes',
                   44 : 'Other Crops', 45 : 'Sugarcane', 46 : 'Sweet Potatoes', 47 : 'Misc Vegs & Fruits', 
                   48 : 'Watermelons', 49 : 'Onions', 50 : 'Cucumbers', 51 : 'Chick Peas', 52 : 'Lentils',
                   53 : 'Peas', 54 : 'Tomatoes', 55 : 'Caneberries', 56 : 'Hops', 57 : 'Herbs',
                   58 : 'Clover/Wildflowers', 59 : 'Sod/Grass Seed', 60 : 'Switchgrass',
                   61 : 'Fallow/Idle Cropland', 62 : 'Pasture/Grass', 63 : 'Forest', 64 : 'Shrubland', 
                   65 : 'Barren', 66 : 'Cherries', 67 : 'Peaches', 68 : 'Apples', 69 : 'Grapes',
                   70 : 'Christmas Trees', 71 : 'Other Tree Crops', 72 : 'Citrus', 74 : 'Pecans', 
                   75 : 'Almonds', 76 : 'Walnuts', 77 : 'Pears', 81 : 'Clouds/No Data', 82 : 'Developed',
                   83 : 'Water', 87 : 'Wetlands', 88 : 'Nonag/Undefined', 92 : 'Aquaculture',
                   111 : 'Open Water', 112	: 'Perennial Ice/Snow', 121	: 'Developed/Open Space',
                   122 : 'Developed/Low Intensity', 123	: 'Developed/Med Intensity', 
                   124 : 'Developed/High Intensity', 131 : 'Barren', 141 : 'Deciduous Forest',
                   142 : 'Evergreen Forest', 143 : 'Mixed Forest', 152 : 'Shrubland', 
                   176 : 'Grassland/Pasture', 190 : 'Woody Wetlands', 195 : 'Herbaceous Wetlands',
                   204 : 'Pistachios', 205 : 'Triticale', 206 : 'Carrots', 207 : 'Asparagus', 208 : 'Garlic',
                   209 : 'Cantaloupes', 210 : 'Prunes', 211 : 'Olives', 212 : 'Oranges', 
                   213 : 'Honeydew Melons', 214 : 'Broccoli', 215 : 'Avocados', 216 : 'Peppers', 
                   217 : 'Pomegranates', 218 : 'Nectarines', 219 : 'Greens', 220 : 'Plums', 
                   221 : 'Strawberries', 222 : 'Squash', 223 : 'Apricots', 224 : 'Vetch', 
                   225 : 'Dbl Crop WinWht/Corn', 226 : 'Dbl Crop Oats/Corn', 227 : 'Lettuce',
                   228 : 'Dbl Crop Triticale/Corn', 229 : 'Pumpkins', 230 : 'Dbl Crop Lettuce/Durum Wht',
                   231 : 'Dbl Crop Lettuce/Cantaloupe', 232 : 'Dbl Crop Lettuce/Cotton',
                   233 : 'Dbl Crop Lettuce/Barley', 234 : 'Dbl Crop Durum Wht/Sorghum', 
                   235 : 'Dbl Crop Barley/Sorghum', 236 : 'Dbl Crop WinWht/Sorghum', 
                   237 : 'Dbl Crop Barley/Corn', 238 : 'Dbl Crop WinWht/Cotton', 
                   239 : 'Dbl Crop Soybeans/Cotton', 240 : 'Dbl Crop Soybeans/Oats',
                   241 : 'Dbl Crop Corn/Soybeans', 242 : 'Blueberries', 243 : 'Cabbage', 244 : 'Cauliflower',
                   245 : 'Celery', 246 : 'Radishes', 247 : 'Turnips', 248 : 'Eggplants', 249 : 'Gourds',
                   250 : 'Cranberries', 254 : 'Dbl Crop Barley/Soybeans'}
        
    return crop_dict


def calculate_metrics_per_lc_type(args, generator, test_ds, checkpoint_folder):
    '''
    Calculate correlation for all land cover type, not just crops
    Store results in a dictionary.
    
    This function will be called after model training to generate information. 
    
    '''
    # Define the values of the CDL raster that correspond to croplands.
    cropland_values = [range(1,7), range(10, 15), range(21, 40),
                       range(41, 59), range(66, 70), range(71, 73),
                       range(74, 78), range(204, 251)]
    cropland_values_flat = [item for sublist in cropland_values for item in sublist]
    crop_dict = return_crop_names()
    
    results_dict = {}
    
    # Set up a progress bar
    total_testing_imgs = int(args.TEST_PATH.strip('.tfrecords').split('_')[-2])
    pbar = tqdm(total=total_testing_imgs, ncols=60) 
    
    all_crop_corr = 0
    all_crop_pixel_count = 0
    
    crops_in_target_data = []
    target_month_list = []
        
    for n, (input_image, target, target_date) in test_ds.enumerate():
        pbar.update(1)
        
        # Find target date
        
        target_month = int(target_date.numpy()[0].decode('utf-8').split('-')[1])
        if target_month not in target_month_list:
            target_month_list.append(target_month)
    
    
        # Generate target image
        gen_output = generator(input_image, training=False)
            
        
        
         # Clip + fit to NDVI range 
        target = target.numpy()    
        target[..., 0] = np.clip((target[..., 0] + 1) / 2, 0, 1)
        gen_output     = np.clip((gen_output.numpy() + 1)/ 2, 0, 1)
        
        
        
        # Count unique crop types + counts in the target image
        crop_types, crop_counts = np.unique(target[..., -1], return_counts = True)
        
        
        # Assess the correlation for all crop types in the image
        for i, crop_type in enumerate(crop_types):
            # Only assess correlation if more than 10 pixels of a certain type are in an image
            if crop_counts[i] > 10:
                
                # If the crop type hasn't yet been added to list that holds crops seen before
                if int(crop_type) not in crops_in_target_data:
                    crops_in_target_data.append(int(crop_type))
                
                # Find indices of target image that correspond to the crop in question
                crop_indices = np.where(target[...,-1] == crop_types[i])
                
                # Find ndvi for these pixels in both target + generated image
                target_pixels_ndvi = target[...,0][crop_indices]
                gen_output_pixels_ndvi = gen_output[..., 0][crop_indices]
                
                # Calculate MSE, MAE
                mae = mean_absolute_error(target_pixels_ndvi, gen_output_pixels_ndvi)                    
                mse = mean_squared_error(target_pixels_ndvi, gen_output_pixels_ndvi)
                
                # Calculate mean NDVI: 0 to 1 range
                mean_ndvi = np.mean(target_pixels_ndvi)
                
                # Store correlation and pixel count in the results dictionary
                dict_string = f'crop_{int(crop_types[i])}_month_{target_month}'

                # Add corr, mae, and crop counts to the dictionary key
                if dict_string not in results_dict.keys():
                    results_dict[dict_string] = [(mse, mae, crop_counts[i], mean_ndvi)]

                else:
                    results_dict[dict_string].append((mse, mae, crop_counts[i], mean_ndvi))
          
                    
    # Begin setting up a results_df dataframe to organize + hold results                
    results_df_columns = [[f'mse_month{i}' for i in range (1,13)],
                          [f'mae_month{i}' for i in range (1,13)],
                          [f'meanndvi_month{i}' for i in range (1,13)],
                          [f'pixelct_month{i}' for i in range(1,13)],
                          ['mean_mse', 'mean_mae', 'mean_ndvi', 'total_pixels']]
    # Flatten
    results_df_columns = [item for sublist in results_df_columns for item in sublist]
    
    # Create rows for aggregate statistics
    addl_rows = ['Total Crops', 'Total Non-Crops']
    
    results_df_indices = [crop_dict[i] for i in crops_in_target_data]    
    results_df_indices.extend(addl_rows)
    
    # Create dataframe
    results_df = pd.DataFrame(0, index = results_df_indices, columns = results_df_columns)
                
    weighted_correlation = 0
    total_num_pixels = 0
        
    # Calculate MAE and pixel count per month per crop
    for crop_index in crops_in_target_data:
        for month_ix in range(1,13):
            # Retrieve results for the crop + month in question
            results_dict_key = f'crop_{crop_index}_month_{month_ix}'
            if results_dict_key in results_dict.keys():
                total_pixels = np.sum([i[2] for i in results_dict[results_dict_key]])
                mse = np.sum([i[0] * i[2] for i in results_dict[results_dict_key]]) / \
                    (total_pixels)
                mae = np.sum([i[1] * i[2] for i in results_dict[results_dict_key]]) / \
                    (total_pixels)
                ndvi = np.sum([i[2] * i[3] for i in results_dict[results_dict_key]]) / \
                    (total_pixels)
    
                # Assign MAE + total pixels to the correct column + row
                results_df[f'mse_month{month_ix}'].loc[crop_dict[crop_index]] = mse
                results_df[f'mae_month{month_ix}'].loc[crop_dict[crop_index]] = mae
                results_df[f'meanndvi_month{month_ix}'].loc[crop_dict[crop_index]] = ndvi
                results_df[f'pixelct_month{month_ix}'].loc[crop_dict[crop_index]] = total_pixels
                
            
    # Populate total + average rows and columns
    for crop_index in crops_in_target_data:
        for month_ix in range(1,13):
            
            if crop_index in cropland_values_flat:
                index = 'Total Crops'
            else:
                index = 'Total Non-Crops'
            
            # Find total # of cropped pixels
            total_ct_per_crop_per_mo = \
            results_df[f'pixelct_month{month_ix}'].loc[crop_dict[crop_index]]
            
            # Proceed with averages calc if pixels exist for this Crop-Month 
            if total_ct_per_crop_per_mo > 0:
                # Find weighted MSE 
                weighted_mse_per_crop_per_mo =  \
                (results_df[f'mse_month{month_ix}'].loc[crop_dict[crop_index]] * 
                results_df[f'pixelct_month{month_ix}'].loc[crop_dict[crop_index]])
                
                # Find weighted MAE 
                weighted_mae_per_crop_per_mo =  \
                (results_df[f'mae_month{month_ix}'].loc[crop_dict[crop_index]] * 
                results_df[f'pixelct_month{month_ix}'].loc[crop_dict[crop_index]])
                
                # Find weighted NDVI
                weighted_ndvi_per_crop_per_mo = \
                (results_df[f'meanndvi_month{month_ix}'].loc[crop_dict[crop_index]] * 
                results_df[f'pixelct_month{month_ix}'].loc[crop_dict[crop_index]])

                # Calculate average monthly MSE, MAE + NDVI for crops/non crops
                results_df[f'mse_month{month_ix}'].loc[index] += weighted_mse_per_crop_per_mo
                results_df[f'mae_month{month_ix}'].loc[index] += weighted_mae_per_crop_per_mo
                results_df[f'meanndvi_month{month_ix}'].loc[index] += weighted_ndvi_per_crop_per_mo
                
                # Calculate total pixel count for crops/non crops
                results_df[f'pixelct_month{month_ix}'].loc[index] += total_ct_per_crop_per_mo
                
                # Calculate average MSE, MAE, NDVI per crop
                results_df['mean_mse'].loc[crop_dict[crop_index]] += weighted_mse_per_crop_per_mo
                results_df['mean_mae'].loc[crop_dict[crop_index]] += weighted_mae_per_crop_per_mo
                results_df['mean_ndvi'].loc[crop_dict[crop_index]] += weighted_ndvi_per_crop_per_mo
                
                # Calculate total pixel count per crop
                results_df['total_pixels'].loc[crop_dict[crop_index]] += total_ct_per_crop_per_mo
                results_df['total_pixels'].loc[index] += total_ct_per_crop_per_mo
                
    
    print('Averaging')
    # Calculate monthly MSEs, MAEs + NDVIs
    for month_ix in range(1,13):
        for index in addl_rows:
            results_df[f'mse_month{month_ix}'].loc[index] /= \
            results_df[f'pixelct_month{month_ix}'].loc[index]
            
            results_df[f'mae_month{month_ix}'].loc[index] /= \
            results_df[f'pixelct_month{month_ix}'].loc[index]

            results_df[f'meanndvi_month{month_ix}'].loc[index] /= \
            results_df[f'pixelct_month{month_ix}'].loc[index]

            
    # Calculate crop total MSEs, MAEs + NDVIs
    for crop_index in crops_in_target_data:
        results_df['mean_mse'].loc[crop_dict[crop_index]] /= \
        results_df['total_pixels'].loc[crop_dict[crop_index]]
        
        results_df['mean_mae'].loc[crop_dict[crop_index]] /= \
        results_df['total_pixels'].loc[crop_dict[crop_index]]
        
        results_df['mean_ndvi'].loc[crop_dict[crop_index]] /= \
        results_df['total_pixels'].loc[crop_dict[crop_index]]
        
    # Calculate average Crop/Non-Crop MAE    
    for crop_index in crops_in_target_data:
        if crop_index in cropland_values_flat:
            index = 'Total Crops'
        else:
            index = 'Total Non-Crops'
            
        results_df['mean_mse'].loc[index] += \
        (results_df['mean_mse'].loc[crop_dict[crop_index]] * 
         results_df['total_pixels'].loc[crop_dict[crop_index]])
        
        results_df['mean_mae'].loc[index] += \
        (results_df['mean_mae'].loc[crop_dict[crop_index]] * 
         results_df['total_pixels'].loc[crop_dict[crop_index]])
        
        results_df['mean_ndvi'].loc[index] += \
        (results_df['mean_ndvi'].loc[crop_dict[crop_index]] * 
         results_df['total_pixels'].loc[crop_dict[crop_index]])
        
    
    # Calculate overall averages, cropwise and monthly    
    for row in addl_rows:
        results_df['mean_mse'].loc[row] /= results_df['total_pixels'].loc[row]
        results_df['mean_mae'].loc[row] /= results_df['total_pixels'].loc[row]
        results_df['mean_ndvi'].loc[row] /= results_df['total_pixels'].loc[row]
        
        
    
    results_df.to_csv(f'../../reports/crop_csv_results/{checkpoint_folder}.csv')
    

def timeseries_analysis(dltile, predicted_ndvi_array, actual_ndvi_array, s1_date_list,
                       checkpoint_folder):
    
    crop_means_dict = {}
    
    # Get crop keys from above function
    crop_dict = return_crop_names()
    
    # Readjust to EVI physcal range ([0,1])
    actual_ndvi_array[..., 0:-1] = (actual_ndvi_array[..., 0:-1] + 1) / 2
    predicted_ndvi_array = np.clip((predicted_ndvi_array + 1)/ 2, 0, 1)
        
    # Find number of timesteps
    num_ts = len(s1_date_list)
    
    # Find unique crop types
    unique_locs_dict = {}
    unique_crop_types = np.unique(actual_ndvi_array[..., -1]).tolist()
    
    # Limit crop types to only those in the dictionary from return_crop_names()
    unique_crop_types = np.array([i for i in unique_crop_types if i in crop_dict.keys()])
        
    # Populate crop type dictionary with crop indices + locations of the crops
    for crop_type in unique_crop_types:
        locs = (actual_ndvi_array[..., -1] == crop_type) 
        pixel_ct = np.count_nonzero(locs)
        
        # Only add to list if more than 50 pixels of that crop type exist in the tile
        if pixel_ct > 50:
            unique_locs_dict[crop_type] = (locs, pixel_ct)
        
    # For all the time periods present, add the ndvi + pixel ct.
    for ts in range(num_ts):
        for crop_type in unique_locs_dict.keys():
            (crop_locs, pixel_ct) = unique_locs_dict[crop_type]
            
            # Calcualte NDVI
            mean_predicted_ndvi = np.mean(predicted_ndvi_array[crop_locs, ts])
            mean_actual_ndvi    = np.mean(actual_ndvi_array[crop_locs, ts])
            
            # If crop type is not in the dictionary, add it
            if crop_type not in crop_means_dict.keys():
                crop_means_dict[crop_type] = [(s1_date_list[ts], pixel_ct,
                                              mean_predicted_ndvi, mean_actual_ndvi)]
            else:
                crop_means_dict[crop_type].append((s1_date_list[ts], pixel_ct,
                                              mean_predicted_ndvi, mean_actual_ndvi))
    
    
    return crop_means_dict
    
    
def process_crop_means_dict(crop_means_dict):
    '''
    Find average statistics for all the cropped pixels present in the
    set of tiles we want to evaluate (currently 1 tile at a time)
    
    '''
    
    # Get crop indices
    crop_keys = crop_means_dict.keys()
   
    # Create a dictionary for the metrics 
    metrics_dict = {}
    
    # Loop through all the crops present the crop_mean_dict
    for ck in crop_keys:
        num_images = len(crop_means_dict[ck])
        
        unique_dates = np.unique([crop_means_dict[ck][i][0] for i in range(num_images)])
        
        total_crop_pixels = 0
        
        # Loop through all the dates present in the dict
        for date in unique_dates:
            date_dt = datetime.datetime.strptime(date, '%Y-%m-%d')
            
            all_crop_tuples_on_date = [crop_means_dict[ck][i] for i in range(num_images) \
                                       if crop_means_dict[ck][i][0] == date]
            n_tuples = len(all_crop_tuples_on_date)
        
        
            total_pred_pixels_per_crop = 0
            total_actual_pixels_per_crop = 0
            weighted_pred_mean = 0
            weighted_actual_mean = 0
            
            # Next portion is for if we're taking the mean across multiple DLTiles
            # Only runs through list index of 1 for now since we're treating each 
            # DLTile independently
            for i in range(n_tuples):
                
                total_pred_pixels_per_crop += all_crop_tuples_on_date[i][1] 
                weighted_pred_mean += all_crop_tuples_on_date[i][1] * \
                all_crop_tuples_on_date[i][2]
                
                if not np.isnan(all_crop_tuples_on_date[i][3]):
                    total_actual_pixels_per_crop += all_crop_tuples_on_date[i][1]
                    weighted_actual_mean += all_crop_tuples_on_date[i][1] * \
                    all_crop_tuples_on_date[i][3]
            
                    
            
            if total_actual_pixels_per_crop == 0:
                weighted_actual_mean = np.nan
            else:
                weighted_actual_mean /= total_actual_pixels_per_crop    

            weighted_pred_mean /= total_pred_pixels_per_crop    
            
            if ck not in metrics_dict.keys():
                # Create dictionary reading containing total num pixels per crop + mean evis
                metrics_dict[ck] = [total_pred_pixels_per_crop, 
                                    [(date_dt, weighted_pred_mean, weighted_actual_mean)]]
            else:
                # Populate dictionary
                metrics_dict[ck][1].append((date_dt, weighted_pred_mean, weighted_actual_mean))

        
    return metrics_dict

    
def plotting_timeseries_results(metrics_dict, dltile, checkpoint_folder, ts_save_folder):
    
    # Sort the dictionary by number of crop pixels
    metrics_dict = {k: v for k, v in sorted(metrics_dict.items(), 
                                            key=lambda item: item[1][0],
                                            reverse=True)}
    
    # Plot a 9-by-9 matrix of predicted/actual timeseries
    nplots_1dim = 3
    
    # Get the crop names 
    crop_names_dict = return_crop_names()
    fig, ax = plt.subplots(nplots_1dim,nplots_1dim, 
                           figsize = (18,12))
    
    min_ax_bottom = 1
    max_ax_top = 0
    
    # Take the crop classes with the 9 highest number of pixels
    min_num_plots = np.min((len(metrics_dict.keys()), nplots_1dim**2))
        
    for ix, key in enumerate(list(metrics_dict.keys())[0:min_num_plots]):
        
        row = ix // nplots_1dim
        col = ix % nplots_1dim
        
        num_dates = len(metrics_dict[key][1])

        # Extract the mean ndvi prediction and target, and plot       
        dates  = [metrics_dict[key][1][i][0] for i in range(num_dates)]
        preds  = [metrics_dict[key][1][i][1] for i in range(num_dates)]
        actual = [metrics_dict[key][1][i][2] for i in range(num_dates)]
        
        num_pred_pixels = metrics_dict[key][0]
                           
        if num_pred_pixels > 100:                   
        
            ax[row][col].plot_date(dates, preds, fmt = 'k-', marker = 'o', ms = 4, 
                                   label = 'Mean Predictions')
            ax[row][col].plot_date(dates, actual, fmt = 'bo', label = 'Mean Target Values')

            ax[row][col].set_title(f'{crop_names_dict[key]} ({num_pred_pixels:.1e} pred. px.)')
            ax[row][col].set_ylabel(f'Mean VI')

            # Find minimum and max subplot range for subplot normalization
            bottom, top = ax[row][col].get_ylim()

            min_ax_bottom = np.min((bottom, min_ax_bottom))
            max_ax_top    = np.max((top, max_ax_top))

            if row == 0 and col == 0:
                ax[row][col].legend(loc = 'upper right')
        
    for ix, key in enumerate(list(metrics_dict.keys())[0:min_num_plots]):
        
        row = ix// nplots_1dim
        col = ix % nplots_1dim
        
        # Normalize all the subplots to the same y-axis limits
        ax[row][col].set_ylim(min_ax_bottom, max_ax_top)
        ax[row][col].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax[row][col].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax[row][col].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        
        
    # Title and save
    fig.suptitle(f'Mean EVI predictions and target values for DLTile {dltile}\n'\
                 f'model {checkpoint_folder}')    
        
    plt.savefig(f'{ts_save_folder}/dltile_{dltile}_model_{checkpoint_folder}_timeseries_preds.png')
    plt.close()
    
