import tensorflow as tf
import glob
import numpy as np
import os
from sen12ts.class_learning.datagenerator import parse_example
import pandas as pd

def calculate_normalization(args, norm_dir):
    '''
    Calculate band means and standard deviations for normalization. This only needs 
    to be run once for a given 'num_images_for_norm': The results will save in a 
    .npy file that can be loaded for future normalization. 
    '''

    print('Calculating normalization')

    num_images_for_norm = args.IMGS_FOR_NORMALIZATION
    
    # Load in training dataset. This dataset is separate from the dataset used for training,
    # as I'm not sure I can reset the status of the Dataset after pulling num_images_for_norm
    # in order to calculate the band normalization.
    norm_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    print(f'Training path: {args.TRAIN_PATH}')
  
    # Apply all training funcs in train_funcs. At this point, train_funcs does not include the
    # apply_band_normalization
    # for func in train_funcs:
    #     norm_dataset = norm_dataset.map(
    #         func, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )

    # Create numpy arrays to hold images taken for normalization
    s1_mean_array  = np.zeros((num_images_for_norm, 
                                  args.S1_INPUT_CHANNELS * args.NUM_TIMESTEPS))
    s1_std_array   = np.zeros((num_images_for_norm, 
                                  args.S1_INPUT_CHANNELS * args.NUM_TIMESTEPS))
    
    s2_mean_array  = np.zeros((num_images_for_norm, 
                                  args.S2_INPUT_CHANNELS * args.NUM_TIMESTEPS + 1))
    s2_std_array   = np.zeros((num_images_for_norm, 
                                  args.S2_INPUT_CHANNELS * args.NUM_TIMESTEPS + 1))
    
    
    
    # Load and assign
    norm_dataset = norm_dataset.take(num_images_for_norm)
    for ix, (s1_image, s2_image, output_date) in enumerate(norm_dataset.as_numpy_iterator()):
        
        s1_mean_array[ix]  = np.mean(s1_image, axis = (0,1))
        s1_std_array[ix]  = np.std(s1_image, axis = (0,1))
        
        s2_mean_array[ix]  = np.mean(s2_image, axis = (0,1))
        s2_std_array[ix]  = np.std(s2_image, axis = (0,1))
        
    
    print('Calculating mean')
    # Calculate means + standards deviations
    s1_mean = np.expand_dims(np.mean(s1_mean_array, axis = 0), 0)
    s1_std  = np.expand_dims(np.mean(s1_std_array, axis = 0), 0)
    
    s2_mean = np.expand_dims(np.mean(s2_mean_array, axis = 0), 0)
    s2_std  = np.expand_dims(np.mean(s2_std_array, axis = 0), 0)
    

    # Save!
    s1_out_file = f'{args.NORMALIZATION_DIR}/model_crop_class_s1_layers_mean_std.csv'
    s2_out_file = f'{args.NORMALIZATION_DIR}/model_crop_class_s2_layers_mean_std.csv'

    s1_df = pd.DataFrame(data=np.concatenate((s1_mean, s1_std), axis = 0))
    s1_df.to_csv(s1_out_file)
       
    s2_df = pd.DataFrame(data=np.concatenate((s2_mean, s2_std), axis = 0))
    s2_df.to_csv(s2_out_file)
        
        
        
def load_normalization_arrays(args, norm_dir):
    '''
    This function loads previously saved normalization arrays.
    Normalization arrays are distinguished by args.IMGS_FOR_NORMALIZATION, or the number
    of images used to calculate the terms required for band normalization.
    '''
    
    # Load in mean and standard deviation
    s1_norm = pd.read_csv(f'{args.NORMALIZATION_DIR}/model_{norm_dir}_s1_layers_mean_std.csv', index_col=0)
    
    s2_norm = pd.read_csv(f'{args.NORMALIZATION_DIR}/model_{norm_dir}_s2_layers_mean_std.csv', index_col=0)
    
    args.S1_BANDS_MEAN  = np.array(s1_norm.loc[0, :])
    args.S1_BANDS_STD   = np.array(s1_norm.loc[1, :])
    
    args.S2_BANDS_MEAN  = np.array(s2_norm.loc[0, :])
    args.S2_BANDS_STD   = np.array(s2_norm.loc[1, :])
    
        
    
def apply_band_normalization(args, s1_image, s2_image, s2_dates):
    '''
    Apply the band normalization so that mean = 0, std = 1 for all bands in 
    input S1 and output S2 images.
    '''
    
    # Save band normalization mean + std
    s1_mean  = tf.constant(args.S1_BANDS_MEAN, dtype = tf.float32)
    s1_std  = tf.constant(args.S1_BANDS_STD, dtype = tf.float32)
    
    s2_mean  = tf.constant(args.S2_BANDS_MEAN, dtype = tf.float32)
    s2_std  = tf.constant(args.S2_BANDS_STD, dtype = tf.float32)
    
    
    # Normalize input band
    s1_image = tf.divide((s1_image - s1_mean), s1_std)
    
    s2_image_norm = tf.divide((s2_image[..., 0:-1] - s2_mean[...,0:-1]), s2_std[...,0:-1])
    
    s2_image = tf.concat((s2_image_norm, s2_image[..., -1][...,None]), axis=-1)
    
    
    return s1_image, s2_image, s2_dates