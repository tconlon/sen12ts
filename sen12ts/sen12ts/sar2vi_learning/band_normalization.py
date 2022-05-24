import tensorflow as tf
import glob
import numpy as np
import os
from sen12ts.sar2vi_learning.datagenerator import parse_example
import pandas as pd

def calculate_normalization(args, train_funcs, norm_dir):
    '''
    Calculate band means and standard deviations for normalization. This only needs 
    to be run once for a given 'num_images_for_norm': The results will save in a 
    .npy file that can be loaded for future normalization. 
    '''
    
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
    for func in train_funcs:
        norm_dataset = norm_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Create numpy arrays to hold images taken for normalization
    input_mean_array  = np.zeros((num_images_for_norm, 
                                  args.INPUT_CHANNELS * args.TIMESTEPS_PER_SAMPLE + \
                                 args.DEM_LAYERS_IN_INPUT))
    input_std_array   = np.zeros((num_images_for_norm, 
                                  args.INPUT_CHANNELS * args.TIMESTEPS_PER_SAMPLE + \
                                 args.DEM_LAYERS_IN_INPUT))
    
    # Load and assign
    norm_dataset = norm_dataset.take(num_images_for_norm)
    for ix, (input_image, output_image, output_date) in enumerate(norm_dataset.as_numpy_iterator()):
        
        input_mean_array[ix]  = np.mean(input_image, axis = (0,1))
        input_std_array[ix]  = np.std(input_image, axis = (0,1))
        
    
    print('Calculating mean')
    # Calculate means + standards deviations
    input_mean = np.expand_dims(np.mean(input_mean_array, axis = 0), 0)
    input_std  = np.expand_dims(np.mean(input_std_array, axis = 0), 0)

    # Save!
    out_file = f'{args.NORMALIZATION_DIR}/model_{norm_dir}_mean_std.csv'
    df = pd.DataFrame(data=np.concatenate((input_mean, input_std), axis = 0))
    df.to_csv(out_file)
            
def load_normalization_arrays(args, norm_dir):
    '''
    This function loads previously saved normalization arrays.
    Normalization arrays are distinguished by args.IMGS_FOR_NORMALIZATION, or the number
    of images used to calculate the terms required for band normalization.
    '''
    
    # Load in mean and standard deviation
    input_norm = pd.read_csv(f'{args.NORMALIZATION_DIR}/model_{norm_dir}_mean_std.csv', index_col=0)
    
    args.INPUT_BANDS_MEAN  = np.array(input_norm.loc[0, :])
    args.INPUT_BANDS_STD   = np.array(input_norm.loc[1, :])
    
    
def apply_band_normalization(args, s1_image, s2_image, s2_date):
    '''
    Apply the band normalization so that mean = 0, std = 1 for all bands in 
    input S1 and output S2 images.
    '''
    
    # Save band normalization mean + std
    input_mean  = tf.constant(args.INPUT_BANDS_MEAN, dtype = tf.float32)
    input_std  = tf.constant(args.INPUT_BANDS_STD, dtype = tf.float32)
    
    # Normalize input band
    s1_image = tf.divide((s1_image - input_mean), input_std)
    
    
    # Stretch VI: -1 to 1 --> 0 to 1 --> -1 to 1
    s2_image_vi = 2 * tf.clip_by_value(s2_image[...,0], 0, 1) - 1
    
    # Restack image
    s2_image = tf.concat([tf.expand_dims(s2_image_vi, -1),
                          tf.expand_dims(s2_image[..., -1], -1)],
                         axis = -1)
    
    return s1_image, s2_image, s2_date