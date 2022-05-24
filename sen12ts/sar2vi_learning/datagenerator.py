import tensorflow as tf
import glob
import numpy as np
import os

def parse_example(example_proto):
    '''
    Helper function converts .tfrecords into input and output images
    '''

    sar2vi = True

    if sar2vi:
        dates = tf.io.FixedLenFeature([], tf.string)
    else:
        dates = tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)

    features = {
        "image/image_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "target/target_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "target/height": tf.io.FixedLenFeature([], tf.int64),
        "target/width": tf.io.FixedLenFeature([], tf.int64),
        "target/channels": tf.io.FixedLenFeature([], tf.int64),
        "dates": dates,
        
    }

    image_features = tf.io.parse_single_example(example_proto, features)

    img_height = tf.cast(image_features["image/height"], tf.int32)
    img_width = tf.cast(image_features["image/width"], tf.int32)
    img_channels = tf.cast(image_features["image/channels"], tf.int32)

    target_height = tf.cast(image_features["target/height"], tf.int32)
    target_width = tf.cast(image_features["target/width"], tf.int32)
    target_channels = tf.cast(image_features["target/channels"], tf.int32)
    

    image_raw = tf.reshape(
        tf.squeeze(image_features["image/image_data"]),
        tf.stack([img_height, img_width, img_channels]),
    )

    target_raw = tf.reshape(
        tf.squeeze(image_features["target/target_data"]),
        tf.stack([target_height, target_width, target_channels]),
    )

    s2_date = image_features['dates']

    return image_raw, target_raw, s2_date


def resize(s1_image, s2_image, s2_date, height, width):
    '''
    Resizes s1 and s2 images to desired height and width
    '''

    s1_image = tf.image.resize(
        s1_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    s2_image = tf.image.resize(
        s2_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return s1_image, s2_image, s2_date


def random_crop(args, s1_image, s2_image, s2_date):
    '''
    Function stacks images depthwise, then randomly crops. This functino must be mapped onto
    tf.data.Dataset only after the subset of bands are selected.
    '''
    
    total_input_channels  = args.INPUT_CHANNELS * args.TIMESTEPS_PER_SAMPLE + \
                            args.DEM_LAYERS_IN_INPUT
    total_output_channels = args.OUTPUT_CHANNELS + args.CDL_IN_OUTPUT * 1
    
    stacked_image = tf.concat([s1_image, s2_image], axis=-1)

    cropped_image = tf.image.random_crop(
        stacked_image, size=[args.IMG_HEIGHT, args.IMG_WIDTH, 
            total_input_channels + total_output_channels]
    )

    cropped_s1_image = cropped_image[:, :, 0:total_input_channels]
    cropped_s2_image = cropped_image[:, :, total_input_channels : 
                (total_input_channels + total_output_channels)]

    return cropped_s1_image, cropped_s2_image, s2_date



def use_select_s1_s2_bands(args, s1_image, s2_image, s2_date):
    '''
    These bands are determined by the layers saved in the .tfrecord file;
    The .tfrecord layers are determined via:
    `dataprocessing/convert_tif_to_tfrecord.py`
    
    SAR vh and vv data are in log space.
    
    S1 layers:
    0: image t, band vh
    1: image t, band vv
    2: image t, band coherence
    3: image t-1, band vh
    4: image t-1, band vv
    5: image t-1, band coherence
    6: image t-2, band vh
    7: image t-2, band vv
    8: image t-2, band coherence
    9: image t-3, band vh
    10: image t-3,band vv
    11: image t-3, band coherence
    12: srtm slope
    
    S2 layers
    0: image t, ndvi
    1: image t, evi
    2: image t, nir
    3: USDA cropland layer
    '''
    
    
    # Most recent data is in Log space
    s1_image_list = []
    
    '''
    Uncomment + edit the lines below to adjust the input layers
    Currently, the commented lines extract all the SAR layers
    from the input stack to create the new s1_image
    
    '''    
#     for i in range(args.TIMESTEPS_PER_SAMPLE):
#         s1_image_list.extend([s1_image[..., i*3],
#                               s1_image[..., i*3+1]])    
        
#     s1_image = tf.stack(s1_image_list, axis = -1)
        
    # Select ndvi + cdl layers. Change this to predict EVI.

    #  Set predict_band = 0 to predict NDVI
    #  Set predict_band = 1 to predict EVI
    predict_band = 1
    s2_image   = tf.concat([tf.expand_dims(s2_image[..., predict_band], -1),
                           tf.expand_dims(s2_image[..., -1], -1)],
                            axis = -1)
            
    return s1_image, s2_image, s2_date


def type_transform(s1_image, s2_image, s2_date):
    '''
    Ensuring the images are of type float 32
    '''
    
    s1_image, s2_image = tf.cast(s1_image, tf.float32), tf.cast(s2_image, tf.float32)

    return s1_image, s2_image, s2_date
    
    
def random_flip(s1_image, s2_image, s2_date):
    ''' 
    Randomly flip the image 50% of the time
    '''

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        s1_image = tf.image.flip_left_right(s1_image)
        s2_image = tf.image.flip_left_right(s2_image)

    return s1_image, s2_image, s2_date