import tensorflow as tf
import descarteslabs as dl
import numpy as np
import json
import os
import datetime
import glob
import rasterio
import random
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from tqdm import tqdm

'''
Takes raw images (.tifs) in the s1/ and s2/ folders and converts to
to a tfrecord. User inputs specify the number of previous s1
images to use in the conversion and, whether the images should be
flattened before conversion (in order to remove spatial correlation 
and test the performance of estimators).
'''


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example_date_ts(img_data, target_data, img_shape, target_shape, s1_date_list):
    """ Converts image and target data into TFRecords example.
    Parameters
    ----------
    img_data: ndarray
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)
    dltile: str
        DLTile key
    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    if len(target_shape) == 2:
        target_shape = (*target_shape, 1)

    features = {
        "image/image_data": _float64_feature(img_data),
        "image/height": _int64_feature(img_shape[0]),
        "image/width": _int64_feature(img_shape[1]),
        "image/channels": _int64_feature(img_shape[2]),
        "target/target_data": _float64_feature(target_data),
        "target/height": _int64_feature(target_shape[0]),
        "target/width": _int64_feature(target_shape[1]),
        "target/channels": _int64_feature(target_shape[2]),
        "dates": _int64_feature(s1_date_list),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def convert_to_example_date(img_data, target_data, img_shape, target_shape, s2_date):
    """ Converts image and target data into TFRecords example.

    Parameters
    ----------
    img_data: ndarray
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)
    dltile: str
        DLTile key

    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    if len(target_shape) == 2:
        target_shape = (*target_shape, 1)

    features = {
        "image/image_data": _float64_feature(img_data),
        "image/height": _int64_feature(img_shape[0]),
        "image/width": _int64_feature(img_shape[1]),
        "image/channels": _int64_feature(img_shape[2]),
        "target/target_data": _float64_feature(target_data),
        "target/height": _int64_feature(target_shape[0]),
        "target/width": _int64_feature(target_shape[1]),
        "target/channels": _int64_feature(target_shape[2]),
        "dates": _bytes_feature(tf.compat.as_bytes(s2_date)),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def filter_s1_image(s1_image):
    sigma_bands = [0, 1, 4, 5, 8, 9, 12, 13]

    for band in sigma_bands:
        s1_image[..., band] = lee_filter(s1_image[..., band], size=15)

    return s1_image

def parse_example(example_proto):
    '''
    Helper funciton for parsing the input tfrecords
    '''
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

    s2_date = image_features['s2_date']

    return image_raw, target_raw, s2_date


def convert_s2_image_rgb_included(s2_image):
    # Convert to EVI
    L = 1
    C1 = 6
    C2 = 7.5
    G = 2.5

    NIR = s2_image[..., 7]
    RED = s2_image[..., 3]
    GREEN = s2_image[..., 2]
    BLUE = s2_image[..., 1]

    # Include machine epsilon so that no divide by zero
    evi = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L + np.finfo(np.float32).eps)

    evi = np.clip(evi, a_min=-1, a_max=1)

    # Concatenate USDA CDL
    s2_image = np.stack((evi, RED, GREEN, BLUE, s2_image[..., -1]), axis=-1)

    return s2_image


def make_sar2vi_tfrecord_from_folder(folder_name):
    '''
    Function that converts s1 + s2 images to a tfrecord.
    Spatial information preserved.


    Input images are organized in the following layers:

    (S1 backscatter layers are in log space)

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
    13: image t-3, band vv
    14: image t-3, band coherence
    15: image t-3, band phase
    16: RTC local inclination angle
    17: SRTM slope


    S2 layers
    0: image t, coastal-aerosol
    1: image t, blue
    2: image t, green
    3: image t, red
    4: image t, red-edge
    5: image t, red-edge-2
    6: image t, red-edge-3
    7: image t, NIR
    8: image t, red-edge-4
    9: image t, water-vapor
    10: image t, cirrus
    11: image t, swir1
    12: image t, swir2
    13: image t, USDA cropland layer

    '''

    s1_directory = f'{folder_name}/s1'
    s2_directory = f'{folder_name}/s2'

    filenames = [os.path.basename(x) for x in glob.glob(f'{s1_directory}/*.tif')]

    np.random.seed(7)
    random.shuffle(filenames)

    # Fraction of files to include in the training set
    frac_training = 0.8
    total_frac = 1

    n_files = int(len(filenames) * total_frac)

    n_training_files = int(frac_training * n_files)
    n_testing_files = int(total_frac * n_files) - n_training_files

    training_files = filenames[0:n_training_files]
    testing_files = filenames[n_training_files:(n_training_files + n_testing_files)]

    print(f'Number of files in training: {len(training_files)}')
    print(f'Number of files in testing: {len(testing_files)}')

    ## Saving training tfrecord
    configs = ['test'] #, 'test']
    all_files = [testing_files] #, testing_files]


    for jx, config in enumerate(configs):

        files = all_files[jx]

        out_file = f"{folder_name}/tfrecords/sar2vi_{config}_s1-all-denoised_s2-evi-rgb_nimgs_{len(files)}_s2date.tfrecords"
        writer = tf.io.TFRecordWriter(out_file)
        pbar = tqdm(total=len(files), ncols=90, desc=f'Saving {config} tfrecord')


        for ix, file in enumerate(files[0:16]):

            if ix == 0:
                print(file)

            # Extract s2 prediction date from the file string
            s2_date = file.split("_")[-5]

            s1_image = rasterio.open(os.path.join(s1_directory, file)).read()
            s2_image = rasterio.open(os.path.join(s2_directory, file)).read()

            if ix == 0:
                print(f'Input S1 image shape: {s1_image.shape}')
                print(f'Input S2 image shape: {s2_image.shape}')

            ## Apply Lee filter to S1 image
            # s1_image = filter_s1_image(s1_image)

            ## Extract EVI + RGB layers from S2 image
            s2_image = convert_s2_image_rgb_included(s2_image)

            if ix == 0:
                print(f'Output S1 image shape: {s1_image.shape}')
                print(f'Output S2 image shape: {s2_image.shape}')

            if (np.count_nonzero(np.isnan(s1_image)) + np.ma.count_masked(s1_image)) > 0:
                print(f'ERROR, MASKED/NAN VALUE IN TRAINING S1 IMAGE {file}')

            if (np.count_nonzero(np.isnan(s2_image)) + np.ma.count_masked(s2_image)) > 0:
                print(f'ERROR, MASKED/NAN VALUE IN TESTING S2 IMAGE {file}')

            print(s2_date)

            ## Uncomment these lines to write out to tfrecord
            # example = convert_to_example_date(
            #     s1_image, s2_image, s1_image.shape, s2_image.shape, s2_date,
            # )
            #
            # writer.write(example.SerializeToString())
            # pbar.update(1)



if __name__ == '__main__':
    '''
    __main__ module that defines the features within the TFRecord file, and then converts the imagery saved as TIFs
    into the correct formatting for the TFRecord.
    
    TFRecords allow must faster training + testing in Tensorflow. 
    '''

    ## Define features contained  the TFRecord file
    dates = tf.io.FixedLenFeature([], tf.string)

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
        'dates': dates
    }

    '''
    Convert to tfrecords

    'folder_name' specifies the save location.
    '''

    area_name = 'ca_central_valley_cropland'
    folder_name = f'/Volumes/sel_external/sen12ts/paired_imagery/{area_name}'

    make_sar2vi_tfrecord_from_folder(folder_name)
