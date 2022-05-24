import tensorflow as tf
import descarteslabs as dl
import numpy as np
import datetime
from tqdm import tqdm
import random
from sen12ts.class_learning.classification_functions import collect_sar_timeseries
import rasterio
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


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


def filter_s1_image_usda_class(s1_image):
    sigma_bands = [0, 1]
    all_sar_bands = [0, 1, 2, 3]

    s1_image_out = np.zeros((256, 256, len(all_sar_bands)))

    for ix, band in enumerate(all_sar_bands):
        if band in sigma_bands:
            s1_image_out[..., ix] = lee_filter(s1_image[..., band], size=15)
        else:
            s1_image_out[..., ix] = s1_image[..., band]

    return s1_image_out


def parse_example(example_proto):
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


def convert_s2_image(s2_image):
    # Convert to EVI
    L = 1
    C1 = 6
    C2 = 7.5
    G = 2.5

    NIR = s2_image[..., 7]
    RED = s2_image[..., 3]
    BLUE = s2_image[..., 1]

    # Include machine epsilon so that no divide by zero
    evi = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L + np.finfo(np.float32).eps)

    evi = np.clip(evi, a_min=-1, a_max=1)

    # Concatenate USDA CDL
    s2_image = np.stack((evi, s2_image[..., -1]), axis=-1)

    return s2_image


def make_usda_classifier_tfrecord_from_folder(folder_name):
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

    multiple_timesteps = False
    s1_directory = f'{folder_name}/s1'
    s2_directory = f'{folder_name}/s2'

    all_tile_files_sorted = collect_sar_timeseries(folder_name)
    random.shuffle(all_tile_files_sorted)

    # Fraction of files to include in the training set
    frac_training = 0.8
    total_frac = 1

    n_files = int(len(all_tile_files_sorted) * total_frac)

    n_training_tiles = int(frac_training * n_files)
    n_testing_tiles = int(total_frac * n_files) - n_training_tiles

    training_tiles = all_tile_files_sorted[0:n_training_tiles]
    testing_tiles = all_tile_files_sorted[n_training_tiles:(n_training_tiles + n_testing_tiles)]

    print(len(training_tiles))
    print(len(testing_tiles))

    ## Saving training tfrecord

    train_out_file = f"{folder_name}/usdaclass_train_s1-t_s2-lulc_ntiles_{n_training_tiles}.tfrecords"
    train_writer = tf.io.TFRecordWriter(train_out_file)
    train_pbar = tqdm(total=len(training_tiles), ncols=90, desc='Saving training tfrecord', position=0)


    for ix, tile in enumerate(training_tiles):

        tile_ts_img = np.full((256, 256, 64), np.nan)
        doy_list = []

        # Extract s2 prediction date from the file string
        for jx, file in enumerate(tile):
            s1_date = file.split('s1date_')[-1][0:10]
            day = datetime.datetime.strptime(s1_date, '%Y-%m-%d')

            doy = day.timetuple().tm_yday
            # doy_list.extend([doy - kx*12 for kx in range(4)])
            doy_list.extend([doy])


            s1_image = rasterio.open(file).read()
            s1_image = filter_s1_image_usda_class(s1_image)
            num_bands = s1_image.shape[-1]
            tile_ts_img[..., jx * num_bands:(jx + 1) * num_bands] = s1_image

            if jx == 0:
                s2_image = rasterio.open(file.replace('/s1/', '/s2/')).read()
                s2_image = s2_image[..., -1]

        if multiple_timesteps:
            doy_list_sorted = np.argsort(doy_list)
            doy_list_sorted_2layers = []
            for jx in range(len(doy_list_sorted)):
                doy_list_sorted_2layers.extend([ 2 *doy_list_sorted[jx], 2* doy_list_sorted[jx] + 1])

            s1_image_sorted = tile_ts_img[np.array(doy_list_sorted_2layers)]
            s1_date_list = sorted(doy_list)


        else:
            s1_date_list = doy_list
            s1_image_sorted = tile_ts_img

        if ix == 0:
            print(f'Tile: {tile[0]}')
            print(f'S1 image shape: {s1_image_sorted.shape}')
            print(f'S2 image shape: {s2_image.shape}')

        if (np.count_nonzero(np.isnan(s1_image_sorted)) + np.ma.count_masked(s1_image_sorted)) > 0:
            print(f'ERROR, MASKED/NAN VALUE IN TRAINING S1 IMAGE {file}')

        if (np.count_nonzero(np.isnan(s2_image)) + np.ma.count_masked(s2_image)) > 0:
            print(f'ERROR, MASKED/NAN VALUE IN TESTING S2 IMAGE {file}')

        example = convert_to_example_date_ts(
            s1_image_sorted, s2_image, s1_image_sorted.shape, s2_image.shape, s1_date_list,
        )

        train_writer.write(example.SerializeToString())
        train_pbar.update(1)

    # ### Saving testing tfrecord
    #
    test_out_file = f"{folder_name}/usdaclass_test_s1-t_s2-lulc_ntiles_{n_testing_tiles}.tfrecords"
    test_writer = tf.io.TFRecordWriter(test_out_file)
    test_pbar = tqdm(total=len(testing_tiles), ncols=90, desc='Saving testing tfrecord', position=1)

    for ix, tile in enumerate(testing_tiles):

        tile_ts_img = np.full((256, 256, 64), np.nan)
        doy_list = []

        # Extract s2 prediction date from the file string
        for jx, file in enumerate(tile):
            s1_date = file.split('s1date_')[-1][0:10]
            day = datetime.datetime.strptime(s1_date, '%Y-%m-%d')

            doy = day.timetuple().tm_yday
            # doy_list.extend([doy - kx*12 for kx in range(4)])
            doy_list.extend([doy])

            s1_image = rasterio.open(file).read()
            s1_image = filter_s1_image_usda_class(s1_image)
            num_bands = s1_image.shape[-1]
            tile_ts_img[..., jx * num_bands:(jx + 1) * num_bands] = s1_image

            if jx == 0:
                s2_image = rasterio.open(file.replace('/s1/', '/s2/')).read()
                s2_image = s2_image[..., -1]

        if multiple_timesteps:
            doy_list_sorted = np.argsort(doy_list)
            doy_list_sorted_2layers = []
            for jx in range(len(doy_list_sorted)):
                doy_list_sorted_2layers.extend([2 * doy_list_sorted[jx], 2 * doy_list_sorted[jx] + 1])

            s1_image_sorted = tile_ts_img[np.array(doy_list_sorted_2layers)]
            s1_date_list = np.array(sorted(doy_list)).astype(np.int64)


        else:
            s1_date_list = doy_list
            s1_image_sorted = np.array(tile_ts_img).astype(np.int64)

        if ix == 0:
            print(f'Tile: {tile[0]}')
            print(f'S1 image shape: {s1_image_sorted.shape}')
            print(f'S2 image shape: {s2_image.shape}')

        if (np.count_nonzero(np.isnan(s1_image_sorted)) + np.ma.count_masked(s1_image_sorted)) > 0:
            print(f'ERROR, MASKED/NAN VALUE IN TRAINING S1 IMAGE {file}')

        if (np.count_nonzero(np.isnan(s2_image)) + np.ma.count_masked(s2_image)) > 0:
            print(f'ERROR, MASKED/NAN VALUE IN TESTING S2 IMAGE {file}')

        example = convert_to_example_date_ts(
            s1_image_sorted, s2_image, s1_image_sorted.shape, s2_image.shape, s1_date_list,
        )

        test_writer.write(example.SerializeToString())
        test_pbar.update(1)


if __name__ == '__main__':

    # Define the features depending on whether it's sar2vi or usda_class


    dates = tf.io.FixedLenSequenceFeature(
        [], dtype=tf.int64, allow_missing=True
    )

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
    folder_name = f'../../data/tfrecords/ca_central_valley_cropland/{area_name}'

    make_usda_classifier_tfrecord_from_folder(folder_name)