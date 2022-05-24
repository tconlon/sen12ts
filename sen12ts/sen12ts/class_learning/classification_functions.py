import tensorflow as tf
import descarteslabs as dl
import numpy as np
import pandas as pd
import json
import os
import datetime
from glob import glob
from shapely.geometry import shape
from osgeo import gdal, ogr
from typing import Sequence
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit
import random
# from findpeaks import lee_enhanced_filter
from sen12ts.class_learning.datagenerator import parse_example
from sen12ts.class_learning.datagenerator import get_args

def collect_sar_timeseries(folder_name):
    s1_directory = f'{folder_name}/s1'
    s2_directory = f'{folder_name}/s2'

    imgs = sorted(glob(s1_directory + '/*.tif'))

    tiles = [i.split('/')[-1].split('_lat')[0].replace('dltile_', '') for i in imgs]

    unique_tiles = len(np.unique(tiles))
    unique_tiles_dict = {}

    for ix, tile in enumerate(tiles):
        if tile not in unique_tiles_dict.keys():
            unique_tiles_dict[tile] = [imgs[ix]]
        else:
            unique_tiles_dict[tile].append(imgs[ix])

    tile_imgs_counter = []
    full_ts_tiles = []


    for tile_key in unique_tiles_dict.keys():
        num_imgs = len(unique_tiles_dict[tile_key])
        tile_imgs_counter.append(num_imgs)
        if num_imgs == 16:
            full_ts_tiles.append(tile_key)


    # tile_imgs_counter_unique = [(np.count_nonzero(tile_imgs_counter == i), i) for i in np.unique(tile_imgs_counter)]

    all_tile_files_sorted = []

    for ix, tile in enumerate(full_ts_tiles):
        tile_files = np.array([i for i in imgs if tile in i])
        tile_files_s2_ixs = np.argsort([i.split('/')[-1].split('s2date_')[-1][0:10] for i in tile_files]).astype(int)

        tile_files_sorted = tile_files[tile_files_s2_ixs]

        all_tile_files_sorted.append(tile_files_sorted)


    return all_tile_files_sorted

def determine_unique_label_ct(args, ds_path):

    dataset = tf.data.TFRecordDataset(ds_path).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    ct_dict = {}

    for n, (input_batch, target_batch, batch_dates) in dataset.enumerate():

        target_batch = target_batch[...,-1]
        
        if n == 0:
            print(batch_dates)

        unique_lulc, unique_cts = np.unique(target_batch, return_counts=True)

        for ix, lulc_ix in enumerate(unique_lulc):
            if lulc_ix in ct_dict.keys():
                ct_dict[lulc_ix] += unique_cts[ix]
            else:
                ct_dict[lulc_ix] = unique_cts[ix]

    # Sort
    ct_dict = {k: v for k, v in sorted(ct_dict.items(), key=lambda item: item[1], reverse=True)}

    print(f'Stats for {ds_path}')

    df = pd.DataFrame()
    label_list = []
    label_count = []


    for i, (k, v) in enumerate(ct_dict.items()):
        print(f'{i}, {k} label, {v} pixels')

        if i < args.NUM_CROPS:
            label_list.append(k)
            label_count.append(v)

    df['label'] = label_list
    df['label_count'] = label_count

    if args.SAVE_CROP_CT_CSV:
        ntiles = ds_path.split('_')[-1].replace('.tfrecords', '')
        ds_type = ds_path.split('usdaclass_')[-1].split('_')[0]
        out_file = f'{args.MAIN_DIR}/crop_label_counts/labels_from_{ds_type}_ntiles_{ntiles}_ncrops_{args.NUM_CROPS}.csv'

        df.to_csv(out_file)




def test_encoding(args, train_path):


    label_df = f'../../class_models/crop_label_counts/labels_from_train_ntiles_160_ncrops_10.csv'
    df = pd.read_csv(label_df)


    train_dataset = tf.data.TFRecordDataset(train_path).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    train_dataset.batch(1).prefetch(1)

    labels = df['label']
    print(labels)

    pred_tensor = tf.random.uniform(shape=(256*256,11), dtype=tf.float32)

    for n, (input_batch, target_batch, batch_dates) in train_dataset.enumerate():

        target_layer_list = []

        for ix, label in enumerate(labels):
            layer = tf.cast(tf.equal(target_batch, label), tf.int32)
            target_layer_list.append(layer)

        target_layer = tf.concat(target_layer_list, axis=-1)
        other_label_layer = 1 - tf.math.reduce_max(target_layer, axis=-1)

        target_layer = tf.concat((target_layer, tf.expand_dims(other_label_layer, axis=-1)), axis=-1)


        tl_reshaped = tf.argmax(tf.reshape(target_layer, shape=(tf.shape(target_layer)[0] *
                                                      tf.shape(target_layer)[1],
                                                      tf.shape(target_layer)[2])),
                                axis=-1)

        tl_crops_ix = tf.squeeze(tf.where(tl_reshaped < 10), axis=-1)
        print(tl_crops_ix)




        tl_crops = tf.gather(pred_tensor, tl_crops_ix)
        print(tl_crops)


        # print(top_k)








if __name__ == '__main__':
    area_name = 'ca_central_valley_cropland'
    folder_name = f'../../data/tfrecords/{area_name}'

    train_path = f"{folder_name}/usdaclass_train_s1-t_s2-lulc_ntiles_160_split_64px_nimgs_1817.tfrecords"
#     test_path = f"{folder_name}/usdaclass_test_s1-t_s2-lulc_ntiles_41.tfrecords"

    args = get_args()

    determine_unique_label_ct(args, train_path)

