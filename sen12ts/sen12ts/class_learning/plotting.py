import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colorbar, colors
from sen12ts.class_learning.metrics import return_crop_names
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import pandas as pd

def plot_class_predictions(args, val_ds, model):
    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue', 'melon', 'peach', 'scarlet',
                   'dark turquoise', 'cobalt', ]
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))
    sns.set_style("whitegrid")

    crops_file = '../../class_models/crop_label_counts/labels_from_train_ntiles_1817_ncrops_10.csv'
    crops = pd.read_csv(crops_file, index_col=0)

    cmap_im_all_preds = colors.ListedColormap(sns.xkcd_palette(colors_xkcd))

    bounds_all_preds = np.arange(start=-0.5, stop=10.5, step=1)
    crop_dict = return_crop_names()

    for n, (s1_list, s2_list, usda_layer, dates) in enumerate(val_ds.take(8)):

        if args.COMBINED_MODEL:
            model_input = [s1_list, s2_list]
        else:
            model_input = s1_list

        pred = model(model_input, training=False)

        pred_argmax = tf.argmax(pred, axis=-1)
        tar_argmax = tf.argmax(usda_layer, axis=-1)

        #         for i in range(11):
        #             print(f'num class {i} in pred: {np.count_nonzero(pred_argmax == i)}')
        #             print(f'num class {i} in target: {np.count_nonzero(tar_argmax == i)}')

        for ix in range(pred_argmax.shape[0]):

            num_ts = 5

            s2_image = s2_list[ix][num_ts]
            rgb_image = np.stack((s2_image[..., 3] * args.S2_BANDS_STD[num_ts * 13 + 3] +
                                  args.S2_BANDS_MEAN[num_ts * 13 + 3],
                                  s2_image[..., 2] * args.S2_BANDS_STD[num_ts * 13 + 2] +
                                  args.S2_BANDS_MEAN[num_ts * 13 + 2],
                                  s2_image[..., 1] * args.S2_BANDS_STD[num_ts * 13 + 1] +
                                  args.S2_BANDS_MEAN[num_ts * 13 + 1]),
                                 axis=0) / 3000

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 6))

            im0 = ax[0].imshow(pred_argmax[ix], interpolation='nearest',
                               origin='upper', cmap=cmap_im_all_preds)
            im1 = ax[1].imshow(tar_argmax[ix], interpolation='nearest',
                               origin='upper', cmap=cmap_im_all_preds)
            im2 = ax[2].imshow(np.transpose(rgb_image, (1, 2, 0)), origin='upper')

            ax[0].set_title('Predicted Crop Classes')
            ax[1].set_title('USDA CDL Crop Classes')
            ax[1].set_title(f'Sentinel 2 RBG Image: {dates[num_ts]}')

            fontprops = fm.FontProperties(size=12)
            bar_width = 10
            scalebar = AnchoredSizeBar(ax[0].transData,
                                       bar_width, '100m', 'lower left',
                                       pad=0.3,
                                       color='Black',
                                       frameon=True,
                                       size_vertical=2,
                                       fontproperties=fontprops)
            ax[0].add_artist(scalebar)

            plt.tight_layout(pad=1.08, h_pad=1.16, w_pad=None, rect=None)

            fig_dir = f'../../figures/{args.CHECKPOINT_FOLDER}/crop_class_comps'
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            plt.savefig(f'{fig_dir}/crop_class_comps_{n * pred_argmax.shape[0] + ix}.png',
                        bbox_inches='tight')
            plt.close()
