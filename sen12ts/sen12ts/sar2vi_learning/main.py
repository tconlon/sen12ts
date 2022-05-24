import tensorflow as tf
import os
import time
import glob
import numpy as np
import argparse, yaml
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from IPython import display

from sen12ts.sar2vi_learning.model import Generator, Discriminator
from sen12ts.sar2vi_learning.loss import generator_loss, discriminator_loss
from sen12ts.sar2vi_learning.datagenerator import *
from sen12ts.sar2vi_learning.metrics import *
from sen12ts.sar2vi_learning.band_normalization import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Image to image translation with Pix2Pix"
    )

    parser.add_argument(
        "--training_params_filename",
        type=str,
        default="params.yaml",
        help="Filename defining model configuration",
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args


def generate_images(args, model, test_input, tar, epoch, ix):
    '''
    Generate predicted image for an input stack. Create plots for comparing input stack,
    ground truth EVI layer, and predicted EVI layer. Print a scale bar on the ground
    truth image
    '''
    prediction  = model(test_input, training=False)
    crop_pixels = tf.cast(return_cropped_pixels(tar), tf.int8)
    
    
    img = tf.cast(crop_pixels[0], tf.int8)    
    
    fig, ax = plt.subplots(2, 2, figsize = (12,8))
    
    if tf.rank(test_input) == 4:
        test_input_for_plotting = test_input[0, ..., 0]
    elif tf.rank(test_input) == 5:
        test_input_for_plotting = test_input[0, 0, ..., 0]
    
    display_list = [test_input_for_plotting, crop_pixels[0],
                    tar[0][..., 0], prediction[0][..., 0]]
    
    title = ["Input Layer 0 ($vh_{t}$)", "Cropped Pixels (CDL)",  
             "Ground Truth ($EVI_t$)", "Predicted Image ($EVI_t$)"]

    
    # Plot 3 images on a single matplotlib figure
    for i in range(2):
        for j in range(2):
            ax[i][j].set_title(title[i * 2 + j])
            ax[i][j].imshow(display_list[i * 2 + j])
            ax[i][j].axis("off")

            # Add scale bar
            if i == 1 and j == 1:
                fontprops = fm.FontProperties(size=12)
                bar_width = 50
                scalebar = AnchoredSizeBar(ax[i][j].transData,
                                           bar_width, '1 km', 'lower right',
                                           pad=0.3,
                                           color='Black',
                                           frameon=True,
                                           size_vertical=2,
                                           fontproperties=fontprops)

                ax[i][j].add_artist(scalebar)

    # Save figure for visual inspection
    image_dir = f'{args.IMAGE_DIR}/{dir_time}/epoch_{epoch}'
    os.makedirs(image_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'{image_dir}/epoch_{epoch}_img_{ix}_input_groundtruth_prediction.png', bbox_inches = 'tight')
    plt.close()


@tf.function
def train_step(input_image, target, epoch, gen_loss_object, disc_loss_object):
    '''
    Function that applies a training step for both generator and discriminator 
    '''
       
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate target iamge
        gen_output = generator(input_image, training=True)

        # Generate discriminator outputs for real and generated data
        disc_real_output = discriminator([input_image, target[..., 0:args.OUTPUT_CHANNELS]], 
                                         training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
            
        # Create generator and discriminator loss terms
        gen_total_loss, gen_gan_loss, gen_l1_loss, gen_crop_loss = generator_loss(
            args, disc_generated_output, gen_output, target, gen_loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, disc_loss_object)
        
        # Calculate MSE
        all_pixel_mse, crop_mse, num_cropped_pixels, all_pixel_mean_ndvi, crop_mean_ndvi = \
        calculate_mse(args, gen_output, target)
        
        # Create ndvi correlation metric term
        ndvi_correlation, _ = calculate_ndvi_correlation(args, gen_output, target)
    
        # Calculate generator and discriminator gradients
        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        
        # Apply generator and discriminator gradients
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        # Write out loss and correlation terms
        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=epoch)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=epoch)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=epoch)
            tf.summary.scalar("gen_crop_loss", gen_crop_loss, step=epoch)
            tf.summary.scalar("disc_loss", disc_loss, step=epoch)
            tf.summary.scalar("mean_ndvi_corr_train_ds_per_image", ndvi_correlation, step=epoch)
            tf.summary.scalar("mean_ndvi_mse_train_ds_per_cropped_pixel", crop_mse, step=epoch)
            tf.summary.scalar("mean_ndvi_mse_train_ds_per_pixel", all_pixel_mse, step=epoch)
            
            
def calculate_testing_performance_metrics(args, test_ds, epoch):
    # Create list to hold NDVI correlations
    ndvi_correlation_list = []
    ndvi_mse_list_cropped = []
    ndvi_mse_list_allpix = []
    ndvi_mean_list_cropped = []
    ndvi_mean_list_allpix = []
    
    
    total_cropped_pixels = 0
    
    for n, (input_image, target, target_date) in test_ds.enumerate():
        # Generate target image
        gen_output = generator(input_image, training=False)
    
        # Create ndvi correlation metric term
        ndvi_correlation, num_cropped_pixels = calculate_ndvi_correlation(args, gen_output, target)
        
        # Calculate MSE
        all_pixel_mse, crop_mse, num_cropped_pixels, all_pixel_mean_ndvi, crop_mean_ndvi = \
        calculate_mse(args, gen_output, target)
        
        
        # Append to list
        if not tf.math.is_nan(ndvi_correlation) and num_cropped_pixels > 10:
                        
            ndvi_correlation_list.append(ndvi_correlation.numpy() * num_cropped_pixels.numpy())            
            ndvi_mse_list_cropped.append(crop_mse.numpy() * num_cropped_pixels.numpy())
            ndvi_mean_list_cropped.append(crop_mean_ndvi.numpy() * num_cropped_pixels.numpy())
            
            total_cropped_pixels += num_cropped_pixels.numpy()
            
        ndvi_mse_list_allpix.append(all_pixel_mse.numpy())
        ndvi_mean_list_allpix.append(all_pixel_mean_ndvi.numpy())       

            

        
    # Find average and write out
    average_ndvi_correlation  = np.nansum(ndvi_correlation_list) / total_cropped_pixels
    average_ndvi_mse_cropped  = np.nansum(ndvi_mse_list_cropped) / total_cropped_pixels
    average_ndvi_cropped      = np.nansum(ndvi_mean_list_cropped) / total_cropped_pixels
    
    average_ndvi_mse_allpix   = np.nanmean(ndvi_mse_list_allpix)
    average_ndvi_allpix       = np.nanmean(ndvi_mean_list_allpix)
    

    print(f'Average NDVI correlation, cropland: {average_ndvi_correlation}')
    print(f'Average NDVI MSE, cropland: {average_ndvi_mse_cropped}')
    print(f'Average NDVI MSE, all pixels: {average_ndvi_mse_allpix}')
    print(f'Average NDVI, cropland: {average_ndvi_cropped}')
    print(f'Average NDVI, all pixels: {average_ndvi_allpix}')
    
    
    with summary_writer.as_default():
        tf.summary.scalar("ndvi_correlation_test_ds_per_cropped_pixel", 
                          average_ndvi_correlation, step=epoch)
        tf.summary.scalar("ndvi_mse_test_ds_per_cropped_pixel", average_ndvi_mse_cropped, step=epoch)
        tf.summary.scalar("ndvi_mse_test_ds_per_pixel", average_ndvi_mse_allpix, step=epoch)
        
    return average_ndvi_mse_cropped, average_ndvi_mse_allpix

def fit(args, train_ds, test_ds, gen_loss_object, disc_loss_object):
    '''
    Fit function. This function generates a set of images for visualization every epoch,
    and applies the training function. 
    
    Calculating the total length of the training set is optional: It takes time to calculate 
    once, but after calculating once, I recommend assigning the length to pbar.
    
    Model checkpointing also occurs once every n epochs.
    '''
    
    min_test_ndvi_mse = np.inf
    total_training_imgs = int(args.TRAIN_PATH.strip('.tfrecords').split('_')[-2])
    

    print('Training')
    for epoch in range(args.EPOCHS):
        start = time.time()
        display.clear_output(wait=True)

        
        print("Epoch: ", epoch)

        # Train and track progress with pbar
        pbar = tqdm(total=np.ceil(total_training_imgs/args.BATCH_SIZE), ncols=60) 
        
    
        for n, (input_batch, target_batch, batch_s2_dates) in train_ds.enumerate():                        
            pbar.update(1)
                        
            train_step(input_batch, target_batch, epoch, gen_loss_object, 
                       disc_loss_object)
        
        # Generate images for visualization
        for ix, (example_input, example_target, example_s2_date) in enumerate(test_ds.take(16)):
                generate_images(args, generator, example_input, example_target, epoch, ix)
            
        print('Calculating testing ds metrics')
        average_ndvi_mse_cropped, average_ndvi_mse_allpix = \
        calculate_testing_performance_metrics(args, test_ds, epoch)

        # saving (checkpoint) the model every n epochs
        if average_ndvi_mse_cropped < min_test_ndvi_mse:
            
            # Limit to one saved checkpoints + set new file as save path
            manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, 
                        max_to_keep=1, 
                        checkpoint_name = f'epoch_{epoch}_cMSE_{average_ndvi_mse_cropped:.4f}')

            manager.save()
            
            # Update max correlation
            min_test_ndvi_mse = average_ndvi_mse_cropped
            print('New minimum test dataset MSE')
        
        
        print("Time taken for epoch {} is {} sec\n".format(epoch, 
                                                               time.time() - start))

        

if __name__ == "__main__":
    '''
    Runs when main.py is called.
    '''
    
    # Get arguments from params.yaml
    args = get_args()
    
    ## Create the generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)

    ## Create a binary cross entropy loss object to be implemented for both G + D
    gen_loss_object  = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                         label_smoothing = 0.2)

    # Create optimizers for G + D 
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{dir_time}/'
    
    if not args.LOAD_EXISTING:
        # Define checkpoint object + save path
        checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator=generator,
                discriminator=discriminator,
            )
        print('Training from scratch')
    else:
        # Load existing checkpoint

        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.CHECKPOINT_FOLDER}/'
        checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator=generator,
                discriminator=discriminator,
            )
        
        print('Loading existing checkpoint:')
        print(tf.train.latest_checkpoint(prev_checkpoint_prefix))
        
        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix))
    
    
    # Create a directory to store training results based on model start time
    summary_writer = tf.summary.create_file_writer(f'{args.LOG_DIR}/fit/{dir_time}')
    
    
    # Define the functions that should be mapped onto the training tf.data.Dataset
    train_funcs = [
        type_transform,
        lambda s1, s2, s2_date: use_select_s1_s2_bands(args, s1, s2, s2_date),
        lambda s1, s2, s2_date: resize(s1, s2, s2_date, 286, 286),
        lambda s1, s2, s2_date: random_crop(args, s1, s2, s2_date),
        
    ]
    
    # Define the functions that should be mapped onto the testing tf.data.Dataset
    test_funcs = [
        type_transform,
        lambda s1, s2, s2_date: use_select_s1_s2_bands(args, s1, s2, s2_date),
    ]
    
    
    # Calculate the dataset normalization to normalize each input band to mean = 0, 
    # standard deviation = 1.
    if args.CALCULATE_NORMALIZATION:
        print('Calculating normalization statistics')
        norm_dir = dir_time
        calculate_normalization(args, train_funcs, norm_dir)
        load_normalization_arrays(args, norm_dir)
    else:
        print('Loading existing normalization')
        # Load existing normalization stats:
        norm_dir = args.CHECKPOINT_FOLDER
        load_normalization_arrays(args, norm_dir)
        
    # Apply normalization
    print('Normalization statistics for input bands only')
    print(f'Input bands, mean + std: {args.INPUT_BANDS_MEAN}, \n {args.INPUT_BANDS_STD}')
    train_funcs.append(lambda s1, s2, s2_date: apply_band_normalization(args, s1, s2, s2_date))
    test_funcs.append(lambda s1, s2, s2_date: apply_band_normalization(args, s1, s2, s2_date))
        
    
    
    # Create tf.data.Dataset for training data -- Not sure about optimal || calls 
    # or prefetch parameters
    train_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # Map functions onto training dataset, then shuffle and batch. 
    for func in train_funcs:
        train_dataset = train_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    train_dataset = train_dataset.shuffle(args.BUFFER_SIZE).batch(args.BATCH_SIZE).prefetch(1)

    
    # Create tf.data.Dataset for testing data -- Not sure about optimal || calls 
    # or prefetch parameters
    test_dataset = tf.data.TFRecordDataset(args.TEST_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Map functions onto testing dataset, then shuffle and batch.
    for func in test_funcs:
        test_dataset = test_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    test_dataset = test_dataset.batch(1).prefetch(1)

    
    tf.Graph().finalize()

    '''
    The following four function calls correspond to the four primary functionalities 
    contained within this script.
    '''
    
    # Train!
    if args.TRAIN:
        fit(args, train_dataset, test_dataset, gen_loss_object, disc_loss_object)
    
    # Calculate land cover-specific statistics 
    if args.CALCULATE_LC_STATS:
        calculate_metrics_per_lc_type(args, generator, test_dataset, args.CHECKPOINT_FOLDER)
