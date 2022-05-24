import tensorflow as tf
import time
from IPython import display

from sen12ts.class_learning.model import convlstm_network, stacked_s1s2_models
from sen12ts.class_learning.loss import crossentropy_loss
from sen12ts.class_learning.datagenerator import *
from sen12ts.class_learning.metrics import *
from sen12ts.class_learning.band_normalization import *
from sen12ts.class_learning.plotting import *


print(tf)
print(tf.__version__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def calculate_dataset_statistics(args, train_ds, val_ds):
    print('Counting samples in the training and validation datasets')
    
    train_count = 0
    val_count = 0
    
    train_count_by_class = np.zeros(args.NUM_CLASSES)
    val_count_by_class = np.zeros(args.NUM_CLASSES)
    
    for n, (s1_list, s2_list, usda_layer, dates) in train_ds.enumerate():
        train_count += usda_layer.shape[0]
        train_tensor = tf.reshape(usda_layer, shape=[-1, args.NUM_CLASSES]).numpy()
        train_tensor_flat = np.sum(train_tensor, axis=0)
        train_count_by_class += train_tensor_flat
        

    for n, (s1_list, s2_list, usda_layer, dates) in val_ds.enumerate():
        val_count += usda_layer.shape[0]
        val_tensor = tf.reshape(usda_layer, shape=[-1, args.NUM_CLASSES]).numpy()
        val_tensor_flat = np.sum(val_tensor, axis=0)
        val_count_by_class += val_tensor_flat

    
    print(f'Total samples in training ds: {train_count}')
    print(f'Total samples in validation ds: {val_count}')

    print(f'Frac samples in each class, training ds {train_count_by_class/np.sum(train_count_by_class)}')
    print(f'Frac samples in each class, validation ds:{val_count_by_class/np.sum(val_count_by_class)}')
    
    training_weights = np.sum(train_count_by_class) / \
          (args.NUM_CLASSES * train_count_by_class)
    print(f'Training weights: {training_weights}')
    
    
    ## Save the weights
    ntiles = args.TRAIN_PATH.split('_')[-1].replace('.tfrecords', '')
    class_weights_file = f'{args.MAIN_DIR}/class_weights/top_{args.NUM_CLASSES-1}_classes_ntiles_{ntiles}.csv'
    
    label_df_fn = f'{args.MAIN_DIR}/crop_label_counts/labels_from_train_ntiles_1817_ncrops_10.csv'
    label_df = pd.read_csv(label_df_fn)
    
    labels = np.array(list(label_df['label']) + [-1])
   
    data_for_export = np.stack((labels, train_count_by_class, 
                                val_count_by_class, training_weights), axis=-1)
    pd.DataFrame(data=data_for_export, columns = ['labels', 'training_count',
                                                  'validation_count', 'training_weight']
                ).to_csv(class_weights_file)

def process_conf_matrix(args, conf_matrix_tensor, top_k_counter, total_crop_counter, epoch):

    conf_matrix_tensor = conf_matrix_tensor.numpy()
    top_k_counter = top_k_counter.numpy()
    total_crop_counter = total_crop_counter.numpy()[0]
        
    print(f'Epoch {epoch}, validation set top 1 accuracy: {top_k_counter[0]/total_crop_counter:.4f}')
    print(f'Epoch {epoch}, validation set top 3 accuracy: {top_k_counter[1]/total_crop_counter:.4f}')
    print(f'Epoch {epoch}, validation set top 5 accuracy: {top_k_counter[2]/total_crop_counter:.4f}')

    if args.LOG:
        with summary_writer.as_default():
            tf.summary.scalar("val_top1_acc", top_k_counter[0]/total_crop_counter, step=epoch)
            tf.summary.scalar("val_top3_acc", top_k_counter[1]/total_crop_counter, step=epoch)
            tf.summary.scalar("val_top5_acc", top_k_counter[2]/total_crop_counter, step=epoch)

        conf_matrix_dir = f'{args.MAIN_DIR}/confusion_matrices/{args.DIR_TIME}'
        if not os.path.exists(conf_matrix_dir):
            os.mkdir(conf_matrix_dir)
        cols =[f'pred_{ix}' for ix in range(args.NUM_CROPS)]
        indices =[f'actual_{ix}' for ix in range(args.NUM_CROPS)]
        cfm_filename = f'{conf_matrix_dir}/conf_matrix_{args.DIR_TIME}_epoch_{epoch}.csv'
        conf_matrix = np.array(conf_matrix_tensor)[0:-1, 0:-1]
        conf_df = pd.DataFrame(data=conf_matrix, columns=cols, index=indices)
        print(conf_df)
        conf_df['total_in_class'] = np.sum(conf_matrix, axis=1)
        conf_df.to_csv(cfm_filename)
            

    calc_f1 = False
    if calc_f1:
        precision = np.zeros(args.NUM_CROPS)
        recall = np.zeros(args.NUM_CROPS)
        f1 = np.zeros(args.NUM_CROPS)

        for ix in range(args.NUM_CROPS):
            TP = conf_matrix_tensor[ix, ix]
            FP = tf.math.reduce_sum(conf_matrix_tensor[:, ix]) - TP
            FN = tf.math.reduce_sum(conf_matrix_tensor[ix]) - TP

            precision[ix] = TP / (TP + FP)
            recall[ix] = TP / (TP + FN)

            if np.sum([precision[ix], recall[ix]]) > 0:
                f1[ix] = 2 * precision[ix] * recall[ix] / (precision[ix] + recall[ix])

    
@tf.function
def predict_over_val_ds(args, model_input, target_batch, model, epoch, conf_matrix_tensor,
                       top_k_tensor, total_crop_counter):
    
        pred = model(model_input, training=False)

        pred_reshape = tf.reshape(pred, shape=(pred.shape[0] * pred.shape[1] * 
                                               pred.shape[2],
                                               pred.shape[3]))

        tar_reshape = tf.reshape(target_batch, shape=(target_batch.shape[0] * 
                                                      target_batch.shape[1] *
                                                      target_batch.shape[2],
                                               target_batch.shape[3]))

        pred_argmax = tf.argmax(pred_reshape, axis=-1)
        tar_argmax = tf.argmax(tar_reshape, axis=-1)
        


        ## Find confidence matrix
        conf_matrix = tf.math.confusion_matrix(tf.reshape(tar_argmax, [-1]),
                                               tf.reshape(pred_argmax, [-1]),
                                               num_classes=args.NUM_CROPS+1)

        conf_matrix_tensor += conf_matrix
        
        tl_crops_ix = tf.squeeze(tf.where(tar_argmax < args.NUM_CROPS), axis=-1)

        pred_crops = tf.gather(pred_reshape, tl_crops_ix)
        tar_crops  = tf.gather(tar_argmax, tl_crops_ix)

        ## Find Top K accuracies
        top1 = tf.math.in_top_k(tar_crops, pred_crops, k=1)
        top3 = tf.math.in_top_k(tar_crops, pred_crops, k=3)
        top5 = tf.math.in_top_k(tar_crops, pred_crops, k=5)

        top_k_new = tf.stack([tf.reduce_sum(tf.cast(top1, tf.int32)), 
                                 tf.reduce_sum(tf.cast(top3, tf.int32)),
                                 tf.reduce_sum(tf.cast(top5, tf.int32))])
        
        top_k_tensor += top_k_new
        total_crop_counter += tf.cast(tf.shape(pred_crops)[0], tf.int32)
    

        return conf_matrix_tensor, top_k_tensor, total_crop_counter
    
    
@tf.function
def train_step(model_input, target_batch, weights, model, epoch):
    '''
    Function that applies a training step for both generator and discriminator
    '''

    with tf.GradientTape() as model_tape:
        # Generate target iamge
        model_output = model(model_input, training=True)

        # Create model loss terms
        model_loss = crossentropy_loss(model_output, target_batch, weights)
#         print(f'loss: {model_loss}')
        
        # Find gradients
        model_gradients = model_tape.gradient(model_loss, model.trainable_variables)
#         print(f'gradients: {model_gradients}')
        
        # Apply gradients
        optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))

        # Write out loss and correlation terms
        if args.LOG:
            with summary_writer.as_default():
                tf.summary.scalar("train_total_loss", model_loss, step=epoch)

        return model_loss, model_output


def fit(args, train_ds, val_ds):
    '''
    Fit function. This function generates a set of images for visualization every epoch,
    and applies the training function

    Calculating the total length of the training set is optional: It takes time to calculate
    once, but after calculating once, I recommend assigning the length to pbar.

    Model checkpointing also occurs once every n epochs.
    '''

    n_train_imgs = int(args.TRAIN_PATH.strip('.tfrecords').split('_')[-1])
    n_test_imgs = int(args.TEST_PATH.strip('.tfrecords').split('_')[-1])

    class_weights_file = f'{args.MAIN_DIR}/class_weights/top_10_classes_fresno_ntiles_1817.csv'
    weights = tf.constant(pd.read_csv(class_weights_file, 
                                      index_col=0)['training_weight'], dtype=tf.float32)
    
    
    total_train_steps = np.ceil(n_train_imgs / args.BATCH_SIZE)
    total_test_steps = np.ceil(n_test_imgs / args.BATCH_SIZE)
    
    top1_max = 0
    

    print('Training')
    for epoch in range(args.EPOCHS):
        start = time.time()
        display.clear_output(wait=True)

        print("Epoch: ", epoch)

        # Train and track progress with pbar
        total_steps = np.ceil(n_train_imgs / args.BATCH_SIZE)
        train_pbar = tqdm(total=total_train_steps, ncols=100)
        test_pbar = tqdm(total=total_test_steps, ncols=100)

        
        for n, (s1_list, s2_list, target_batch, dates) in train_ds.enumerate():
            train_pbar.update(1)

            
            if args.COMBINED_MODEL:
                model_input = [s1_list, s2_list]
            else:
                model_input = s2_list



            # print(model_input[0].shape)
            # print(model_input[1].shape)
            # print(args.S1_BANDS_MEAN.shape)
            # print(args.S1_BANDS_STD.shape)
            # print(args.S2_BANDS_MEAN.shape)
            # print(args.S2_BANDS_STD.shape)


            model_loss, model_output = train_step(model_input, target_batch, weights, model, epoch)
            if model_loss < 1e-3:
                print(model_output)
                print(np.argmax(model_output, axis=-1))
            
    
            train_pbar.set_description(f'Training loss: {model_loss}')
        
        
        print('Calculating validation ds metrics')
        
        conf_matrix_tensor = tf.zeros(shape=[args.NUM_CROPS+1, args.NUM_CROPS+1], dtype=tf.dtypes.int32)
        top_k_counter = tf.constant([0, 0, 0], dtype = tf.int32)
        total_crop_counter = tf.constant([0], dtype = tf.int32)
        
        for n, (s1_list, s2_list, target_batch, dates) in val_ds.enumerate():
            test_pbar.update(1)
            
            if args.COMBINED_MODEL:
                model_input = [s1_list, s2_list]
            else:
                model_input = s2_list
                
            conf_matrix_tensor, top_k_counter, total_crop_counter = \
                predict_over_val_ds(args, model_input, target_batch, model, epoch,
                                    conf_matrix_tensor, top_k_counter, total_crop_counter)             
            
        process_conf_matrix(args, conf_matrix_tensor, top_k_counter, total_crop_counter, epoch)
        top1_score = top_k_counter[0]/total_crop_counter[0]
        
        print(f'Top k counter: {top_k_counter}')
        print(f'Total crop counter: {total_crop_counter}')
          
        if top1_score > top1_max:
            
            print(f'New maximum top-1 accuracy: {top1_score}. Checkpointing model.')
            top1_max = top1_score
            
#             checkpoint_prefix = f'{args.MAIN_DIR}/training_checkpoints/{args.DIR_TIME}'
#             manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, 
#                         max_to_keep=1, 
#                         checkpoint_name = f'epoch_{epoch}_top1_acc_{top1_max:.4f}')

#             manager.save()
            
        else:
            print(f'Maximum top-1 accuracy has not improved from {top1_max}')
        
        
        print("Time taken for epoch {} is {} sec\n".format(epoch,
                                                           time.time() - start))


def prepare_datasets(args):

    label_df_fn = f'{args.MAIN_DIR}/class_weights/top_10_classes_fresno_ntiles_1817.csv'
    label_df = pd.read_csv(label_df_fn)

    train_funcs = [
        type_transform,
        lambda s1, s2, dates: apply_band_normalization(args, s1, s2, dates),
        lambda s1, s2, dates: list_imagery_by_ts (s1, s2, dates),
        lambda s1_list, s2_list, usda_layer, dates: encode_target_to_select_labels(label_df, s1_list, 
                                                                                   s2_list, usda_layer, dates)
    ]

    # Similarly, define the functions that should be mapped onto the validation tf.data.Dataset
    val_funcs = [
        type_transform,
        lambda s1, s2, dates: apply_band_normalization(args, s1, s2, dates),
        lambda s1, s2, dates: list_imagery_by_ts (s1, s2, dates),
        lambda s1_list, s2_list, usda_layer, dates: encode_target_to_select_labels(label_df, s1_list, 
                                                                                   s2_list, usda_layer, dates)
    ]

    ## NEED TO DEFINE NORMALIZATION FUNCTIONS HERE TOO

    # Create tf.data.Dataset for training data -- Not sure about optimal # calls
    # or prefetch parameters
    train_dataset = tf.data.TFRecordDataset(args.TRAIN_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Map functions onto training dataset, then shuffle and batch.
    for func in train_funcs:
        train_dataset = train_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    train_dataset = train_dataset.shuffle(args.BUFFER_SIZE).batch(args.BATCH_SIZE, 
                                                                  drop_remainder=True).prefetch(args.BATCH_SIZE)

    # Create tf.data.Dataset for validation data -- Not sure about optimal # calls
    # or prefetch parameters
    val_dataset = tf.data.TFRecordDataset(args.TEST_PATH).map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Map functions onto testing dataset, then shuffle and batch.
    for func in val_funcs:
        val_dataset = val_dataset.map(
            func, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    val_dataset = val_dataset.batch(args.BATCH_SIZE, drop_remainder=True).prefetch(args.BATCH_SIZE)

    return train_dataset, val_dataset


if __name__ == '__main__':

    '''
    Runs when main.py is called.
    '''

    # Get arguments from params.yaml
    args = get_args()
    args = dotdict(vars(args))

    optimizer = tf.keras.optimizers.Adam(lr=1e-3, clipnorm=3.0, clipvalue=0.5)

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.DIR_TIME = dir_time
    checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{dir_time}/'

    if args.COMBINED_MODEL:
        model = stacked_s1s2_models(args)
    else:
        model = convlstm_network(args, 's2')
        
    print(model.summary(line_length=150))

    
    if not args.LOAD_EXISTING:
        print('Training from scratch')
                    
        # Define checkpoint object + save path
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
            
    else:
        print('Loading from existing')
        # Load existing checkpoint
        prev_checkpoint_prefix = f'{args.CHECKPOINT_DIR}/{args.CHECKPOINT_FOLDER}/'
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
        print(f'Loading existing checkpoint: {tf.train.latest_checkpoint(prev_checkpoint_prefix)}')
        status = checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_prefix))

    # Save new normalization file if necessary
    if args.CALCULATE_NORMALIZATION:
        norm_dir = dir_time
        calculate_normalization(args, dir_time)
    else:
        print('Loading existing normalization')
        norm_dir = 'crop_class_final'

    # Load normalization
    load_normalization_arrays(args, norm_dir)

    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    tf.Graph().finalize()

    # Create a directory to store training results based on new model start time
    if args.LOG:
        summary_writer = tf.summary.create_file_writer(f'{args.LOG_DIR}/{dir_time}')

    # Define the functions that get mapped onto the training tf.data.Dataset
    # These are imported from learning/datagenerator.py

    train_dataset, val_dataset = prepare_datasets(args)
    
    if args.CALC_DS_STATS:
        calculate_dataset_statistics(args, train_dataset, val_dataset)
        
    if args.TRAIN:
        fit(args, train_dataset, val_dataset)
        
    if args.SAVE_MODEL:
        print('Saving model')
        model.save(f'{args.MAIN_DIR}/trained_models/{dir_time}')
        
    if args.PLOT_CROP_PREDS:
        plot_class_predictions(args, val_dataset, model)
