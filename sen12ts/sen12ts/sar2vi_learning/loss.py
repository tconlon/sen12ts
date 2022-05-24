import tensorflow as tf
from sen12ts.sar2vi_learning.metrics import return_cropped_pixels
import numpy as np

def generator_loss(args, disc_generated_output, gen_output, target, gen_loss_object):
    '''
    Generator loss function. 
    The gan loss is: maximize(log(D)).
    Lambda sets the weighting for the L1 loss
    
    Target has dimensions: (None, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS + CDL_IN_OUTPUT)
    
    '''
    
    # Define GAN loss
    gan_loss = gen_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

   
    # Implement the cropland specific L1 loss
    if args.CROPPED_PIXEL_LOSS:
        
        # Find valid pixels and extend the last dimension
        val_pixels = return_cropped_pixels(target)
        val_pix_ix_tf = tf.where(val_pixels)
        
        # Take the VI values of the target + generated image at the pixel locations
        target_cropped     = tf.gather_nd(target[..., 0:args.OUTPUT_CHANNELS], val_pix_ix_tf)
        gen_output_cropped = tf.gather_nd(gen_output, val_pix_ix_tf)
        
        # Calculate the loss
        crop_loss = tf.reduce_sum(tf.abs(target_cropped - gen_output_cropped))
        
    else:
        crop_loss = tf.constant(0, dtype = tf.float32)
    
     # L1 loss represents the mean absolute error between generated and target pixels
    l1_loss = tf.reduce_mean(tf.abs(target[..., 0:args.OUTPUT_CHANNELS] - gen_output))
    
    
    # These values determined heuristically 
    LAMBDA_L1 = 100
    LAMBDA_CROP = 1/5000
    
    # Total generator loss is combined GAN + L1 loss
    total_gen_loss =  gan_loss + (LAMBDA_L1 * l1_loss) + (LAMBDA_CROP * crop_loss)

    return total_gen_loss, gan_loss, l1_loss, crop_loss


def discriminator_loss(disc_real_output, disc_generated_output, disc_loss_object):
    '''
    Discriminator loss function.
    Binary cross entropy loss is applied.
    '''
    real_loss = disc_loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = disc_loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss