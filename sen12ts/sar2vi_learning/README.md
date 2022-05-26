## SAR2VI file descriptions

`band_normalization.py`: Contains functions for calculating and applying normalizations (mean=0, std=0) to each input band.
`convert_to_tfrecord.py`: Contains functions for converting local `.tifs` to `.tfrecords` for model training and testing.
`datagenerator.py`: Contains functions for loading `.tfrecords` into `tf.data.Dataset` objects, and then applying necessary preprocessing steps for model training and testing. 
`loss.py`: Contains the function that defines the GAN loss for the SAR2VI model. 
`main.py`: The script that runs all aspects of the SAR2VI model functionality. Run this script to train a new model, load an existing model for inference, or calculate performance statistics. 
`metrics.py`: Contains functions that calculate SAR2VI model performance. 
`model.py`: Contains the SAR2VI model definition, which is based off a Pix2Pix architecture. 
`params.yaml`: Contains all parameters for SAR2VI model training and testing.
`tensorboard_run.py`: Allows users to track model training using tensorboard. 