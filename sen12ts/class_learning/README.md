## Cropland classification file descriptions

`band_normalization.py`: Contains functions for calculating and applying normalizations (mean=0, std=0) to each input band.
`classification_functions.py`: Contains utility functions for training cropland classification models.
`convert_to_tfrecord.py`: Contains functions for converting local `.tifs` to `.tfrecords` for model training and testing.
`datagenerator.py`: Contains functions for loading `.tfrecords` into `tf.data.Dataset` objects, and then applying necessary preprocessing steps for model training and testing. 
`loss.py`: Contains the function that defines the classificaiton loss. 
`main.py`: The script that runs all aspects of the cropland classification model functionality. Run this script to train a new model, load an existing model for inference, or calculate performance statistics. 
`metrics.py`: Contains functions that calculate classifier performance. 
`model.py`: Contains the model definition, which is based on a model introduced [here](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf). 
`params.yaml`: Contains all parameters for classifier training or testing.
`plotting.py`: Allows users to plot classifier predictions. 