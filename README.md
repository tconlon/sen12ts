# sen12ts
SEN12TS Dataset Creation and Applications Repository

This repository contains code related to the SEN12TS dataset, a public resource described here and hosted by [Radiant Earth]( https://mlhub.earth/data/sen12ts).

The SEN12TS dataset contains Sentinel-1, Sentinel-2, and labeled land cover image triplets over six agro-ecologically diverse areas of interest: California, Iowa, Catalonia, Ethiopia, Uganda, and Sumatra. Using the [Descartes Labs](https://descarteslabs.com/) geospatial analytics platform, 246,400 triplets are produced at 10m resolution over 31,398 256-by-256-pixel unique spatial tiles for a total size of 1.69 TB. The image triplets include radiometric terrain corrected synthetic aperture radar (SAR) backscatter measurements; interferometric synthetic aperture radar (InSAR) coherence and phase layers; local incidence angle and ground slope values; multispectral optical imagery; and decameter-resolution land cover data. Moreover, sensed imagery is available in timeseries: Within an image triplet, radar-derived imagery is collected at four timesteps 12 days apart. For the same spatial extent, up to 16 image triplets are available across the calendar year of 2020.

This repository contains code used to 1) create the SEN12TS dataset using the Descartes Labs platform; 2) translate radar imagery into predicted vegetation indices via a model called SAR2VI; and 3) classify croplands in the California Central Valley using combinations of radar and optical imagery. The directory structure is as follows:

## Repository Structure

```
sen12ts
├── environment.yml
├── LICENSE
├── README.md
├── setup.py
├── data/
│   ├── README.md
│   ├── shapefiles/
│   ├── tfrecords/
├── models/
│   ├── README.md
│   ├── class_models/
│   ├── sar2vi_models/
├── sen12ts/
│   ├── __init__.py
│   ├── README.md
│   ├── class_learning/
│   │   ├── __init__.py
│   │   ├── band_normalization.py
│   │   ├── classification_functions.py
│   │   ├── convert_to_tfrecord.py
│   │   ├── datagenerator.py
│   │   ├── loss.py
│   │   ├── main.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── params.yaml
│   │   ├── plotting.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── deploy_download_script.py
│   │   ├── download_paired_data.py
│   │   ├── utils.py
│   ├── sar2vi_learning/
│   │   ├── __init__.py
│   │   ├── band_normalization.py
│   │   ├── convert_to_tfrecord.py
│   │   ├── datagenerator.py
│   │   ├── loss.py
│   │   ├── main.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── params.yaml
│   │   ├── tensorboard_run.py
```

## Root level repository description

The following text describes the files and folders at root directory level. For additional infromation on the files contained within subfolders, please reference the README files within those folders. 

For the two applications of the SEN12TS dataset demonstrated in this repository, data used for training and the trained models should reside within the `data/` and `models/` folders, respectively. However, because these files are too large for GitHub (33 GB), they are hosted on Zenodo where they are available for download. Users should then place the downloaded files in the corresponding directories. 

`environment.yml`: Creates the necessary Python environment for running the SEN12TS code. 

`LICENSE`: Contains the Creative Commons Open License for this repository.

`setup.py`: Allows users to install the repository as a importable module. 

`sen12ts/`: Contains the code used to create the SEN12TS dataset (`data_collection/`), translate radar imagery into predicted vegetation indices (`sar2vi_learning/`), and create a cropland classifier (`class_learning/`). 

