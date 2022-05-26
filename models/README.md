## SEN12TS models subdirectory description

These subdirectories (and files contained within) should be downloaded from Zenodo.

### Repository structure
```
├── models/
│   ├── class_models/
│   │   ├── training_checkpoints/: contains Tensorflow training checkpoints.
│   │   ├── logs/: contains training logs.
│   │   ├── figures/: contains saved cropland classification prediction figures.
│   │   ├── confusion_matrices/: contains prediction confusion matrices.
│   │   ├── class_weights/: contains weights coresponding to the top-10 crop types used in classification.
│   │   ├── band_normalization_calculations/: contains bandwise mean and standard deviation values for normalization.
│   ├── sar2vi_models/
│   │   ├── training_checkpoints/: contains Tensorflow training checkpoints.
│   │   ├── logs/: contains training logs.
│   │   ├── figures/: contains figures showing predicted and actual vegetation indices.
│   │   ├── band_normalization_calculations/: contains bandwise mean and standard deviation values for normalization.
```