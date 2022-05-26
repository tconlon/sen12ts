## SEN12TS imagery download file descriptions 

`deploy_download_script.py`: Script that allows the `download_paired_data.py` file to be deployed in parallel on a series of virtual machines via [Descartes Labs Tasks](https://docs.descarteslabs.com/guides/tasks.html) functionality. All files are downloaded to a Google Cloud Platform bucket.
`download_paired_data.py`: Main script that downloads paired Sentinel-1, Sentinel-2, and region-specific land use/land-cover layers. Files are downloaded either locally or to a Google Cloud Platform bucket. 
`utils.py`: Contains utility functions for SEN12TS dataset downloading. 