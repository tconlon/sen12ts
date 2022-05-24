import descarteslabs as dl
from sen12ts.data_collection.download_paired_data import PairedInputOutputGenerator
from descarteslabs import Auth
import geopandas as gpd

auth = Auth()
print(f'Signed into DL account: {auth.payload["email"]}')


def deploy_downloader(dltile_key):
    from download_paired_data import PairedInputOutputGenerator
    import descarteslabs as dl

    dltile = dl.scenes.DLTile.from_key(dltile_key)
    seasons = ['spring', 'summer', 'fall', 'winter']

    generator = PairedInputOutputGenerator(deploy_virtually=True)


    for season in seasons:
        generator.find_imagery(dltile, season)
        generator.download_imagery_for_dltile(dltile, season)



def deploy_on_tasks():
    docker_image = 'us.gcr.io/dl-ci-cd/images/tasks/public/py3.8:v2021.11.08-4-g4f1620ce'

    tasks = dl.Tasks()
    async_predict = tasks.create_function(
        f=deploy_downloader,
        name='s1s2_download_deploy',
        image=docker_image,
        maximum_concurrency=50,
        memory="3Gi",
        include_modules=['download_paired_data'],
        requirements=['google-cloud-storage'],
        retry_count=0,
        task_timeout=2700, # 45 min

    )

    # async_predict = tasks.get_function_by_id('67de032a')
    print(async_predict)

    return async_predict


if __name__ == '__main__':
    generator = PairedInputOutputGenerator()

    valid_tiles = generator.load_dltiles()
    print(f'Number of valid tiles for download: {len(valid_tiles)}')

    async_predict = deploy_on_tasks()
    region = 'california'


    for dltile_key in valid_tiles[0:3]:
        ## Check to see if centroid of tile in in shapefile
        shp_fn = f'../../data/shapefiles/final_shapefiles_renamed/{region}.geojson'
        region_polygon = gpd.read_file(shp_fn)['geometry'].iloc[0]
        dltile_centroid = dl.scenes.DLTile.from_key(dltile_key).geometry.centroid

        if region_polygon.contains(dltile_centroid):
            async_predict(dltile_key)



    # tasks = dl.Tasks()
    # tasks.rerun_failed_tasks(group_id='9c231721')

