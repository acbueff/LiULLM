import logging
import pathlib

import dask
import dask.distributed as dd
from dask.dataframe import read_parquet
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed

src = '/p/fastdata/trustllmd/WP2/data/minhash'
dest = '/p/fastdata/trustllmd/WP2/data/dedup'
root = pathlib.Path(src)

def fuzzy_dedup(df, buckets, bucketsize):
    def reduce(ldf):
        if len(ldf) == 0:
            dd.print('!!!!!!!!!!!!!!!!!!!!')
        return ldf.groupby('bucket').head()

    def step(ddf):
        bdf = ddf.assign(bucket=ddf.hash.str.slice(0, bucketsize), hash=ddf.hash.str.slice(bucketsize))
        return bdf.map_partitions(reduce, meta=bdf).shuffle('bucket', ignore_index=True).map_partitions(reduce, meta=bdf)

    deduped = df
    for i in range(buckets):
        deduped = step(deduped).repartition(partition_size='256MiB')

    deduped = deduped.drop(columns=['hash', 'bucket']).set_index('id')

    return (df[['id']].set_index('id').merge(deduped, how='left', indicator='keep') == 'both').persist()


if __name__ == '__main__':

    files = []
    for l in root.iterdir():
        if l.name not in ['nor', 'nob', 'nno', 'swe', 'dan', 'isl', 'fao']:
            continue
        futures = []
        for s in l.iterdir():
            files.extend(s.iterdir())
            break
        break

    import pyarrow.parquet as pq

    f = pq.ParquetFile(files[0])
    df = f.read_row_group(0).to_pandas()
    print(df.dtypes)
    import json
    print(json.dumps(dask.config.config))

    #cluster = SLURMCluster(
    #        account='trustllm-eu',
    #        walltime='12:00:00',
    #        local_directory='$SCRATCH_trustllm_eu/cubagyllensten1',
    #        cores=48,
    #        memory='96GB',
    #        processes=8,
    #        interface='ib0',
    #        job_extra_directives=['--partition=batch', '--exclusive'],
    #        )

    #cluster.scale(jobs=4)
    #client = Client(cluster)

    #ddf = read_parquet(files, split_row_roups=True).repartition(partition_size='256MiB')

    #ids = fuzzy_dedup(ddf, 25, 8*10)

    #ids.repartition(partition_size='256MiB').to_parquet(dest)
