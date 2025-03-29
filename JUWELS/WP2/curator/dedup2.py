import dask
import logging
import ftfy
from minhash import Combined, Shingler, MinHash
from dask.dataframe import read_parquet
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed

src = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/minhash/*.parquet'
dest = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/minhash_dedup'

def fuzzy_dedup(df, buckets, bucketsize):
    def reduce(ldf):
        return ldf.groupby('bucket').head()

    def step(ddf):
        bdf = ddf.assign(bucket=ddf.hash.str.slice(0, bucketsize), hash=ddf.hash.str.slice(bucketsize))
        return bdf.map_partitions(reduce, meta=bdf).shuffle('bucket').map_partitions(reduce, meta=bdf)

    deduped = df
    for i in range(buckets):
        deduped = step(deduped).persist()

    deduped = deduped.drop(columns=['hash', 'bucket']).set_index('id')

    return (df[['id']].set_index('id').merge(deduped, how='left', indicator='keep') == 'both').persist()


if __name__ == '__main__':
    cluster = SLURMCluster(
            account='trustllm-eu',
            walltime='12:00:00',
            cores=48,
            memory='96GB',
            processes=24,
            interface='ib0',
            job_extra_directives=['--partition=batch', '--exclusive'],
            )

    cluster.scale(jobs=4)
    client = Client(cluster)
    hasher = Combined(
            Shingler(
                shinglog=4
                ),
            MinHash(
                seed=0x5eed,
                perms=256,
                )
            )
    print(client.dashboard_link)
    ddf = read_parquet(src, columns=['id', 'hash'])
    deduped = fuzzy_dedup(ddf, 25, 10*8)
    deduped.repartition(partition_size='100MiB').to_parquet(dest)
