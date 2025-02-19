import dask
import logging
import ftfy
from minhash import Combined, Shingler, MinHash
from dask.dataframe import read_parquet
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed

src = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/normalized/*.parquet'
dest = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/minhash'

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

    cluster.scale(jobs=8)
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
    ddf = read_parquet(src, columns=['id', 'text'], split_row_groups=True)
    ddf.assign(hash=ddf.text.map(hasher.hex, meta=('hash', str))).drop(columns='text').to_parquet(dest)
