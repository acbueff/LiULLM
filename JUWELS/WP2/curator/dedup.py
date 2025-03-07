import logging
import pathlib
import sys

import dask
import dask.distributed as dd
import pyarrow.parquet as pq
import numpy
import hashlib
from dask.dataframe import read_parquet, from_delayed
#from dask.distributed import PipInstall
#from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed

src = pathlib.Path('/p/data1/trustllmd/WP2/data/minhash')
dest = pathlib.Path('/p/data1/trustllmd/WP2/data/dedup')

BUCKETS=25
BUCKETSIZE=8*10

def read_group(gid, file):
    f = pq.ParquetFile(file)
    df = f.read_row_group(gid, columns=['id', 'hash']).to_pandas()
    buckets = {}
    for bix in range(BUCKETS):
        # Use two "hashed" hashes (uint64) to reduce risk of collision
        buckets[f'B_{bix}_0'] = df.hash.str.slice(bix*BUCKETSIZE, (bix+1)*BUCKETSIZE).map(lambda x: numpy.uint64(int.from_bytes(hashlib.sha1(x.encode('utf-8')).digest()[:8], 'little')))
        buckets[f'B_{bix}_1'] = df.hash.str.slice(bix*BUCKETSIZE, (bix+1)*BUCKETSIZE).map(lambda x: numpy.uint64(int.from_bytes(hashlib.sha1(x.encode('utf-8')).digest()[8:16], 'little')))
    return df.drop(columns='hash').assign(**buckets)


def get_bucket(files, bix):
    parts = []
    for file in files:
        gids = [gid for gid in range(pq.ParquetFile(file).num_row_groups)]
        parts.extend(client.map(read_bucket, gids, file=file))
    df = from_delayed(parts)

if __name__ == '__main__':
    lang = sys.argv[1]
    scheduler_file = sys.argv[2]

    dask.config.set({"dataframe.shuffle.method": 'p2p'})  
    client = Client(scheduler_file=scheduler_file)
    WORKERS = len(client.scheduler_info()['workers'])
    PARTITIONS = WORKERS*4
    print(lang)
    print(WORKERS)
    print(client.dashboard_link, flush=True)

    files = []
    for l in src.iterdir():
        if l.name == lang:
            for s in l.iterdir():
                files.extend(s.iterdir())

    print(f'reading {len(files)} files', flush=True)
    parts = []
    for file in files:
        gids = [gid for gid in range(pq.ParquetFile(file).num_row_groups)]
        parts.extend(client.map(read_group, gids, batch_size=PARTITIONS, file=file))

    df = from_delayed(parts).repartition(npartitions=PARTITIONS)#.set_index('id', npartitions='auto')

    deduped = df
    for bix in range(BUCKETS):
        bucket = [f'B_{bix}_0', f'B_{bix}_1']
        deduped = deduped.drop_duplicates(subset=bucket, split_out=PARTITIONS).drop(columns=bucket)
    
    keep = df[['id']].merge(deduped, how='left', left_on='id', right_on='id', indicator=True)
    keep = keep.assign(keep=keep._merge=='both').drop(columns='_merge')
    keep.repartition(partition_size='256MiB').to_parquet(dest / lang)
    client.shutdown()
