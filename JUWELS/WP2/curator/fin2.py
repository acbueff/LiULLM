import logging
import pathlib
import sys

import dask
import dask.distributed as dd
import pyarrow.parquet as pq
import numpy
import hashlib
from dask.dataframe import read_parquet, from_delayed
from dask.distributed import Client, as_completed

keep = pathlib.Path('/p/data1/trustllmd/WP2/data/dedup')
src = pathlib.Path('/p/data1/trustllmd/WP2/data/collected')
dest = pathlib.Path('/p/data1/trustllmd/WP2/data/final2')

def files2chunks(files, chunksize=1024*1024*1024*8):
    size = 0 
    chunk = []
    for file in files:
        meta = pq.ParquetFile(file).metadata
        for gid in range(meta.num_row_groups):
            size += meta.row_group(gid).total_byte_size
            chunk.append((file, gid))
            if size >= chunksize:
                yield chunk
                chunk = []
                size = 0
    if chunk:
        yield chunk

def readgroup(file, gid):
    return pq.ParquetFile(file).read_row_group(gid).to_pandas()

#def chunk2ddf(chunk):
#    files, gids = zip(*chunk)
#    client.map(readgroup, files, gids)

if __name__ == '__main__':
    lang = sys.argv[1]
    scheduler_file = sys.argv[2]

    dask.config.set({"dataframe.shuffle.method": 'p2p'})  
    client = Client(scheduler_file=scheduler_file)
    WORKERS = len(client.scheduler_info()['workers'])
    print(lang)
    print(WORKERS, flush=True)

    keepers = read_parquet(str(keep / lang), columns=['id'], filters=[('keep', '==', True)]).set_index('id', npartitions=WORKERS).persist()

    CHUNKSIZE=8*1024*1024*1024 # 8GB
    size = 0
    buf = []
    chunks = []
    for f in sorted((src / lang).rglob('*.parquet')):
        size += f.stat().st_size
        buf.append(str(f))
        if size >= CHUNKSIZE:
            chunks.append(buf)
            size = 0 
            buf = []
    if buf:
        chunks.append(buf)
    
    print(len(keepers), flush=True)
    (dest/ lang).mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        print(f'joining chunk {i} / {len(chunks)}', flush=True)
        print(f'{chunk[0]}-{chunk[-1]}', flush=True)
        read_parquet(chunk, split_row_groups=True). \
                repartition(npartitions=WORKERS*2). \
                join(keepers, on='id', how='inner'). \
                repartition(partition_size='128MB'). \
                to_parquet(str(dest / lang), name_function=lambda j: f'chunk-{i}_part-{j}.parquet')

    client.shutdown()
