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

if __name__ == '__main__':
    lang = sys.argv[1]
    scheduler_file = sys.argv[2]

    (dest / lang).mkdir(parents=True, exist_ok=True)
    dask.config.set({"dataframe.shuffle.method": 'p2p'})  
    client = Client(scheduler_file=scheduler_file)
    WORKERS = len(client.scheduler_info()['workers'])
    PARTITIONS = WORKERS*4
    print(lang)
    print(WORKERS, flush=True)
    keepers = read_parquet(str(keep / lang), columns=['id'], filters=[('keep', '==', True)]).set_index('id').persist()
    read_parquet(str(src/lang)). \
            join(keepers, on='id', how='inner'). \
            repartition(partition_size='128MB'). \
            to_parquet(str(dest/lang))

    client.shutdown()
