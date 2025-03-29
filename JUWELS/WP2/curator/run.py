import logging
import sys
import pathlib
import uuid

from dataclasses import dataclass

import dask
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from dask.dataframe import read_parquet, from_delayed
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed, worker_client
import dask.distributed

from minhash import Combined, Shingler, MinHash

src = '/p/data1/trustllmd/WP2/data/collected'
dest = '/p/data1/trustllmd/WP2/data/minhash'
root = pathlib.Path(src)

def run_group(gid, src, hasher):
    schema = pa.schema([
        ('id', pa.string()),
        ('hash', pa.string()),
    ])

    f = pq.ParquetFile(src)

    data = f.read_row_group(gid, columns=['id', 'text'])

    texts = data.column('text').to_pylist()

    return pa.record_batch([
        pa.array(data.column('id')),
        pa.array(map(hasher.hex, texts)),
        ], 
        schema=schema)

def run_file(src, dst, hasher):
    schema = pa.schema([
        ('id', pa.string()),
        ('hash', pa.string()),
    ])

    tmp = pathlib.Path(f'{dst}_TMP')

    gids = [gid for gid in range(pq.ParquetFile(src).num_row_groups)]

    with pq.ParquetWriter(tmp, schema) as writer:
        if len(gids) > 1:
            with worker_client() as client:
                #hashf = client.scatter(hasher)
                for part in as_completed(client.map(run_group, gids, src=src, hasher=hasher)):
                    writer.write_batch(part.result())
        elif len(gids) == 1:
            part = run_group(0, src, hasher)
            writer.write_batch(part)
    tmp.rename(dst)
    return str(src)

if __name__ == '__main__':
    cluster = SLURMCluster(
            account='trustllm-eu',
            walltime='24:00:00',
            cores=48,
            memory='96GB',
            processes=24,
            interface='ib0',
            job_extra_directives=['--partition=batch', '--exclusive'],
            )

    cluster.scale(jobs=2)
    client = Client(cluster)
    scheduler_file = sys.argv[1]
    client = Client(scheduler_file=scheduler_file)

    hasher = Combined(
            Shingler(
                shinglog=4
                ),
            MinHash(
                seed=0x5eed,
                perms=256,
                )
            )

    #client.scatter(hasher, broadcast=True)
    #client.scatter(hasher, broadcast=True)

    #def f(t):
    #    return hasher.hex(t)

    #for l in root.iterdir():
    #    futures = []
    #    for s in l.iterdir():
    #        dst = pathlib.Path(dest) / l.name / s.name
    #        if not dst.exists():
    #            print(f'RUNNING: {s}')
    #            dst.parent.mkdir(parents=True, exist_ok=True)
    #            tmp = pathlib.Path(f'{dst}_TMP')
    #            ddf = read_parquet(s, columns=['id', 'text'])
    #            ddf.assign(hash=ddf.text.map(f, meta=('hash', str))).drop(columns='text').to_parquet(tmp)
    #            tmp.rename(dst)
    #        else:
    #            print(f'SKIPPED: {s}')


    #            if not dst.exists():
    #                dst.parent.mkdir(parents=True, exist_ok=True)
    #                futures.append(client.submit(run_file, f, dst, hasher))
    #            else:
    #                print(f'{dst} already exists, skipping')
    #    for f in as_completed(futures):
    #        print(f'DONE: {f.result()}')

    for l in root.iterdir():
        futures = []
        for s in l.iterdir():
            #if s.name not in ['hplt_v.1.2', 'oscar_clean_dedup']:
            #    continue
            for f in s.iterdir():
                dst = pathlib.Path(dest) / l.name / s.name / f.name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    futures.append(client.submit(run_file, f, dst, hasher))
                else:
                    print(f'{dst} already exists, skipping', flush=True)
        for f in as_completed(futures):
            print(f'DONE: {f.result()}', flush=True)

