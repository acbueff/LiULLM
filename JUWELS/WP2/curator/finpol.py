import pathlib
import sys

import polars as pl

import dask
import dask.distributed as dd
import pyarrow.parquet as pq
from dask.dataframe import read_parquet, from_delayed
from dask.distributed import Client, as_completed

keep = pathlib.Path('/p/data1/trustllmd/WP2/data/dedup')
src_root = pathlib.Path('/p/data1/trustllmd/WP2/data/collected')
dst_root = pathlib.Path('/p/data1/trustllmd/WP2/data/final_new_nld')

CHUNKSIZE = 1024*1024*1024*8

def get_chunks(dir):
    size = 0
    buf = []
    for f in sorted(dir.rglob('*.parquet')):
        size += f.stat().st_size
        buf.append(str(f))
        if size >= CHUNKSIZE:
            yield (size, buf)
            size = 0 
            buf = []
    if buf:
        yield (size, buf)

def run_all(lang, tid, tot):
    keepers = pl.scan_parquet(keep / lang).filter(keep=True).select('id').sort('id').cache()
    all_chunks = list(enumerate(get_chunks(src_root / lang)))
    my_chunks = all_chunks[tid::tot]
    for i, chunk in my_chunks:
        size, files = chunk
        print(f'Running global chunk {i} / {len(all_chunks)}', flush=True)
        print(f'Files: {files}')
        print(f'Original size on disk: {size}', flush=True)
        tmp = dst_root / lang / f'TMP_chunk_{i}.parquet'
        dst = dst_root / lang / f'chunk_{i}.parquet'
        if not dst.exists():
            dst.parent.mkdir(exist_ok=True, parents=True)
            pl.scan_parquet(files).join(keepers, on='id', how='semi').collect().write_parquet(tmp, statistics=False)
            tmp.rename(dst)
            print(f'DONE {dst}')
        else:
            print(f'SKIP {dst}')

if __name__ == '__main__':
    lang = sys.argv[1]
    task_id = int(sys.argv[2])
    task_tot = int(sys.argv[3])
    print(f'Running {lang} task {task_id} / {task_tot}')
    run_all(lang, task_id, task_tot)
