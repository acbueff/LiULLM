import dask
import logging
import ftfy
from dask.dataframe import read_parquet
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed, worker_client

src = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/collected/*.parquet'
dest = '/p/scratch/trustllm-eu/cubagyllensten1/prepare/normalized'

schema = pa.schema([
    ('id', pa.string()),
    ('text', pa.string()),
])

def fix_txt(path, group):
    f = pq.ParquetFile(path)
    b, = f.read_row_group(group, columns=['id', 'text']).to_batches()
    ids = b.column('id')
    fixed = [ftfy.fix_text(t) for t in b.column('text').to_pylist()]
    return pa.record_batch([ids, fixed], schema)

def run_file(fun, client, file, dst):
    groups = pq.ParquetFile(file).num_row_groups

    with worker_client() as client, pq.ParquetWriter(dst, schema) as writer:
        parts = client.map(fix_txt, [file]*groups, [group for group in range(groups)])
        for part in as_completed(parts):
            writer.write_batch(part)

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
    print(client.dashboard_link)
    ddf = read_parquet(src, columns=['id', 'text'], split_row_groups=True)
    ddf.assign(text=ddf.text.map(ftfy.fix_text, meta=ddf.text)).to_parquet(dest)
