import dask
import dask.bag as db
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import os
import pathlib
import click
import json
import logging
import hashlib
from functools import partial, update_wrapper
from operator import add

@click.command
@click.option('--id-field', required=True)
@click.option('--input-files', required=True)
@click.option('--output-files', required=True)
@click.option('--keymap', type=(str, str), multiple=True)
@click.option('--id-fmt', default='{}')
def prepare(id_field, input_files, output_files, keymap, id_fmt):
    print('Starting script')
    cluster = SLURMCluster(
            account='trustllm',
            walltime='02:00:00',
            cores=48,
            memory='96GB',
            processes=12,
            interface='ib0',
            job_extra_directives=['--partition=devel', '--exclusive']
            )

    print(cluster.job_script())

    client = Client(cluster)
    cluster.adapt(maximum_jobs=4)

    print(client)

    print('reading df')
    df = dd.read_json(input_files)

    print('computing sizes and offsets')
    sizes = df.map_partitions(lambda x: [len(x)]).compute()

    offsets = [0]
    for size in sizes:
        offsets.append(size + offsets[-1])
    *offsets, total = offsets

    print('running it')

    #def __prepare(ix, line):
    #    km = {src:dst for src, dst in keymap}
    #    doc = {km.get(k, k): v for k, v in json.loads(line).items()}

    #    doc[id_field] = id_fmt.format(ix)

    #    return json.dumps(doc)
    #
    #lines = dd.read_text(input_files)

    #print('Computig sizes and offsets')
    #sizes = lines.map_partitions(lambda x: [sum(1 for _ in x)]).compute()

    #offsets = [0]
    #for size in sizes:
    #    offsets.append(size + offsets[-1])
    #*offsets, total = offsets

    #print('Running it')

    #ix_bag = db.concat([db.range(size, 1).map(partial(add, offset)) if size != 0 else db.from_sequence([]) for offset, size in zip(offsets, sizes)])
    #lines = db.map(__prepare, ix_bag, lines)
    #lines.to_textfiles(output_files)

if __name__ == '__main__':
    prepare()

