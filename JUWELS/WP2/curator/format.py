import pathlib
import io
import uuid
import json
from dataclasses import dataclass

import fsspec

import pyarrow as pa
import pyarrow.parquet as pq

import ftfy

import pandas

import jq

import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed, warn

@dataclass(frozen=True)
class arg:
    src: str
    dst: str
    lang: str
    kind: str
    text_getter: str

ftfy_conf = ftfy.TextFixerConfig(
        unescape_html="auto",
        remove_terminal_escapes=True,
        fix_encoding=True,
        restore_byte_a0=True,
        replace_lossy_sequences=True,
        decode_inconsistent_utf8=True,
        fix_c1_controls=True,
        fix_latin_ligatures=False,
        fix_character_width=False,
        uncurl_quotes=False,
        fix_line_breaks=True,
        fix_surrogates=True,
        remove_control_chars=True,
        normalization='NFC',
        explain=False,
        )

def to_parquet(arg, blocksize='128MiB'):
    bsz = dask.utils.parse_bytes(blocksize)
    offset = 0

    file = fsspec.open(arg.src, compression='infer', mode='rb')
    
    _jq = jq.compile(arg.text_getter)

    def get_text(line):
        return ftfy.fix_text(_jq.input_text(line.decode('utf-8')).first(), config=ftfy_conf)

    schema = pa.schema([
        ('id', pa.string()),
        ('text', pa.string()),
        ('offset', pa.uint64()),
        ('origin', pa.string()),
        ('lang', pa.string()),
        ('kind', pa.string()),
    ])

    dst = pathlib.Path(arg.dst)
    tmp = pathlib.Path(f'{arg.dst}_TMP')

    with file as raw_handle, pq.ParquetWriter(tmp, schema) as writer:
        offset = 0
        handle = io.BufferedReader(raw_handle)
        while (lines := handle.readlines(bsz)):
            offsets = []
            texts = []
            idents = []

            N = len(lines)
            for line in lines:
                try:
                    texts.append(get_text(line))
                    idents.append(uuid.uuid4().hex)
                    offsets.append(offset)
                except UnicodeDecodeError as e:
                    warn(f'Unicode error in file {arg.src}, skipping. Error: {e}')
                offset += len(line)
            
            writer.write_batch(
                pa.record_batch(
                    [pa.array(x) for x in [idents, texts, offsets, [arg.src]*N, [arg.lang]*N, [arg.kind]*N]], 
                    schema=schema
                )
            )
    
    tmp.rename(dst)

    return arg.src

destination = '/p/data1/trustllmd/WP2/data'

if __name__ == '__main__':
    
    df = pandas.read_json('sources.json', lines=True)
    args = []
    for (lang, kind), g in df.groupby(['lang', 'source']):
        root = pathlib.Path(destination) / 'collected' / lang / kind
        root.mkdir(parents=True, exist_ok=True)
        for _, r in g.iterrows():
            src = r.path
            fid = uuid.uuid5(uuid.NAMESPACE_URL, src)
            dst = root / f'{fid}.parquet'
            if not dst.exists():
                args.append(arg(r.path, str(dst), lang, kind, r.text_getter))

    cluster = SLURMCluster(
            account='trustllm-eu',
            walltime='24:00:00',
            cores=48,
            memory='96GB',
            processes=24,
            interface='ib0',
            job_extra_directives=['--partition=batch', '--exclusive']
            )

    client = Client(cluster)
    cluster.scale(jobs=1)

    writes = client.map(to_parquet, args, blocksize='128MiB')
    print(client.dashboard_link, flush=True)
    for fut in as_completed(writes):
        print(f'DONE: {fut.result()}', flush=True)

