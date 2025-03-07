import collections
import inspect
from itertools import combinations, product, islice

from functools import wraps, reduce
from typing import Any
from dataclasses import dataclass, field

import polars as pl
import polars.datatypes as pld

def normalize(text, remove_punct=True, lowercase=True, whitespace=True):
    if remove_punct:
        text = text.str.replace_all(r'\p{P}', ' ')
    if lowercase:
        text = text.str.to_lowercase()
    if whitespace:
        text = text.str.replace_all(r'[\s--\n]+', ' ')
        text = text.str.replace_all(r' ?\n ?', '\n')
    return text

def alias(f):
    name = f.__name__
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs).alias(name)
    return inner

def ratio(a, b):
    return (a / b).cast(pld.Float32)

EMPTY_LINES = r'(?m:^\s*$)'
LINES = r'(?m:$)'
ENDING_PUNCTUATION = r'(?m:\p{P}\s*$)'
ENDING_ELLIPSIS = r'(?m:(\u2026|\.\.\.|\. \. \.)\s*$)'
INITIAL_BULLET = r'(?m:^\s*(\u25cf|\u2022|\u002a|\u002d))'
SYMBOLS = r'(?m:#|\.\s?\.\s?\.|\u2026)'
ALPHANUMERIC = r'\w'
WORD = r'\w+'
WORD_WITH_LETTER = r'\w*\p{L}\w*'

def content_length(text):
    return text.str.len_chars()

def num_lines(text):
    return text.str.count_matches(LINES)

def num_ending_punctuation(text):
    return text.str.count_matches(ENDING_PUNCTUATION)

def ratio_ending_punctuation(text):
    return ratio(num_ending_punctuation(text), num_lines(text))

def num_ending_ellipsis(text):
    return text.str.count_matches(ENDING_ELLIPSIS)

def ratio_ending_ellipsis(text):
    return ratio(num_ending_ellipsis(text), num_lines(text))

def num_initial_bullet(text):
    return text.str.count_matches(INITIAL_BULLET)

def ratio_initial_bullet(text):
    return ratio(num_initial_bullet(text), num_lines(text))

def num_symbols(text):
    return text.str.count_matches(SYMBOLS)

def num_words(text):
    return text.str.count_matches(WORD)

def num_words_with_letter(text):
    return text.str.count_matches(WORD_WITH_LETTER)

def ratio_words_with_letter(text):
    return ratio(num_words_with_letter(text), num_words(text))

def ratio_symbols_to_words(text):
    return ratio(num_symbols(text), num_words(text))

def num_short_lines(text):
    return text.str.split('\n').list.eval(pl.element().str.strip_chars().str.len_chars() <= 30).list.sum()

def ratio_short_lines(text):
    normalized = normalize(text)
    return ratio(num_short_lines(normalized), num_lines(normalized))

def num_alphanumeric(text):
    return text.str.count_matches(ALPHANUMERIC)

def ratio_alphanumeric(text):
    return ratio(num_alphanumeric(text), text.str.len_chars())

def average_word_length(text):
    return ratio(num_alphanumeric(text), num_words(text))

def get_words(text):
    return normalize(text).str.extract_all(WORD)

def word_count(text):
    return get_words(text).list.eval(pl.element().value_counts())

def unigram_entropy(text):
    return word_count(text).list.eval(pl.element().struct[1].entropy()).list.first().cast(pld.Float32)

def ratio_repetition(segments, char_ratio=True):
    unique = segments.list.unique()
    if char_ratio:
        return ratio(
                unique.list.eval(pl.element().str.len_chars()).list.sum(),
                segments.list.eval(pl.element().str.len_chars()).list.sum()
            )
    else:
        return ratio(
                unique.list.len(),
                segments.list.len()
            )

def ratio_unique_lines(text):
    return ratio_repetition(normalize(text).str.split('\n'), char_ratio=False)

def ratio_unique_line_chars(text):
    return ratio_repetition(normalize(text).str.split('\n'))

def ratio_unique_paragraph_chars(text):
    return ratio_repetition(normalize(text).str.split('\n\n'))

def ratio_unique_word_chars(text):
    return ratio_repetition(normalize(text).str.replace('\n', ' ').str.split(' '))

def ratio_lines_to_words(text):
    return ratio(num_lines(text), num_words(text))

@dataclass
class Bound:
    low: Any = None
    high: Any = None
    def __call__(self, x):
        match self.low, self.high:
            case (None, None):
                return True
            case (None, high):
                return x <= high
            case (low, None):
                return low <= x
            case (low, high):
                return x.is_between(low, high)

def gopher_signals(text):
    functions = [
            num_words, 
            average_word_length, 
            ratio_symbols_to_words, 
            ratio_initial_bullet, 
            ratio_ending_ellipsis, 
            ratio_words_with_letter,
            ]
    return [alias(f)(text) for f in functions]

def repetition_signals(text):
    functions = [
            ratio_unique_lines, 
            ratio_unique_line_chars,
            ratio_unique_paragraph_chars,
            ]
    return [alias(f)(text) for f in functions]

def fineweb_signals(text):
    functions = [
            ratio_ending_punctuation, 
            ratio_short_lines,
            ratio_lines_to_words,
            ]
    return [alias(f)(text) for f in functions]

def sweb_signals(text):
    functions = [
            ratio_alphanumeric,
            unigram_entropy,
            content_length,
        ]
    return [alias(f)(text) for f in functions]

def all_signals(text):
    return *gopher_signals(text), *repetition_signals(text), *fineweb_signals(text), *sweb_signals(text)

GOPHER_FILTER = dict(
        num_words=Bound(low=50, high=100_000),
        average_word_length=Bound(low=3, high=10),
        ratio_symbols_to_words=Bound(high=.1),
        ratio_initial_bullet=Bound(high=.9),
        ratio_ending_ellipsis=Bound(high=.3),
        ratio_words_with_letter=Bound(low=.8)
        )

REPETITION_FILTER = dict(
        ratio_unique_lines=Bound(low=.7),
        ratio_unique_line_chars=Bound(low=.9),
        ratio_unique_paragraph_chars=Bound(low=.8),
        )

FINEWEB_FILTER = dict(
        ratio_ending_punctuation=Bound(low=.12),
        ratio_short_lines=Bound(high=.67),
        ratio_lines_to_words=Bound(high=.3),
        )

SWEB_FILTER = dict(
        ratio_alphanumeric=Bound(low=.4),
        content_length=Bound(low=100, high=1e10),
        unigram_entropy=Bound(3, 100),
        )

COMBINED_FILTER = {**GOPHER_FILTER, **REPETITION_FILTER, **FINEWEB_FILTER, **SWEB_FILTER}

def mk_filter(**kwargs):
    return reduce(lambda a, b: a & b, [f(pl.col(k)) for k, f in kwargs.items()])

def mk_filter_signals(**kwargs):
    return [f(pl.col(k)) for k, f in kwargs.items()]

def ngrams(text, n):
    segments = get_words(text)
    length = pl.element().list.len()
    ngrams = segments.list.eval(pl.concat_str(*[pl.element().shift(-i) for i in range(n)], separator=' '))
    return ngrams.list.eval(pl.element().value_counts().struct[1]).list.sum()

def get_signals(src):
    return pl.scan_parquet(src, low_memory=True).select(*all_signals(pl.col('text')))

async def runner(queue):
    while True:
        src, dst = await queue.get()
        print(f'{datetime.datetime.now()} STARTING: {src}', flush=True)
        df = await get_signals(src).collect_async(streaming=True)
        tmp = dst.with_name(f'TMP_{dst.name}')
        df.write_parquet(tmp)
        tmp.rename(dst)
        del df
        print(f'{datetime.datetime.now()} FINISHED: {src}', flush=True)
        queue.task_done()

def ordered_scan(root, pattern):
    def inner(curr):
        if curr.is_dir():
            for result in sorted(curr.iterdir()):
                yield from inner(result)
        elif curr.is_file() and curr.match(pattern):
            yield curr
    yield from inner(root)

async def memwatch():
    while True:
        print(f'memory: {psutil.virtual_memory().percent}%')
        m1, m5, m15 = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        print(f'cpu: {m1}% {m5}% {m15}%')
        await asyncio.sleep(30)


def get_files(dir):
    for f in sorted(dir.rglob('*.parquet')):
        yield f

async def run_all(src_root, dst_root, tid, tot, WORKERS=12):
    my_files = islice(ordered_scan(src_root, '*.parquet'), tid, None, tot)
    queue = asyncio.Queue(WORKERS*2)
    tasks = [asyncio.create_task(runner(queue)) for _ in range(WORKERS)]
    watcher = asyncio.create_task(memwatch())
    for src in my_files:
        dst = dst_root / src.relative_to(src_root)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            await queue.put((src, dst))
    
    await queue.join()
    
    watcher.cancel()
    for task in tasks:
        task.cancel()


if __name__ == '__main__':
    import psutil
    import datetime
    import asyncio
    import pathlib
    import sys
    import os
    print(f'POLARS THREAD POOL SIZE: {pl.thread_pool_size()}')
    src_root = pathlib.Path(sys.argv[1])
    dst_root = pathlib.Path(sys.argv[2])
    task_id = int(sys.argv[3])
    task_tot = int(sys.argv[4])
    asyncio.run(run_all(src_root, dst_root, task_id, task_tot))
