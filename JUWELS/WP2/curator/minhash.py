import sys
import math
import re
import unicodedata
import struct
import hashlib

from dataclasses import dataclass
from functools import partial

from mersenne_field import affine_61, p_61, w_32

import numpy
from numba import vectorize, guvectorize, jit, uint64, uint32

@guvectorize([(uint64, uint64, uint64[:], uint32[:])], '(),(),(h)->()', nopython=True)
def hashmin_gu(a, b, h, out):
    out[0] = w_32
    for hi in h:
        out[0] = min(out[0], affine_61(a, b, hi) & w_32)

def hash7(b):
    return int.from_bytes(hashlib.sha1(b).digest()[:7], 'little')
    
def chunks(xs, chunksize):
    its = iter(xs)
    while (chunk := list(islice(its, chunksize))):
        yield chunk
    
NON_CHAR = re.compile('[^\w ]+')
CODEC = f'utf-32-{"le" if sys.byteorder == "little" else "be"}'
UTYPE = '<U' if sys.byteorder == 'little' else '>U'
    
def string2arr(str, dtype=f'{UTYPE}1'):
    return numpy.frombuffer(bytearray(str, CODEC), dtype=dtype)

def normalize(doc, maxlen=5*1024*1024):
    ret = NON_CHAR.sub('-', doc.lower())[:maxlen]
    return ret

#def shinglify(doc, shinglog=3):
#    tmp = string2arr(((1<<shinglog)-1)*'!' + doc)
#    for i in range(shinglog):
#        d = (1 << i)
#        tmp = numpy.char.add(tmp[:-d], tmp[d:]) 
#    return tmp

@jit(nopython=True)
def slider(vals, levels, fill=0):
    N = vals.shape[0]
    width = (1<<levels)
    out = numpy.full((N, width), fill, dtype=vals.dtype)
    for n in range(N):
        read = min(n+1, width)
        start = n-read+1
        out[n, -read:] = vals[start:start+read]
    return out

@dataclass(frozen=True)
class Shingler:
    shinglog: int = 3

    def __call__(self, doc):
        shingles = numpy.frombuffer(
            slider(
                string2arr(normalize(doc), f'{UTYPE}1'),
                self.shinglog, fill='!'
            ).tobytes(), dtype=f'{UTYPE}{1<<self.shinglog}'
        )
        return numpy.char.encode(shingles, 'utf-8')

@dataclass(frozen=True)
class MinHash:
    seed: int = 0x5eed
    perms: int = 144

    def __getstate__(self):
        return {k: self.__getattribute__(k) for k in ['seed', 'perms']}

    def __setstate__(self, d):
        for k,v in d.items():
            super().__setattr__(k, v)
        self.__post_init__()
        
    def __post_init__(self):
        rng = numpy.random.default_rng(seed=self.seed)
        a = rng.integers(low=1, high=p_61, size=self.perms, dtype=numpy.uint64)
        b = rng.integers(low=0, high=p_61, size=self.perms, dtype=numpy.uint64)
        super().__setattr__('a', a)
        super().__setattr__('b', b)

    def __call__(self, hvs):
        return hashmin_gu(self.a, self.b, hvs)


@dataclass(frozen=True)
class Combined:
    shingle_cfg: Shingler = Shingler()
    minhash_cfg: MinHash = MinHash()

    def __post_init__(self):
        super().__setattr__('_hash', numpy.vectorize(hash7, [numpy.uint64]))
        
    def __getstate__(self):
        return {k: self.__getattribute__(k) for k in ['shingle_cfg', 'minhash_cfg']}

    def __setstate__(self, d):
        for k,v in d.items():
            super().__setattr__(k, v)
        self.__post_init__()

    def shingle(self, doc):
        return self.shingle_cfg(doc)
    
    def hash(self, doc):
        return self._hash(self.shingle(doc))
    
    def minhash(self, doc):
        return self.minhash_cfg(self.hash(doc))

    def hex(self, doc):
        return self.minhash(doc).tobytes().hex()
    
    def __call__(self, doc, stage='hex'):
        match stage:
            case 'minhash':
                return self.minhash(doc)
            case 'hash':
                return self.hash(doc)
            case 'shingle':
                return self.shingle(doc)
            case 'hex':
                return self.hex(doc)
