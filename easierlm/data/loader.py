# fast_c4_token_subset_loader.py
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
from jax.lax import dynamic_update_slice
import threading, queue, random
from dataclasses import dataclass
from typing import Iterator, List, Optional
from easierlm.config import DataConfig
import os

# ---------------------------
# Prefetch thread
# ---------------------------
class Prefetch:
    def __init__(self, gen, buffer_size=2):
        self.q = queue.Queue(maxsize=buffer_size)
        self.t = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.t.start()
    def _run(self, gen):
        try:
            for x in gen: self.q.put(x)
            self.q.put(None)
        except Exception as e: self.q.put(e)
    def __iter__(self): return self
    def __next__(self):
        x = self.q.get()
        if x is None: raise StopIteration
        if isinstance(x, Exception): raise x
        return x

# ---------------------------
# Token streaming
# ---------------------------
def _iter_tokens_from_npy(files: List[str]) -> Iterator[np.ndarray]:
    for path in files:
        size = os.path.getsize(path)
        arr = np.memmap(path, dtype='uint16', mode='r')
        if arr.dtype == object:
            for doc in arr:
                d = np.asarray(doc, dtype=np.int32).ravel()
                if d.size: yield d
        elif arr.ndim == 1:
            yield arr.astype(np.int32, copy=False)
        elif arr.ndim == 2:
            for row in arr: yield np.asarray(row, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported {arr.shape} in {path}")

# ---------------------------
# Rolling packer
# ---------------------------
class RollingPacker:
    def __init__(self, seq_len: int, eos_id: Optional[int]):
        self.L = int(seq_len); self.eos = eos_id
        self.buf = np.empty(1_000_000, dtype=np.int32)
        self.s = self.e = 0
    def _ensure(self, need: int):
        used = self.e - self.s
        if need <= (self.buf.size - self.e): return
        if used > 0: self.buf[:used] = self.buf[self.s:self.e]
        self.s, self.e = 0, used
        if need > (self.buf.size - self.e):
            nb = np.empty(max(self.buf.size*2, self.e+need), dtype=np.int32)
            nb[:self.e] = self.buf[:self.e]; self.buf = nb
    def push(self, arr: np.ndarray):
        n = int(arr.size)
        if n == 0: return
        self._ensure(n)
        self.buf[self.e:self.e+n] = arr; self.e += n
    def push_doc(self, doc: np.ndarray):
        self.push(doc)
        if self.eos is not None: self.push(np.array([self.eos], np.int32))
    def pop_seq(self):
        if (self.e - self.s) < self.L: return None
        out = self.buf[self.s:self.s+self.L].copy()
        self.s += self.L
        return out

class TokenSubsetLoader:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.ldc = jax.local_device_count()
        self.global_batch = cfg.per_device_batch * self.ldc

        # --- choose subset once ---
        rng = random.Random(cfg.seed)
        files = list(cfg.files); rng.shuffle(files)

        self.subset_files = []
        seen = 0
        for f in files:
            self.subset_files.append(f)
            size = os.path.getsize(f)

            arr = np.memmap(f, dtype='uint16', mode='r')
            est = arr.size if arr.dtype != object else sum(len(np.asarray(d)) for d in arr)
            seen += est
            if seen >= cfg.total_tokens: break
        print(f"[Loader] Using {len(self.subset_files)} shards (~{seen:,} tokens)")

    def epoch(self, epoch_index: int=0):
        if self.cfg.reshuffle_each_epoch:
            rng = random.Random(self.cfg.seed + epoch_index)
            files = self.subset_files[:]; rng.shuffle(files)
        else: files = self.subset_files

        def gen():
            packer = RollingPacker(self.cfg.seq_len, self.cfg.eos_id)
            batch = []
            total = 0
            for chunk in _iter_tokens_from_npy(files):
                if self.cfg.eos_id is not None: packer.push_doc(chunk)
                else: packer.push(chunk)
                seq = packer.pop_seq()
                while seq is not None and total < self.cfg.total_tokens:
                    batch.append(seq); total += len(seq)
                    if len(batch) == self.global_batch:
                        x = np.stack(batch)
                        batch.clear()
                        y = jnp.roll(x, -1)

                        mask = jnp.ones_like(x)
                        mask = dynamic_update_slice(mask, jnp.zeros((mask.shape[0], 1), dtype=jnp.int32), (0, mask.shape[-1] - 1))

                        yield(x, y, mask)
                    seq = packer.pop_seq()
            # drop_last only
        return Prefetch(gen(), buffer_size=self.cfg.prefetch)
