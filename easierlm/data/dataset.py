from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import os
import numpy as np
import jax.numpy as jnp


class MemMapDatasetJAX:
    """
    A local-file dataset that reads contiguous chunks of token IDs from one or more
    numpy memory-mapped arrays and returns JAX arrays.

    - No torch dependency.
    - No S3/R2/Weka; local files only via np.memmap.
    - __getitem__ returns dict with jnp arrays.
    - If array length is not a multiple of chunk_size, the trailing remainder is ignored.
    """

    def __init__(
        self,
        *paths,
        chunk_size: int = 1024,
        memmap_dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: bool = True,
        generate_attention_mask: bool = False,
        generate_doc_lengths: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        label_mask_paths = None,
        instance_filter_config: Optional[InstanceFilterConfig] = None,
    ):
        if not paths:
            raise ValueError("At least one path is required")

        if generate_attention_mask and pad_token_id is None:
            raise ValueError("'pad_token_id' is required for 'generate_attention_mask'")

        if generate_doc_lengths and eos_token_id is None:
            raise ValueError("'eos_token_id' is required for 'generate_doc_lengths'")

        if label_mask_paths and len(label_mask_paths) != len(paths):
            raise ValueError("There must be the same number of 'label_mask_paths' as there are 'paths'")

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise ValueError("'metadata' should have the same length as the number of file paths")
            _metadata = metadata
        else:
            _metadata = [metadata or {}] * len(paths)

        self._memmap_paths = list(paths)
        self._metadata: List[Dict[str, Any]] = _metadata
        self._label_mask_paths = label_mask_paths
        self._chunk_size = chunk_size
        self.dtype = memmap_dtype

        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self.instance_filter_config = instance_filter_config

        # Open memmaps eagerly so we know sizes and can compute offsets.
        self._memmaps: List[np.memmap] = [
            np.memmap(path, dtype=self.dtype, mode="r") for path in self._memmap_paths
        ]
        self._label_memmaps: Optional[List[np.memmap]] = None
        if self._label_mask_paths is not None:
            self._label_memmaps = [
                np.memmap(mp, dtype=np.bool_, mode="r") for mp in self._label_mask_paths
            ]
            # sanity check equal lengths
            for a, b, p, mp in zip(self._memmaps, self._label_memmaps, self._memmap_paths, self._label_mask_paths):
                if a.size != b.size:
                    raise ValueError(f"mask file '{mp}' must have same number of elements as '{p}'")

        # Compute chunk counts and global offsets.
        self._chunks_per_path: List[int] = [arr.size // self._chunk_size for arr in self._memmaps]
        self._offsets: List[Tuple[int, int]] = []
        start = 0
        for n_chunks in self._chunks_per_path:
            end = start + n_chunks
            self._offsets.append((start, end))
            start = end
        self._num_instances = self._offsets[-1][1] if self._offsets else 0

    # -------------------------
    # Properties
    # -------------------------
    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        return self._chunk_size

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        return self._offsets

    # -------------------------
    # Core
    # -------------------------
    def __len__(self) -> int:
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        if index < 0:
            index = len(self) + index
        if not (0 <= index < len(self)):
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # find which file this index maps into
        memmap_idx = None
        local_idx = None
        for i, (s, e) in enumerate(self._offsets):
            if s <= index < e:
                memmap_idx = i
                local_idx = index - s
                break
        assert memmap_idx is not None and local_idx is not None

        # slice the memmap
        start = local_idx * self._chunk_size
        end = start + self._chunk_size

        tokens_np = self._memmaps[memmap_idx][start:end]
        # Convert to int32 tokens (common choice); change to int64 if needed.
        input_ids = jnp.asarray(tokens_np.astype(np.int32), dtype=jnp.int32)

        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._label_memmaps is not None:
            label_np = self._label_memmaps[memmap_idx][start:end]
            out["label_mask"] = jnp.asarray(label_np, dtype=jnp.bool_)

        return out