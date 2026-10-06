"""Read the handful of GGUF header fields needed to size a GPU offload.

To split a model between GPU and CPU sensibly you need its layer count before
loading it. Guessing wastes a whole load attempt - for a 9 GB model that means
reading 9 GB off disk only to fail - and hardcoding per-model counts goes stale
and does not work for a user-supplied .gguf at all.

Only the header is read: a few hundred KB at most, regardless of model size.

GGUF layout (v2/v3):
    magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
    then kv_count entries of:  key (u64 len + bytes) | type u32 | value
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# GGUF metadata value types.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED_SIZES = {
    _UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2, _UINT32: 4, _INT32: 4,
    _FLOAT32: 4, _BOOL: 1, _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}
_FIXED_FORMATS = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I",
    _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?", _UINT64: "<Q", _INT64: "<q",
    _FLOAT64: "<d",
}


@dataclass
class GgufInfo:
    architecture: str = ""
    layers: int = 0           # <arch>.block_count
    context_length: int = 0   # <arch>.context_length


class _Reader:
    def __init__(self, handle: BinaryIO) -> None:
        self._h = handle

    def raw(self, count: int) -> bytes:
        data = self._h.read(count)
        if len(data) != count:
            raise ValueError("truncated GGUF header")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", errors="replace")

    def skip(self, count: int) -> None:
        self._h.seek(count, 1)

    def value(self, value_type: int):
        """Read a value, or skip it cheaply when its contents do not matter."""
        if value_type in _FIXED_FORMATS:
            return struct.unpack(_FIXED_FORMATS[value_type], self.raw(_FIXED_SIZES[value_type]))[0]
        if value_type == _STRING:
            return self.string()
        if value_type == _ARRAY:
            element_type = self.u32()
            count = self.u64()
            if element_type in _FIXED_SIZES:
                # Token lists and similar run to hundreds of thousands of
                # entries; seek past rather than materialising them.
                self.skip(_FIXED_SIZES[element_type] * count)
            elif element_type == _STRING:
                for _ in range(count):
                    self.skip(self.u64())
            else:
                raise ValueError(f"unsupported GGUF array element type {element_type}")
            return None
        raise ValueError(f"unsupported GGUF value type {value_type}")


def read_info(path: Path, max_pairs: int = 2000) -> GgufInfo:
    """Read architecture, layer count and context length from a GGUF file.

    Returns whatever could be determined; fields stay at their defaults when
    the file does not declare them. Never raises for a malformed file - an
    unreadable header simply yields an empty result, and the caller falls back
    to its own heuristics.
    """
    info = GgufInfo()
    try:
        with path.open("rb") as handle:
            reader = _Reader(handle)
            if reader.raw(4) != b"GGUF":
                return info
            reader.u32()               # version
            reader.u64()               # tensor count
            kv_count = reader.u64()

            wanted_suffixes = (".block_count", ".context_length")
            for _ in range(min(kv_count, max_pairs)):
                key = reader.string()
                value_type = reader.u32()
                value = reader.value(value_type)

                if key == "general.architecture" and isinstance(value, str):
                    info.architecture = value
                elif key.endswith(wanted_suffixes) and isinstance(value, int):
                    if key.endswith(".block_count"):
                        info.layers = value
                    else:
                        info.context_length = value

                if info.layers and info.context_length and info.architecture:
                    break
    except (OSError, ValueError, struct.error):
        return info
    return info
