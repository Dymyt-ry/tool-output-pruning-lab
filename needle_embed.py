"""Needle 3 embeddings through its C API, without its Python package.

Cactus Compute's Needle 3 (Apache-2.0, https://github.com/cactus-compute/needle):
libneedle3.so from the pinned wheel, needle3.cact from the pinned model repo, both
checked against their sha256. The library imports no socket, exec or fork symbol;
run it under `unshare -rn` anyway, so it cannot reach the network.
"""
import ctypes
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent / ".models" / "needle3"


class Needle:
    def __init__(self):
        lib = ctypes.CDLL(str(HERE / "libneedle3.so"))
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong]
        lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.needle_embed.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.needle_last_error.restype = ctypes.c_char_p
        self.lib = lib
        blob = (HERE / "needle3.cact").read_bytes()
        self._blob = blob  # the engine reads the container in place
        if lib.needle_load(blob, len(blob)) < 0:
            raise RuntimeError(lib.needle_last_error())
        if lib.needle_init(b"", b"[]", None) < 0:
            raise RuntimeError(lib.needle_last_error())
        self.dim = lib.needle_embed(b"x", None, 0)

    def embed(self, text: str) -> list[float]:
        buf = (ctypes.c_float * self.dim)()
        n = self.lib.needle_embed(text.encode("utf-8", "replace"), buf, self.dim)
        if n < 0:
            raise RuntimeError(self.lib.needle_last_error())
        v = list(buf)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
