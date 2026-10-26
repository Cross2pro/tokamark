"""Read-only zarr store for per-shot `<shot>.zarr.zip` archives.

The MAST zips contain a top-level `<shot>.zarr/` directory, so keys are transparently prefixed.
Stores are created per access (see MASTStorageManager.make_shot_store), so every DataLoader worker
opens its own zip file handle; nothing is shared across fork. Pickling reopens lazily (ZipStore.__getstate__).
"""
from __future__ import annotations

import io
import json
import os
import threading
import zipfile
from collections.abc import AsyncIterator

from zarr.storage import ZipStore


class PrefixedZipStore(ZipStore):
    def __init__(self, path, prefix: str | None = None):
        super().__init__(path, mode="r", read_only=True)
        if prefix is None:
            prefix = os.path.basename(str(path))[: -len(".zip")] if str(path).endswith(".zip") else ""
        self.prefix = prefix.strip("/")

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def __getstate__(self):
        state = super().__getstate__()
        state["prefix"] = self.prefix
        return state

    def __setstate__(self, state):
        prefix = state.pop("prefix", "")
        super().__setstate__(state)
        self.prefix = prefix

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get(self, key, prototype, byte_range=None):
        return super()._get(self._k(key), prototype, byte_range)

    async def exists(self, key: str) -> bool:
        if not self._is_open:
            self._sync_open()
        return await super().exists(self._k(key))

    async def list(self) -> AsyncIterator[str]:
        if not self._is_open:
            self._sync_open()
        p = self.prefix + "/" if self.prefix else ""
        async for key in super().list():
            if key.startswith(p) and key != p:
                yield key[len(p):]

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.strip("/")
        seen = set()
        async for key in self.list():
            if prefix:
                if not key.startswith(prefix + "/"):
                    continue
                rest = key[len(prefix) + 1:]
            else:
                rest = key
            k = rest.split("/")[0]
            if k and k not in seen:
                seen.add(k)
                yield k


class _SliceFile(io.RawIOBase):
    """Read-only file view of bytes [offset, offset+size) of an underlying file (a tar member)."""

    def __init__(self, path, offset, size):
        self._f = open(path, "rb", buffering=0)
        self._off, self._size, self._pos = offset, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=0):
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, min(self._size, base + pos))
        return self._pos

    def readinto(self, b):
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        self._f.seek(self._off + self._pos)
        got = self._f.readinto(memoryview(b)[:n])
        self._pos += got
        return got

    def close(self):
        try:
            self._f.close()
        finally:
            super().close()


class TarMemberZipStore(PrefixedZipStore):
    """Read a `<shot>.zarr.zip` directly from inside an uncompressed tar batch (no extraction needed).

    The file handle is opened lazily on first access, i.e. inside the DataLoader worker that uses it.
    """

    def __init__(self, tar_path, offset, size, prefix):
        super().__init__(tar_path, prefix=prefix)
        self.tar_offset, self.tar_size = int(offset), int(size)

    def _sync_open(self):
        if self._is_open:
            raise ValueError("store is already open")
        self._lock = threading.RLock()
        self._zf = zipfile.ZipFile(_SliceFile(self.path, self.tar_offset, self.tar_size), mode="r")
        self._is_open = True


_TAR_INDEX_CACHE = {}


def load_tar_index(base_dir):
    """Load `<base_dir>/mast_tar_index.json` ({shot: [tar_path, data_offset, size]}) once per process."""
    path = os.path.join(str(base_dir), "mast_tar_index.json")
    if path not in _TAR_INDEX_CACHE:
        _TAR_INDEX_CACHE[path] = json.load(open(path)) if os.path.isfile(path) else {}
    return _TAR_INDEX_CACHE[path]
