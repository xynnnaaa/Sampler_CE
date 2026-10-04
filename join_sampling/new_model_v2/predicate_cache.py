"""Immutable, run-scoped predicate blocks shared by workers through read-only mmap.

Each row occupies one little-endian uint64 per block of up to 64 predicates.
No QIDs or sampler random state are stored here. Base tables must stay unchanged.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from template_annotations import process_identity, quote_identifier

FORMAT_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path, timings, phase):
    with path.open('a+') as lock:
        with timings.span(phase):
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def map_array(path, dtype, count, mode='r'):
    # numpy.memmap cannot map an empty file.
    return np.memmap(path, dtype=dtype, mode=mode, shape=(count,)) if count else np.empty(0, dtype=dtype)


def close_array(value):
    if isinstance(value, np.memmap):
        value._mmap.close()


class CachedTable:
    def __init__(self, directory, manifest, timings):
        self.directory = directory
        self.manifest = manifest
        self.count = manifest['count']
        self.first = manifest['first']
        self.dense = manifest['dense']
        self.locations = {predicate: (i // manifest['block_size'], i % manifest['block_size'])
                          for i, predicate in enumerate(manifest['predicates'])}
        self.ids = None
        self.blocks = {}
        self.timings = timings

    def ids_array(self):
        if self.ids is None and not self.dense:
            with self.timings.span('predicate_cache.open_mmap'):
                self.ids = map_array(self.directory / 'ids.bin', '<i8', self.count)
            self.timings.counters['predicate_cache.mapped_bytes'] += self.count * 8
        return self.ids

    def block(self, index):
        if index not in self.blocks:
            with self.timings.span('predicate_cache.open_mmap'):
                self.blocks[index] = map_array(self.directory / f'block_{index}.bin', '<u8', self.count)
            self.timings.counters['predicate_cache.blocks_mapped'] += 1
            self.timings.counters['predicate_cache.mapped_bytes'] += self.count * 8
        return self.blocks[index]

    def close(self):
        for value in self.blocks.values():
            close_array(value)
        self.blocks.clear()
        close_array(self.ids)
        self.ids = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SharedPredicateCache:
    def __init__(self, directory, run_id, database_identity, predicates,
                 timings, batch_size=10000, block_size=64, max_bytes=96 * 2**30):
        if not run_id:
            raise ValueError('A common predicate cache run ID is required')
        self.root = Path(directory) / digest([FORMAT_VERSION, run_id, database_identity])
        self.root.mkdir(parents=True, exist_ok=True)
        self.predicates = {table: sorted(set(values.values())) for table, values in predicates.items()}
        self.timings = timings
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.max_bytes = int(max_bytes)
        self.manifests = {}
        if self.batch_size <= 0 or not 1 <= self.block_size <= 64 or self.max_bytes <= 0:
            raise ValueError('Invalid predicate cache batch size, block size, or capacity')

    def _key(self, table):
        return digest([table, self.predicates[table], self.block_size])

    def _read(self, directory, table):
        path = directory / 'manifest.json'
        if not path.exists():
            return None
        manifest = json.loads(path.read_text())
        if (manifest['version'] != FORMAT_VERSION or manifest['table'] != table
                or manifest['predicates'] != self.predicates[table]
                or manifest['block_size'] != self.block_size):
            raise ValueError(f'{table}: incompatible predicate cache manifest')
        for name, size in manifest['files'].items():
            if (directory / name).stat().st_size != size:
                raise ValueError(f'{table}: incomplete predicate cache file {name}')
        return manifest

    def _reserve(self, key, size):
        """Bound persistent block/ID files plus concurrent construction files."""
        with file_lock(self.root / 'capacity.lock', self.timings, 'predicate_cache.capacity_lock_wait'):
            path = self.root / 'capacity.json'
            state = json.loads(path.read_text()) if path.exists() else {'limit': self.max_bytes, 'entries': {}}
            entries = state['entries']
            for other, entry in list(entries.items()):
                ready = (self.root / other / 'manifest.json').exists()
                live = process_identity(entry['pid']) == entry['identity']
                if not ready and not live:
                    entries.pop(other)
            if state['limit'] != self.max_bytes:
                raise ValueError('Workers must share the same predicate cache capacity')
            used = sum(v['bytes'] for other, v in entries.items() if other != key)
            if used + size > self.max_bytes:
                raise MemoryError(f'Predicate cache needs {(used + size)/2**30:.2f} GiB, '
                                  f'exceeding limit {self.max_bytes/2**30:.2f} GiB')
            entries[key] = {'bytes': size, 'pid': os.getpid(), 'identity': process_identity(os.getpid())}
            atomic_json(path, state)

    def _unreserve(self, key):
        with file_lock(self.root / 'capacity.lock', self.timings, 'predicate_cache.capacity_lock_wait'):
            path = self.root / 'capacity.json'
            state = json.loads(path.read_text())
            state['entries'].pop(key, None)
            atomic_json(path, state)

    def ensure(self, conn, table):
        key = self._key(table)
        if key in self.manifests:
            self.timings.counters['predicate_cache.table_hits'] += 1
            return self.manifests[key]
        directory = self.root / key
        manifest = self._read(directory, table)
        if manifest is not None:
            self.timings.counters['predicate_cache.table_hits'] += 1
            self.manifests[key] = manifest
            return manifest
        print(f'    [PredicateCache] Waiting/building {table}', flush=True)
        with file_lock(self.root / (key + '.lock'), self.timings, 'predicate_cache.table_lock_wait'):
            manifest = self._read(directory, table)
            if manifest is not None:
                self.timings.counters['predicate_cache.table_hits'] += 1
                self.manifests[key] = manifest
                return manifest
            # The same table lock guarantees these are abandoned partial builds.
            for partial in self.root.glob(key + '.building.*'):
                shutil.rmtree(partial)
            with self.timings.span('predicate_cache.build'):
                with conn.cursor() as cursor:
                    with self.timings.span('predicate_cache.stats.execute'):
                        cursor.execute(f'SELECT COUNT(*), MIN(id), MAX(id) FROM {quote_identifier(table)}')
                    with self.timings.span('predicate_cache.stats.fetchone'):
                        count, first, last = cursor.fetchone()
                count = int(count)
                first, last = (int(first), int(last)) if count else (0, -1)
                dense = last - first + 1 == count if count else True
                predicates = self.predicates[table]
                nblocks = (len(predicates) + self.block_size - 1) // self.block_size
                files = {f'block_{i}.bin': count * 8 for i in range(nblocks)}
                if not dense:
                    files['ids.bin'] = count * 8
                size = sum(files.values())
                self._reserve(key, size)
                temporary = None
                try:
                    temporary = Path(tempfile.mkdtemp(prefix=key + '.building.', dir=self.root))
                    for name, length in files.items():
                        with (temporary / name).open('wb') as stream:
                            stream.truncate(length)
                            if length and hasattr(os, 'posix_fallocate'):
                                # Fail with ENOSPC before mmap writes can SIGBUS.
                                os.posix_fallocate(stream.fileno(), 0, length)
                    for index in range(nblocks):
                        group = predicates[index*self.block_size:(index+1)*self.block_size]
                        print(f'        [PredicateCache] {table}: block {index+1}/{nblocks}, '
                              f'predicates={len(group)}, rows={count}', flush=True)
                        self._fill_block(conn, table, group, temporary, index, count, first, dense)
                    manifest = dict(version=FORMAT_VERSION, table=table, predicates=predicates,
                                    block_size=self.block_size, count=count, first=first, last=last,
                                    dense=dense, files=files, bytes=size)
                    atomic_json(temporary / 'manifest.json', manifest)
                    os.replace(temporary, directory)
                except BaseException:
                    if temporary is not None:
                        shutil.rmtree(temporary, ignore_errors=True)
                    self._unreserve(key)
                    raise
                self.timings.counters['predicate_cache.tables_built'] += 1
                self.timings.counters['predicate_cache.bytes_built'] += size
                self.timings.counters['predicate_cache.predicates_built'] += len(predicates)
                print(f'    [PredicateCache] Ready {table}: {len(predicates)} predicates, '
                      f'{nblocks} scans, {size/2**30:.3f} GiB', flush=True)
                self.manifests[key] = manifest
                return manifest

    def open(self, conn, table):
        manifest = self.ensure(conn, table)
        return CachedTable(self.root / self._key(table), manifest, self.timings)

    def _fill_block(self, conn, table, predicates, directory, index, count, first, dense):
        # Always use 64 SQL bits. Low bit i represents predicates[i].
        zero = "B'" + '0' * 64 + "'"
        parts = [zero] + [f"(CASE WHEN ({predicate}) THEN B'{1 << i:064b}' ELSE {zero} END)"
                          for i, predicate in enumerate(predicates)]
        statement = f'SELECT id, ({" | ".join(parts)})::text FROM {quote_identifier(table)}'
        if not dense:
            statement += ' ORDER BY id'
        output = map_array(directory / f'block_{index}.bin', '<u8', count, 'r+')
        ids = map_array(directory / 'ids.bin', '<i8', count, 'r+' if index == 0 else 'r') if not dense else None
        seen = np.zeros((count + 7) // 8, dtype=np.uint8) if dense else None
        position = 0
        previous = None
        try:
            with conn.cursor(name='predicate_' + uuid.uuid4().hex) as cursor:
                cursor.itersize = self.batch_size
                with self.timings.span('predicate_cache.scan.execute'):
                    cursor.execute(statement)
                self.timings.counters['predicate_cache.scans'] += 1
                while True:
                    with self.timings.span('predicate_cache.scan.fetchmany'):
                        rows = cursor.fetchmany(self.batch_size)
                    if not rows:
                        break
                    with self.timings.span('predicate_cache.pack_and_store_python'):
                        row_ids = np.fromiter((int(row[0]) for row in rows), dtype=np.int64, count=len(rows))
                        if position + len(rows) > count:
                            raise RuntimeError(f'{table}: row count changed during cache build')
                        if dense:
                            # Compare before subtracting to avoid signed overflow.
                            if np.any(row_ids < first) or np.any(row_ids > first + count - 1):
                                raise RuntimeError(f'{table}: ID range changed during cache build')
                            positions = row_ids - first
                            byte_positions = positions >> 3
                            flags = np.left_shift(np.uint8(1), (positions & 7).astype(np.uint8))
                            if np.unique(positions).size != positions.size or np.any(seen[byte_positions] & flags):
                                raise ValueError(f'{table}: id must be unique')
                            np.bitwise_or.at(seen, byte_positions, flags)
                        else:
                            if ((previous is not None and row_ids[0] <= previous)
                                    or np.any(row_ids[1:] <= row_ids[:-1])):
                                raise ValueError(f'{table}: id must be unique and sorted')
                            positions = slice(position, position + len(rows))
                            if index == 0:
                                ids[positions] = row_ids
                            elif not np.array_equal(ids[positions], row_ids):
                                raise RuntimeError(f'{table}: IDs changed between cache blocks')
                            previous = int(row_ids[-1])
                        raw = b''.join(row[1].encode('ascii') for row in rows)
                        if len(raw) != len(rows) * 64:
                            raise ValueError('Invalid predicate bit-string width')
                        bits = np.frombuffer(raw, dtype=np.uint8).reshape(len(rows), 64)
                        if np.any((bits != ord('0')) & (bits != ord('1'))):
                            raise ValueError('Invalid predicate bit-string')
                        packed = np.packbits(bits[:, ::-1] == ord('1'), axis=1, bitorder='little')
                        output[positions] = packed.view('<u8').reshape(-1)
                        position += len(rows)
                    self.timings.counters['predicate_cache.rows_scanned'] += len(rows)
                    del rows
            if position != count:
                raise RuntimeError(f'{table}: row count changed during cache build')
            if isinstance(output, np.memmap):
                output.flush()
            if index == 0 and isinstance(ids, np.memmap):
                ids.flush()
        finally:
            close_array(output)
            close_array(ids)
