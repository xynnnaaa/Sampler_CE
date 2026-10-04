"""Per-template, alias-specific QID masks in contiguous little-endian buffers."""
from array import array
from bisect import bisect_left
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def quote_identifier(name):
    return '.'.join('"' + part.replace('"', '""') + '"' for part in name.split('.'))


def process_identity(pid):
    """Linux process start time, also preventing stale leases after PID reuse."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text.rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class SharedMemoryBudget:
    """Reserve an entire template before allocation; no partial-reservation deadlock.

    This limits annotation buffers only, not the entire RSS or PostgreSQL memory.
    Workers using the same path must use the same limit. Dead-process reservations
    are reclaimed on the next access. The JSON ledger contains no bitmap data.
    """
    def __init__(self, path, limit_bytes):
        self.path = Path(path)
        self.limit_bytes = int(limit_bytes)
        if self.limit_bytes <= 0:
            raise ValueError('annotation_memory_budget_gib must be positive')
        self.token = uuid.uuid4().hex
        self.pid = os.getpid()
        self.identity = process_identity(self.pid)
        self.reserved = False

    def _update(self, requested=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock on a separate inode so atomic ledger replacement is safe.
        with Path(str(self.path) + '.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            raw = self.path.read_text() if self.path.exists() else ''
            state = json.loads(raw) if raw else {'limit': self.limit_bytes, 'leases': {}}
            leases = state['leases']
            for key, lease in list(leases.items()):
                if process_identity(lease['pid']) != lease['identity']:
                    del leases[key]
            # An idle ledger can be reused by a later run with a different limit.
            if leases and state['limit'] != self.limit_bytes:
                raise ValueError('Workers sharing an annotation budget must use the same limit')
            state['limit'] = self.limit_bytes
            accepted = False
            if requested is None:
                leases.pop(self.token, None)
            elif sum(lease['bytes'] for lease in leases.values()) + requested <= self.limit_bytes:
                leases[self.token] = {'pid': self.pid, 'identity': self.identity, 'bytes': requested}
                accepted = True
            # Killing a worker during a write must not corrupt other reservations.
            descriptor, temporary = tempfile.mkstemp(prefix=self.path.name + '.', dir=self.path.parent)
            try:
                with os.fdopen(descriptor, 'w') as ledger:
                    json.dump(state, ledger)
                    ledger.flush()
                    os.fsync(ledger.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return accepted

    def acquire(self, requested):
        requested = int(requested)
        if requested < 0:
            raise ValueError('Requested annotation bytes must not be negative')
        if self.reserved:
            raise RuntimeError('A worker cannot reserve two templates at once')
        if requested > self.limit_bytes:
            raise MemoryError(f'Template needs {requested / 2**30:.2f} GiB of annotation '
                              f'buffers, exceeding shared budget {self.limit_bytes / 2**30:.2f} GiB')
        if not requested:
            return
        last_notice = 0.0
        while not self._update(requested):
            now = time.monotonic()
            if now - last_notice >= 30:
                print(f'    [Annotations] Waiting for {requested / 2**30:.2f} GiB '
                      'of shared annotation budget...', flush=True)
                last_notice = now
            time.sleep(1)
        self.reserved = True

    def release(self):
        if self.reserved:
            self._update()
            self.reserved = False


@dataclass
class AliasPlan:
    alias: str
    table: str
    predicate_masks: dict
    global_mask: int
    row_count: int = 0
    first_id: int = 0
    last_id: int = -1
    dense: bool = True


class AliasBitmaps:
    """Rows occupy ceil(Q/64)*8 bytes, equivalent to a uint64 matrix.

    bytearray avoids a NumPy dependency and never creates a Python int per
    stored row. Only the requested rows are decoded into temporary Python ints.
    Sparse IDs are stored in a sorted signed-64-bit array, not a Python dict.
    """
    def __init__(self, plan, row_bytes):
        self.row_bytes = row_bytes
        self.row_count = plan.row_count
        self.first_id = plan.first_id
        self.constant = plan.global_mask if not plan.predicate_masks else None
        self.buffer = bytearray() if self.constant is not None else bytearray(plan.row_count * row_bytes)
        self.ids = (array('q', [0]) * plan.row_count
                    if self.constant is None and not plan.dense else None)
        if self.ids is not None and self.ids.itemsize != 8:
            raise RuntimeError('Sparse IDs require 64-bit array elements')

    @property
    def nbytes(self):
        return len(self.buffer) + (len(self.ids) * self.ids.itemsize if self.ids is not None else 0)

    def lookup(self, row_id, default):
        if self.constant is not None:
            return self.constant
        row_id = int(row_id)
        if self.ids is None:
            position = row_id - self.first_id
            if not 0 <= position < self.row_count:
                return default
        else:
            position = bisect_left(self.ids, row_id)
            if position == self.row_count or self.ids[position] != row_id:
                return default
        offset = position * self.row_bytes
        # A view prevents copying the slice of the full annotation buffer.
        return int.from_bytes(memoryview(self.buffer)[offset:offset + self.row_bytes], 'little')


class TemplateAnnotations:
    def __init__(self, timings, batch_size=10000, budget=None, table_stats_cache=None):
        self.timings = timings
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('annotation_batch_size must be positive')
        self.budget = budget
        self.aliases = {}
        self.query_count = 0
        self.nbytes = 0
        self.global_masks = {}
        # Owned by the sampler/connection and retained across template releases.
        # The workload's base tables must remain unchanged during this run.
        self.table_stats_cache = {} if table_stats_cache is None else table_stats_cache

    @staticmethod
    def sql_expression(plan, query_count):
        """Each distinct conjunction is evaluated once in this expression.

        Bit-string leftmost position is QID Q-1, so int(text, 2) == QID mask.
        SQL CASE treats NULL predicates as false, matching the original anno.
        """
        zero = "B'" + '0' * query_count + "'"
        literal = lambda mask: "B'" + format(mask, f'0{query_count}b') + "'"
        parts = [literal(plan.global_mask)]
        for predicate, mask in plan.predicate_masks.items():
            parts.append(f'(CASE WHEN ({predicate}) THEN {literal(mask)} ELSE {zero} END)')
        return '(' + ' | '.join(parts) + f')::bit varying({query_count})'

    def prepare(self, conn, graph, instances, pid_to_pred):
        if self.aliases or self.nbytes:
            raise RuntimeError('Previous template annotations must be released first')
        self.query_count = len(instances)
        if not self.query_count:
            raise ValueError('Cannot annotate an empty template')
        row_bytes = ((self.query_count + 63) // 64) * 8
        plans = []
        table_stats = self.table_stats_cache
        try:
            with self.timings.span('annotations.prepare'):
                with self.timings.span('annotations.plan_python'):
                    for alias, info in graph.nodes(data=True):
                        table = info['real_name']
                        masks = {}
                        global_mask = 0
                        for qid, instance in enumerate(instances):
                            pid = instance[alias]
                            if pid == -1:
                                global_mask |= 1 << qid
                            else:
                                predicate = pid_to_pred[table][pid]
                                masks[predicate] = masks.get(predicate, 0) | (1 << qid)
                        plans.append(AliasPlan(alias, table, masks, global_mask))
                # Exact counts avoid relying on hardcoded cardinalities/reltuples.
                # Only the first filtered use of each table scans its metadata.
                for plan in plans:
                    if not plan.predicate_masks:
                        continue
                    if plan.table not in table_stats:
                        self.timings.counters['annotations.stats.cache_misses'] += 1
                        with conn.cursor() as cursor:
                            with self.timings.span('annotations.stats.execute'):
                                cursor.execute(f'SELECT COUNT(*), MIN(id), MAX(id) FROM '
                                               f'{quote_identifier(plan.table)}')
                            with self.timings.span('annotations.stats.fetchone'):
                                table_stats[plan.table] = cursor.fetchone()
                    else:
                        self.timings.counters['annotations.stats.cache_hits'] += 1
                    count, first, last = table_stats[plan.table]
                    plan.row_count = int(count)
                    if count:
                        plan.first_id, plan.last_id = int(first), int(last)
                        plan.dense = plan.last_id - plan.first_id + 1 == count
                estimate = sum(plan.row_count * (row_bytes + (0 if plan.dense else 8))
                               for plan in plans if plan.predicate_masks)
                # One transient bit per dense row checks uniqueness despite the
                # unordered stream. Aliases are filled sequentially.
                scratch_bytes = max(((plan.row_count + 7) // 8 for plan in plans
                                     if plan.predicate_masks and plan.dense), default=0)
                estimate += scratch_bytes
                self.timings.counters['annotations.validation_scratch_bytes'] = scratch_bytes
                self.timings.counters['annotations.planned_bytes'] = estimate
                print(f'    [Annotations] QIDs={self.query_count}, bytes/row={row_bytes}, '
                      f'planned={estimate / 2**30:.3f} GiB', flush=True)
                if self.budget is not None:
                    with self.timings.span('annotations.budget_wait'):
                        self.budget.acquire(estimate)
                for plan in plans:
                    with self.timings.span('annotations.allocate_python'):
                        store = AliasBitmaps(plan, row_bytes)
                        self.aliases[plan.alias] = store
                        self.global_masks[plan.alias] = plan.global_mask
                        self.nbytes += store.nbytes
                    if not plan.predicate_masks:
                        self.timings.counters['annotations.constant_aliases'] += 1
                        continue
                    print(f'        [Annotations] {plan.alias} ({plan.table}): '
                          f'rows={plan.row_count}, '
                          f'layout={"dense" if plan.dense else "sparse"}, '
                          f'buffer={store.nbytes / 2**30:.3f} GiB', flush=True)
                    self._fill(conn, plan, store)
                self.timings.counters['annotations.buffer_bytes'] = self.nbytes
                self.timings.counters['annotations.aliases'] = len(self.aliases)
                print(f'    [Annotations] Ready: {self.nbytes / 2**30:.3f} GiB', flush=True)
        except BaseException:
            self.release()
            raise

    def _fill(self, conn, plan, store):
        # A named psycopg2 cursor streams FETCH batches, unlike client fetchmany.
        name = 'qid_anno_' + uuid.uuid4().hex
        with self.timings.span(f'annotations.build_alias.{plan.alias}'):
            with self.timings.span('annotations.sql_build_python'):
                expression = self.sql_expression(plan, self.query_count)
                statement = (f'SELECT id, ({expression})::text FROM '
                             f'{quote_identifier(plan.table)}')
                if not plan.dense:
                    statement += ' ORDER BY id'
            with self.timings.span('annotations.allocate_python'):
                seen = bytearray((plan.row_count + 7) // 8) if plan.dense else None
            position = 0
            previous_id = None
            with conn.cursor(name=name) as cursor:
                cursor.itersize = self.batch_size
                with self.timings.span('annotations.scan.execute'):
                    cursor.execute(statement)
                self.timings.counters['annotations.scans'] += 1
                while True:
                    with self.timings.span('annotations.scan.fetchmany'):
                        rows = cursor.fetchmany(self.batch_size)
                    if not rows:
                        break
                    with self.timings.span('annotations.decode_and_store_python'):
                        for row_id, raw_mask in rows:
                            row_id = int(row_id)
                            if position >= plan.row_count:
                                raise RuntimeError(f'{plan.table}: row count changed during annotation')
                            if store.ids is None:
                                target_position = row_id - plan.first_id
                                if not 0 <= target_position < plan.row_count:
                                    raise RuntimeError(f'{plan.table}: ID range changed during annotation')
                                seen_byte, seen_bit = divmod(target_position, 8)
                                flag = 1 << seen_bit
                                if seen[seen_byte] & flag:
                                    raise ValueError(f'{plan.table}: id must be unique')
                                seen[seen_byte] |= flag
                            else:
                                if previous_id is not None and row_id <= previous_id:
                                    raise ValueError(f'{plan.table}: id must be unique and sorted')
                                if not -(1 << 63) <= row_id < (1 << 63):
                                    raise ValueError('Sparse IDs must fit signed 64-bit integers')
                                store.ids[position] = row_id
                                target_position = position
                            if len(raw_mask) != self.query_count:
                                raise ValueError('Invalid QID bitmap returned by database')
                            mask = int(raw_mask, 2)
                            start = target_position * store.row_bytes
                            store.buffer[start:start + store.row_bytes] = mask.to_bytes(store.row_bytes, 'little')
                            previous_id = row_id
                            position += 1
                    self.timings.counters['annotations.rows_built'] += len(rows)
                    # Avoid keeping the old batch alive while fetching the next one.
                    del rows
            if position != plan.row_count:
                raise RuntimeError(f'{plan.table}: row count changed during annotation')

    def lookup_many(self, alias, ids):
        with self.timings.span('annotations.lookup.total'):
            store = self.aliases[alias]
            with self.timings.span('annotations.lookup_python'):
                # Default matches the old missing-sidecar-row global-mask fallback.
                result = {str(row_id): store.lookup(row_id, self.global_masks[alias])
                          for row_id in set(ids)}
            self.timings.counters['annotations.lookup_ids'] += len(result)
            return result

    def release(self):
        with self.timings.span('annotations.release_python'):
            self.aliases.clear()
            self.nbytes = 0
            self.query_count = 0
            self.global_masks = {}
            if self.budget is not None:
                self.budget.release()
