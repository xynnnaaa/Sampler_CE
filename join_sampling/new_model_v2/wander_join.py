import random
from collections import defaultdict
import re
import time
from sampling_timing import SamplingTimings, timed


class WanderJoinEngine:
    def __init__(self, conn, cursor):
        self.conn = conn
        self.cursor = cursor
        self.timings = SamplingTimings()
        self.annotations = None

    def close(self):
        if self.annotations is not None:
            self.annotations.release()
            self.annotations = None

    def _batch_lookup_qid_bitmaps(self, alias, ids):
        if self.annotations is None:
            raise RuntimeError('Template annotations have not been prepared')
        return self.annotations.lookup_many(alias, ids), 0.0

    def connect(self):
        # if not self.conn:
        #     self.conn = psycopg2.connect(**self.db_config)
        #     self.cursor = self.conn.cursor()
        pass 


    def _parse_cond(self, cond_str, my_alias, parent_alias):
        """解析连接条件，返回 (my_col, parent_col)"""
        parts = re.split(r'[=]', cond_str)
        if len(parts) != 2: return None, None
        
        left = parts[0].strip().split('.')
        right = parts[1].strip().split('.')
        
        if left[0] == my_alias and right[0] == parent_alias:
            return left[1], right[1]
        elif right[0] == my_alias and left[0] == parent_alias:
            return right[1], left[1]
        return None, None


    @timed('neighbors.total')
    def _batch_fetch_neighbors(self, table_real_name, my_join_col, parent_vals, sels, alias):
        """
        批量获取邻居。同时查询出sels中所有连接列, 不Join Sidecar, 不查Bitmap
        """
        execute_query_time = 0.0
        if not parent_vals: return {}, execute_query_time
        unique_vals = list(set(parent_vals))

        db_cols = []
        result_keys = []

        # 构造 SQL
        vals_str = ",".join([f"'{v}'" for v in unique_vals])

        filter_sql = f"t.{my_join_col} IN ({vals_str})"

        for sel in sels:
            col_pure = sel.split('.')[-1]
            db_cols.append(f"t.{col_pure}")
            # result_key的格式是"alias.col"
            result_keys.append(sel)
            filter_sql += f" AND t.{col_pure} IS NOT NULL"

        cols_sql = ", ".join(db_cols) if db_cols else "t.id"
        
        sql = f"""
            SELECT {cols_sql}
            FROM {table_real_name} t
            WHERE {filter_sql}
        """

        # 使用temp_partition_filter表来避免SQL长度过长问题
        # self.cursor.execute("TRUNCATE temp_partition_filter")
        # buf = io.StringIO()
        # for val in unique_vals:
        #     buf.write(f"{val}\n")
        # buf.seek(0)
        # self.cursor.copy_from(buf, 'temp_partition_filter', columns=("pid",))
        # sql = f"""
        #     SELECT {cols_sql}
        #     FROM {table_real_name} t
        #     JOIN temp_partition_filter pf ON t.{my_join_col} = pf.pid
        # """

        execute_start = time.perf_counter()
        with self.timings.span('neighbors.execute'):
            self.cursor.execute(sql)
        execute_query_time = time.perf_counter() - execute_start
        with self.timings.span('neighbors.fetchall'):
            rows = self.cursor.fetchall()
        self.timings.counters['neighbors.queries'] += 1
        self.timings.counters['neighbors.rows'] += len(rows)
        self.timings.counters['neighbors.keys'] += len(unique_vals)
        
        # 结果分组
        # neighbors = { parent_join_val: [ row_dict, ... ] }
        neighbors = defaultdict(list)

        try:
            join_col_idx = -1
            for i, col in enumerate(result_keys):
                if col == f"{alias}.{my_join_col}":
                    join_col_idx = i
                    break
            if join_col_idx == -1:
                raise ValueError(f"Join column {my_join_col} not found in result keys.")
        except Exception as e:
            print(f"Error determining join column index: {e}")
            return {}, execute_query_time

        with self.timings.span('neighbors.group_rows_python'):
            for r in rows:
                p_val = str(r[join_col_idx])

                row_data = {}
                for i, key in enumerate(result_keys):
                    row_data[key] = str(r[i]) if r[i] is not None else None
                
                neighbors[p_val].append(row_data)
            
        return neighbors, execute_query_time


    @timed('wander_join.step')
    def extend_paths_one_step(self, active_paths, step_info, pid_map_full, global_map_full, workload_name=""):
        """
        [新增] 蒙特卡洛单步扩展：将父节点传来的所有路径，向当前表进行 1 步 Wander Join。
        """
        self.timings.counters['wander_join.paths_in'] += len(active_paths)
        self.timings.counters['wander_join.steps'] += 1
        self.connect()
        alias = step_info['alias']
        real_name = step_info['real_name']
        parent_alias = step_info['parent']
        raw_cond = step_info['join_condition']
        sel_cols = step_info.get('sels',[])

        my_col, parent_col = self._parse_cond(raw_cond, alias, parent_alias)
        if not my_col: return[]

        parent_key = f"{parent_alias}.{parent_col}"
        with self.timings.span('wander_join.collect_keys_python'):
            batch_vals = []
            for path in active_paths:
                val = path['vals'].get(parent_key)
                if val: batch_vals.append(val)
                else: path['alive'] = False

        if not batch_vals: return[]

        # 批量查库获取邻居
        neighbors, _ = self._batch_fetch_neighbors(real_name, my_col, batch_vals, sel_cols, alias)

        my_global_mask = global_map_full.get(alias, 0)

        pending_bitmap_ids = set()
        path_selections = {}

        with self.timings.span('wander_join.choose_neighbors_python'):
            for i, path in enumerate(active_paths):
                if not path.get('alive', True): continue
            
                p_val = str(path['vals'].get(parent_key))
                candidates = neighbors.get(p_val,[])
            
                if not candidates:
                    path['alive'] = False
                    continue

                # Wander Join 核心：随机选 1 个邻居
                chosen = random.choice(candidates)
                path_selections[i] = chosen

                chosen_id = chosen.get(f"{alias}.id") or chosen.get(f"{alias}.Id")
                if chosen_id:
                    pending_bitmap_ids.add(chosen_id)
                else:
                    path['alive'] = False

        if not pending_bitmap_ids:
            return[]

        # 从当前 template 的内存数组直接读取 QID Bitmap
        translated_map, _ = self._batch_lookup_qid_bitmaps(alias, pending_bitmap_ids)

        # 组装新的存活路径列表，返回给节点缓存
        with self.timings.span('wander_join.intersection_and_path_update_python'):
            surviving_paths =[]
            for i, path in enumerate(active_paths):
                if not path.get('alive', True) or i not in path_selections: continue
            
                chosen = path_selections[i]
                chosen_id = chosen.get(f"{alias}.id") or chosen.get(f"{alias}.Id")
                qid_mask = translated_map.get(chosen_id, my_global_mask)

                # Linear sampler owns each path and its private vals dict:
                # root clones are independent, and callers discard the previous
                # active list. Read shared neighbor rows without modifying them.
                path['acc_bmp'] &= qid_mask
                path['vals'].update(chosen)
                path['alive'] = True
                surviving_paths.append(path)

        self.timings.counters['wander_join.paths_out'] += len(surviving_paths)
        return surviving_paths
