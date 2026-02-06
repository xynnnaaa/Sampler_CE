import psycopg2
import random
from collections import defaultdict
import re
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

class WanderJoinEngine:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.connect()

        self.bitmap_cache = {}

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            self.cursor.execute("""
                CREATE TEMP TABLE IF NOT EXISTS temp_partition_filter (
                    pid bigint
                ) ON COMMIT PRESERVE ROWS;
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS temp_partition_filter_pid_idx
                ON temp_partition_filter(pid);
            """)

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()

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

    def _translate_pid_bitmap(self, pid_bitmap_str, pid_map, global_mask_int):
        """
        将数据库读出的 PID Bitmap (str) 翻译为 QID Bitmap (int).
        Args:
            pid_bitmap_str: 例如 '10100...' (Postgres BIT VARYING)
            pid_map: { pid(int): qid_mask(int) }
            global_mask_int: 该表的全局 QID Mask (int)
        Returns:
            qid_mask (int)
        """
        result_mask = global_mask_int
        
        if not pid_bitmap_str:
            return result_mask
        
        pos = pid_bitmap_str.find('1')
        while pos != -1:
            result_mask |= pid_map.get(pos, 0)
            pos = pid_bitmap_str.find('1', pos + 1)
        
        return result_mask


    def _batch_fetch_neighbors(self, table_real_name, my_join_col, parent_vals, sels, alias):
        """
        批量获取邻居。同时查询出sels中所有连接列, 不Join Sidecar, 不查Bitmap
        """
        execute_query_time = 0.0
        if not parent_vals: return {}, execute_query_time
        unique_vals = list(set(parent_vals))

        db_cols = []
        result_keys = []

        for sel in sels:
            col_pure = sel.split('.')[-1]
            db_cols.append(f"t.{col_pure}")
            # result_key的格式是"alias.col"
            result_keys.append(sel)

        cols_sql = ", ".join(db_cols) if db_cols else "t.id"
        
        # 构造 SQL
        # vals_str = ",".join([f"'{v}'" for v in unique_vals])
        
        # sql = f"""
        #     SELECT {cols_sql}
        #     FROM {table_real_name} t
        #     WHERE t.{my_join_col} IN ({vals_str})
        # """

        # 使用temp_partition_filter表来避免SQL长度过长问题
        self.cursor.execute("TRUNCATE temp_partition_filter")
        buf = io.StringIO()
        for val in unique_vals:
            buf.write(f"{val}\n")
        buf.seek(0)
        self.cursor.copy_from(buf, 'temp_partition_filter', columns=("pid",))
        sql = f"""
            SELECT {cols_sql}
            FROM {table_real_name} t
            JOIN temp_partition_filter pf ON t.{my_join_col} = pf.pid
        """

        execute_start = time.time()
        self.cursor.execute(sql)
        execute_query_time = time.time() - execute_start
        rows = self.cursor.fetchall()
        
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

        for r in rows:
            p_val = str(r[join_col_idx])

            row_data = {}
            for i, key in enumerate(result_keys):
                row_data[key] = str(r[i])
                
            neighbors[p_val].append(row_data)
            
        return neighbors, execute_query_time
    
    def _batch_fetch_translate_bitmaps(self, table_real_name, ids, workload_name="", alias="", pid_map={}, global_map=0):
        """
        [新增] 给定一批主键 ID，批量从 Sidecar 表获取 Bitmap 并翻译，存入缓存。
        """
        execute_query_time = 0.0
        unique_ids = list(set(ids))

        result_map = {} # {id_str: qid_int}
        missing_ids = []

        # 统一使用字符串形式的 id 作为 cache key，保证一致性
        for uid in unique_ids:
            uid_str = str(uid)
            cache_key = (alias, uid_str)
            if cache_key in self.bitmap_cache:
                result_map[uid_str] = self.bitmap_cache[cache_key]
            else:
                missing_ids.append(uid_str)

        if missing_ids:
            # vals_str = ",".join([f"'{v}'" for v in missing_ids])
            # sidecar = f"{table_real_name}_anno_idx_{workload_name}" if workload_name else f"{table_real_name}_anno_idx"

            # # 直接查 Sidecar
            # sql = f"""
            #     SELECT query_anno_id, query_anno::text
            #     FROM {sidecar}
            #     WHERE query_anno_id IN ({vals_str})
            # """


            # 同样用 temp_partition_filter 来避免 SQL 长度问题
            self.cursor.execute("TRUNCATE temp_partition_filter")
            buf = io.StringIO()
            for mid in missing_ids:
                buf.write(f"{mid}\n")
            buf.seek(0)
            self.cursor.copy_from(buf, 'temp_partition_filter', columns=("pid",))
            sidecar = f"{table_real_name}_anno_idx_{workload_name}" if workload_name else f"{table_real_name}_anno_idx"
            sql = f"""
                SELECT query_anno_id, query_anno::text
                FROM {sidecar}
                JOIN temp_partition_filter pf ON {sidecar}.query_anno_id = pf.pid
            """
            execute_start = time.time()
            self.cursor.execute(sql)
            execute_query_time += time.time() - execute_start
            rows = self.cursor.fetchall()
            for r in rows:
                rid = str(r[0])
                raw_bmp_str = r[1]
                qid_mask = self._translate_pid_bitmap(raw_bmp_str, pid_map, global_map)
                self.bitmap_cache[(alias, rid)] = qid_mask
                result_map[rid] = qid_mask

        return result_map, execute_query_time

    def sample_beam_extensions(self, current_beam, lookahead_plan, pid_map_full, global_map_full, k_samples=1, workload_name="", uncovered_mask_int=0):
        """
        [核心] 对 Beam 中的元组进行随机扩展采样。
        
        Args:
            current_beam: List[Dict], T 中的元组列表, 每个元素包含 {'data': dict, 'bmp': int}
            lookahead_plan: List[Dict], 执行计划，第一项是 Rj
            k_samples: 每个元组采样多少条路径
            
        Returns:
            List[Dict]: 扩展后的候选元组列表。
            每个元素包含:
              - 't_idx': 原始 T 元组的下标
              - 'rj_id': 选中的 Rj ID
              - 'rj_bmp': Rj 的真实 Bitmap
              - 'score': Lookahead 聚合后的潜力
        """
        self.connect()

        execute_query_time = 0.0
        
        # 记录所有 beam tuple 的 k 次采样
        active_paths = []

        # 计时统计 - 初始化累计变量
        total_batch_fetch_time = 0.0
        batch_fetch_calls = 0

        total_translate_time = 0.0
        translate_calls = 0

        total_init_active_paths_time = 0.0
        total_build_batch_vals_time = 0.0
        total_batch_vals = 0
        total_selection_time = 0.0
        total_candidates_chosen = 0
        total_update_paths_time = 0.0
        total_update_count = 0
        total_aggregation_time = 0.0

        # 初始化 active_paths（计时）
        t_init0 = time.time()
        for t_idx, t_item in enumerate(current_beam):
            t_data = t_item['data']
            t_bmp = t_item['bmp']

            # t_data 是 { 'alias_col': val }
            for _ in range(k_samples):
                active_paths.append({
                    't_idx': t_idx,
                    'vals': t_data.copy(), # Context: { "mi.id": "1", "mi.kind": "2" }
                    'acc_bmp': t_bmp,          # Path Accumulate
                    'rj_data': None,       # 记录这一路选了哪个 Rj
                    'rj_bmp': 0,           # 记录 Rj 本身的 Bitmap
                    'alive': True
                })
        t_init1 = time.time()
        total_init_active_paths_time += (t_init1 - t_init0)

        # 执行 Plan (Step 0 是 Rj, Step 1... 是 Lookahead)
        for step_idx, step in enumerate(lookahead_plan):
            alias = step['alias']
            real_name = step['real_name']
            parent_alias = step['parent']
            raw_cond = step['join_condition']
            sel_cols = step.get('sels', [])

            my_col, parent_col = self._parse_cond(raw_cond, alias, parent_alias)
            if not my_col: continue

            parent_key = f"{parent_alias}.{parent_col}"
            batch_vals = []
            # 收集 batch_vals（计时）
            t_bvals0 = time.time()
            for path in active_paths:
                if path['alive']:
                    val = path['vals'].get(parent_key)
                    if val:
                        batch_vals.append(val)
                    else:
                        path['alive'] = False
            t_bvals1 = time.time()
            total_build_batch_vals_time += (t_bvals1 - t_bvals0)
            total_batch_vals += len(batch_vals)

            if not batch_vals: break

            # 计时：_batch_fetch_neighbors
            t_bf0 = time.time()
            neighbors, execute_time = self._batch_fetch_neighbors(real_name, my_col, batch_vals, sel_cols, alias)
            t_bf1 = time.time()
            batch_fetch_calls += 1
            total_batch_fetch_time += (t_bf1 - t_bf0)
            execute_query_time += execute_time

            my_pid_map = pid_map_full.get(alias, {})
            my_global_mask = global_map_full.get(alias, 0)

            pending_bitmap_ids = set()
            path_selections = {}
            # 为每个 path 选择候选并收集 pending ids（计时）
            t_sel0 = time.time()
            chosen_count = 0
            for i, path in enumerate(active_paths):
                if not path['alive']: continue

                p_val = str(path['vals'].get(parent_key))
                candidates = neighbors.get(p_val, [])

                if not candidates:
                    path['alive'] = False
                    continue

                # wander join: 随机选一个扩展
                chosen = random.choice(candidates)
                path_selections[i] = chosen
                chosen_count += 1

                chosen_id = chosen.get(f"{alias}.id") or chosen.get(f"{alias}.Id")
                if chosen_id:
                    pending_bitmap_ids.add(chosen_id)
                else:
                    path['alive'] = False
            t_sel1 = time.time()
            total_selection_time += (t_sel1 - t_sel0)
            total_candidates_chosen += chosen_count

            if not pending_bitmap_ids:
                continue

            # 批量获取并翻译 Bitmap

            t_tr0 = time.time()
            translated_map, execute_time = self._batch_fetch_translate_bitmaps(
                real_name, 
                pending_bitmap_ids, 
                workload_name=workload_name, 
                alias=alias, 
                pid_map=my_pid_map, 
                global_map=my_global_mask
            )
            t_tr1 = time.time()
            translate_calls += 1
            total_translate_time += (t_tr1 - t_tr0)
            execute_query_time += execute_time

            # 更新 path 状态（计时）
            t_up0 = time.time()
            update_count = 0
            for i, chosen in path_selections.items():
                path = active_paths[i]
                chosen_id = chosen.get(f"{alias}.id") or chosen.get(f"{alias}.Id")
                qid_mask = translated_map.get(chosen_id, my_global_mask)
                path['acc_bmp'] &= qid_mask
                path['vals'].update(chosen)
                update_count += 1
                if step_idx == 0:
                    path['rj_data'] = chosen
                    path['rj_bmp'] = qid_mask
            t_up1 = time.time()
            total_update_paths_time += (t_up1 - t_up0)
            total_update_count += update_count

        # 聚合结果
        proposed_candidates = []
        # 结构: { (t_idx, rj_id): {'score': max_score, 'rj_data': data, 'rj_bmp': bmp} }
        best_results = {}

        rj_alias = lookahead_plan[0]['alias']
        rj_pk_key = f"{rj_alias}.id"
        
        # 遍历所有wander join结果
        # 聚合结果（计时）
        t_ag0 = time.time()
        for path in active_paths:
            if path['alive'] and path['rj_data'] is not None:
                rj_pk = path['rj_data'][rj_pk_key]
                key = (path['t_idx'], rj_pk)

                score = bin(path['acc_bmp'] & uncovered_mask_int).count('1')
                
                if key not in best_results or score > best_results[key]['score']:
                    best_results[key] = {
                        'score': score,
                        'rj_data': path['rj_data'],
                        'rj_bmp': path['rj_bmp']
                    }
        t_ag1 = time.time()
        total_aggregation_time += (t_ag1 - t_ag0)

        # 格式化输出
        for (t_idx, rj_id), res in best_results.items():
            proposed_candidates.append({
                't_idx': t_idx,
                'rj_data': res['rj_data'], # 所需要的Rj的所有数据
                'rj_bmp': res['rj_bmp'],
                'score': res['score']
            })

        print(f"                _batch_fetch_neighbors called {batch_fetch_calls} times, total time: {total_batch_fetch_time:.4f}s")
        print(f"                _translate_pid_bitmap called {translate_calls} times, total time: {total_translate_time:.4f}s")
        # # 新增详细计时输出
        # print(f"                init active_paths time: {total_init_active_paths_time:.6f}s")
        # print(f"                build batch_vals total time: {total_build_batch_vals_time:.6f}s, total batch_vals collected: {total_batch_vals}")
        # print(f"                selection total time: {total_selection_time:.6f}s, candidates chosen: {total_candidates_chosen}")
        # print(f"                update paths total time: {total_update_paths_time:.6f}s, updates applied: {total_update_count}")
        # print(f"                aggregation time: {total_aggregation_time:.6f}s")

        return proposed_candidates, execute_query_time