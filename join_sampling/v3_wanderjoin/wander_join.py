import psycopg2
import random
from collections import defaultdict
import re

class WanderJoinEngine:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()

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

    def _batch_fetch_neighbors(self, table_real_name, my_join_col, parent_vals, sels, alias, workload_name=""):
        """
        批量获取邻居。同时查询出sels中所有连接列
        """
        if not parent_vals: return {}
        unique_vals = list(set(parent_vals))

        db_cols = []
        result_keys = []

        for sel in sels:
            col_pure = sel.split('.')[-1]
            db_cols.append(f"t.{col_pure}")
            result_keys.append(sel)

        cols_sql = ", ".join(db_cols) if db_cols else "t.id"
        
        # 构造 SQL
        vals_str = ",".join([f"'{v}'" for v in unique_vals])
        sidecar = f"{table_real_name}_anno_idx_{workload_name}" if workload_name else f"{table_real_name}_anno_idx"
        
        sql = f"""
            SELECT {cols_sql}, s.query_anno::text
            FROM {table_real_name} t
            JOIN {sidecar} s ON t.id = s.query_anno_id
            WHERE t.{my_join_col} IN ({vals_str})
        """
        
        self.cursor.execute(sql)
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
            return {}

        for r in rows:
            bmp_str = r[-1]
            p_val = str(r[join_col_idx])

            row_data = {'_bmp_str': bmp_str}
            for i, key in enumerate(result_keys):
                row_data[key] = str(r[i])
                
            neighbors[p_val].append(row_data)
            
        return neighbors

    def sample_beam_extensions(self, current_beam, lookahead_plan, pid_map_full, global_map_full, k_samples=1, workload_name="", uncovered_mask_int=0):
        """
        [核心] 对 Beam 中的元组进行随机扩展采样。
        
        Args:
            current_beam: List[Dict], T 中的元组列表
            lookahead_plan: List[Dict], 执行计划，第一项是 Rj
            k_samples: 每个元组采样多少条路径
            
        Returns:
            List[Dict]: 扩展后的候选元组列表。
            每个元素包含:
              - 't_idx': 原始 T 元组的下标
              - 'rj_id': 选中的 Rj ID
              - 'rj_bmp': Rj 的真实 Bitmap
              - 'score_bmp': Lookahead 聚合后的潜力 Bitmap
        """
        self.connect()
        
        # 1. 准备并行路径
        # 追踪所有 beam tuple 的 k 次采样
        active_paths = []
        
        for t_idx, t_data in enumerate(current_beam):
            # t_data 是 { 'alias_col': val }
            for _ in range(k_samples):
                active_paths.append({
                    't_idx': t_idx,
                    'vals': t_data.copy(), # Context: { "mi.id": "1", "mi.kind": "2" }
                    'acc_bmp': 0,          # Path Accumulate
                    'rj_data': None,       # 记录这一路选了哪个 Rj
                    'rj_bmp': 0,           # 记录 Rj 本身的 Bitmap
                    'alive': True
                })

        # 2. 执行 Plan (Step 0 是 Rj, Step 1... 是 Lookahead)
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
            for path in active_paths:
                if path['alive']:
                    val = path['vals'].get(parent_key)
                    if val: batch_vals.append(val)
                    else: path['alive'] = False

            if not batch_vals: break

            neighbors = self._batch_fetch_neighbors(real_name, my_col, batch_vals, sel_cols, alias, workload_name)

            my_pid_map = pid_map_full.get(alias, {})
            my_global_mask = global_map_full.get(alias, 0)
            
            for path in active_paths:
                if not path['alive']: continue
                
                p_val = str(path['vals'].get(parent_key))
                candidates = neighbors.get(p_val, [])
                
                if not candidates:
                    path['alive'] = False
                    continue

                # wander join: 随机选一个扩展
                chosen = random.choice(candidates)
                
                # 计算 Bitmap
                c_id = chosen['id'] 

                raw_bmp = chosen['_bmp_str']
                qid_mask = 0
                if raw_bmp:
                    pos = raw_bmp.find('1')
                    while pos != -1:
                        qid_mask |= my_pid_map.get(pos, 0)
                        pos = raw_bmp.find('1', pos + 1)
                qid_mask |= my_global_mask
                
                # 更新路径状态
                path['acc_bmp'] &= qid_mask
                path['vals'].update(chosen)

                if step_idx == 0:
                    path['rj_data'] = chosen
                    path['rj_bmp'] = qid_mask

        # 聚合结果
        proposed_candidates = []
        max_results = defaultdict(int) # key是（t_idx, rj_id），值是能覆盖最多未覆盖查询的数量
        real_bmps = {} # rj_id -> rj_bmp (这个是固定的)
        real_data_map = {}

        rj_alias = lookahead_plan[0]['alias']
        rj_pk_key = f"{rj_alias}.id"
        
        # 遍历所有wander join结果
        for path in active_paths:
            if path['alive'] and path['rj_data'] is not None:
                rj_pk = path['rj_data'][rj_pk_key]
                key = (path['t_idx'], rj_pk)

                if rj_pk not in real_bmps:
                    real_bmps[rj_pk] = path['rj_bmp']
                if key not in real_data_map:
                    real_data_map[key] = path['rj_data']

                score = bin(path['acc_bmp'] & uncovered_mask_int).count('1')
                if score > max_results[key]:
                    max_results[key] = score

        # 格式化输出
        for (t_idx, rj_id), score in max_results.items():
            proposed_candidates.append({
                't_idx': t_idx,
                'rj_data': real_data_map[(t_idx, rj_pk)], # 所需要的Rj的所有数据
                'rj_bmp': real_bmps[rj_id],
                'score': score
            })
            
        return proposed_candidates