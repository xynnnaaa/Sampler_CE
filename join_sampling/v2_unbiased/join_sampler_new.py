import pickle
import glob
import os
import io
from collections import defaultdict
from networkx.readwrite import json_graph
import networkx as nx
import time
import json
import psycopg2
from psycopg2.extras import execute_values
import sys
import re
import heapq

import sqlglot
from sqlglot import exp

# [Added] Import the Engine
from Sampler.join_sampling.v2_unbiased.unbiased_sample_join import JoinSamplingEngine

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def normalize_condition(cond_str):
    if not cond_str:
        return "None"
    parts = [p.strip() for p in cond_str.split('=')]
    parts.sort()
    return "=".join(parts)

TABLE_CARD = {
    "cast_info": 37610440,
    "movie_info": 13399115,
    "name": 5245495,
    "title": 4733511,
    "movie_keyword": 4523930,
    "char_name": 3140339,
    "person_info": 2963664,
    "movie_companies": 2609129,
    "movie_info_idx": 1380035,
    "aka_name": 901343,
    "aka_title": 361472,
    "company_name": 234997,
    "complete_cast": 135086,
    "keyword": 134170,
    "movie_link": 29997,
    "info_type": 113,
    "link_type": 18,
    "role_type": 12,
    "kind_type": 7,
    "comp_cast_type": 4,
    "company_type": 4,
}

class JoinSampler:
    def __init__(self, config_path: str):
        print(f"Initializing JoinSampler with config: {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        samp_conf = self.config.get("sampling", {})
        self.base_query_dir = samp_conf.get("base_query_dir", "./qrep")
        self.workload_name = samp_conf.get("workload_name", "")
        self.output_path = samp_conf.get("output_path", "./samples.json")
        self.skip_7a = samp_conf.get("skip_7a", True)

        self.m_partitions = samp_conf.get("m_partitions", 10)
        self.k_bitmaps = samp_conf.get("k_bitmaps", 5)
        self.limit_x = samp_conf.get("limit_x", 50)
        self.batch_size = samp_conf.get("batch_size", 1000)
        
        # [Added] Lookahead configuration
        self.lookahead_k = samp_conf.get("lookahead_k", 5) # 向后看几步
        self.lookahead_samples = samp_conf.get("lookahead_samples", 500) # 采样次数

        print("Configuration Loaded:")
        print(f"  Base Query Directory: {self.base_query_dir}")
        print(f"  Workload Name: {self.workload_name}")
        print(f"  Output Path: {self.output_path}")
        print(f"  Skip Template '7a': {self.skip_7a}")
        print(f"  M Partitions: {self.m_partitions}")
        print(f"  K Bitmaps: {self.k_bitmaps}")
        print(f"  Limit X: {self.limit_x}")
        print(f"  Lookahead K: {self.lookahead_k}")

        self.join_templates = {}
        self.global_predicate_map = defaultdict(lambda: {}) 
        self.global_pid_counters = defaultdict(int)         

        db_conf = self.config.get("database", {})
        self.db_config = {
            "host": db_conf.get("host", "localhost"),
            "port": db_conf.get("port", 5432),
            "dbname": db_conf.get("dbname", "imdb"),
            "user": db_conf.get("user", "your_username")
        }

        try:
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

            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise e
        
        # [Added] Initialize Engine
        self.engine = JoinSamplingEngine(self.db_config)

        import itertools
        self.tie_breaker = itertools.count()

    def close(self):
        print("Closing database connection...")
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        # [Added] Close Engine
        if self.engine:
            self.engine.close()
        print("Resources released.")

    def _remove_alias_safe(self, pred_sql, alias):
        """
        使用 sqlglot 安全地去除谓词中的表别名。
        例如: "t.kind_id = 1" -> "kind_id = 1"
        """
        try:
            expression = sqlglot.parse_one(pred_sql, read="postgres")

            for column in expression.find_all(exp.Column):
                if column.table == alias:
                    column.set("table", None)

            return expression.sql(dialect="postgres")
            
        except Exception as e:
            print(f"Warning: sqlglot failed on '{pred_sql}': {e}")
            pattern = fr"\b{alias}\."
            return re.sub(pattern, "", pred_sql).strip()


    def load_and_parse_workload(self):
        self.all_involved_tables = set()

        temp_groups = defaultdict(list)
        temp_graphs = {}

        try:
            template_names = [d for d in os.listdir(self.base_query_dir) if os.path.isdir(os.path.join(self.base_query_dir, d))]
        except FileNotFoundError:
            print(f"ERROR: Base query directory not found: {self.base_query_dir}")
            return
        
        if not template_names:
            print(f"No template subdirectories found in '{self.base_query_dir}'.")
            return
        
        for template_name in sorted(template_names):
            if template_name == "7a" and self.skip_7a:
                print("Skipping template '7a' as per configuration.")
                continue  # Skip known problematic template
            input_template_dir = os.path.join(self.base_query_dir, template_name)
            pkl_files = sorted(glob.glob(os.path.join(input_template_dir, "*.pkl")))
            if not pkl_files:
                print(f"No .pkl files found in '{input_template_dir}'. Skipping this template.")
                continue
            for pkl_file in pkl_files:
                try:
                    qrep = load_qrep(pkl_file)
                except Exception as e:
                    print(f"Error loading {pkl_file}: {e}")
                    continue

                join_graph = qrep["join_graph"]
                subset_graph = qrep["subset_graph"]

                for subplan_tuple in sorted(subset_graph.nodes()):
                    # check 这里跳过了单表template
                    if len(subplan_tuple) < 2:
                        continue

                    sorted_aliases = sorted(list(subplan_tuple))
                    sub_graph = join_graph.subgraph(subplan_tuple)
                    edges_info = []

                    for u, v, data in sub_graph.edges(data=True):
                        if u > v:
                            u, v = v, u

                        cond = data.get("join_condition", "")
                        if not cond:
                            print(f"Warning: No join condition found between {u} and {v} in {pkl_file}")
                            continue

                        cond_clean = normalize_condition(cond)
                        edges_info.append(f"{u}|{v}|{cond_clean}")

                    edges_info.sort()
                    join_sig_str = "||".join(edges_info)

                    template_key = (tuple(sorted_aliases), join_sig_str)

                    # 存储当前查询的谓词-PID 映射
                    current_query_pids = {} # {alias: pid}

                    for alias in sorted_aliases:
                        node_data = join_graph.nodes[alias]
                        real_name = node_data["real_name"]

                        self.all_involved_tables.add(real_name)

                        preds_list = node_data.get("predicates", [])

                        clean_pred_list = [] # 没有别名的谓词
                        for pred in preds_list:
                            pattern = fr"\b{alias}\."
                            pred_clean = re.sub(pattern, "", pred).strip()
                            clean_pred_list.append(pred_clean)

                        if clean_pred_list:
                            combined_pred = " AND ".join(clean_pred_list)
                            if combined_pred not in self.global_predicate_map[real_name]:
                                pid = self.global_pid_counters[real_name]
                                # 从0开始编号
                                self.global_predicate_map[real_name][combined_pred] = pid
                                self.global_pid_counters[real_name] += 1

                            # 这里存 pid，不用存字符串了
                            current_pid = self.global_predicate_map[real_name][combined_pred]
                            current_query_pids[alias] = current_pid
                        else:
                            current_query_pids[alias] = -1  # 表示没有谓词

                    temp_groups[template_key].append(current_query_pids)

                    if template_key not in temp_graphs:
                        # 创建子图副本，保留边上的连接条件信息
                        temp_graphs[template_key] = sub_graph.copy()

        print(f"Parsing complete. Found {len(temp_groups)} distinct join templates.")

        print("Global Predicate Map Summary:")
        for real_table, pred_map in self.global_predicate_map.items():
            print(f"  Table '{real_table}': {len(pred_map)} unique predicates.")

         # 构建最终的 join_templates 结构
        for template_key, query_list in temp_groups.items():
            aliases_tuple, join_sig = template_key

            import hashlib
            sig_hash = hashlib.md5(join_sig.encode('utf-8')).hexdigest()[:6]
            template_id = f"{'_'.join(aliases_tuple)}_{sig_hash}"
            
            sub_graph = temp_graphs[template_key]
            num_queries = len(query_list)

            real_names = {}
            for alias in aliases_tuple:
                real_names[alias] = sub_graph.nodes[alias]["real_name"]

            # {alias: {q_idx: pid}}
            table_pid_map = defaultdict(dict)

            for q_idx, pids_dict in enumerate(query_list):
                for alias, pid in pids_dict.items():
                    table_pid_map[alias][q_idx] = pid
            
            self.join_templates[template_id] = {
                "aliases": list(aliases_tuple),
                "real_names": real_names,
                "join_graph": sub_graph,
                "queries_count": num_queries,
                "table_pids": dict(table_pid_map)
            }

        # 将join_templates按照alias个数升序排列
        # self.join_templates = dict(sorted(self.join_templates.items(), key=lambda item: len(item[1]['aliases'])))


    def prepare_pid_to_qid_map(self, template_data):
        """
        为当前 Template 构建 PID -> QID Bitmap (Int) 的映射。
        用于 greedy_join_selection 中的 Python 端翻译。
        """

        # template_data['table_pids']: {alias: {qid: pid}}
        table_pids = template_data['table_pids']
        total_queries = template_data['queries_count']

        pid_map = defaultdict(dict)
        global_map = defaultdict(int)

        for alias, qid_pid_map in table_pids.items():
            temp_pid_groups = defaultdict(list)
            temp_globals = []

            for qid, pid in qid_pid_map.items():
                if pid == -1:
                    temp_globals.append(qid)
                else:
                    temp_pid_groups[pid].append(qid)
            
            g_mask = 0
            for q in temp_globals:
                g_mask |= (1 << q)
            global_map[alias] = g_mask

            for pid, qid_list in temp_pid_groups.items():
                mask = 0
                for q in qid_list:
                    mask |= (1 << q)
                pid_map[alias][pid] = mask

        return pid_map, global_map
    
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
        
        # for pid, char in enumerate(pid_bitmap_str):
        #     if char == '1':
        #         result_mask |= pid_map.get(pid, 0)

        pos = pid_bitmap_str.find('1')
        while pos != -1:
            result_mask |= pid_map.get(pos, 0)
            pos = pid_bitmap_str.find('1', pos + 1)
        
        return result_mask

    def compute_root_weights(self):
        pass


    def build_join_tree_structure(self, join_graph, aliases):
        aliases = sorted(aliases)
        scored = []
        for a in aliases:
            real_name = join_graph.nodes[a]['real_name']
            card = TABLE_CARD.get(real_name, float("inf"))
            scored.append((card, a))
        scored.sort()

        root_table = None

        for card, alias in scored:
            if card > 10000:
                root_table = alias
                break

        if root_table is None or root_table == 'ci' or root_table == 'mi1' or root_table == 'mi2':
            root_table = scored[0][1]

        visited = {root_table}

        join_execution_plan = [{
            'alias': root_table,
            'real_name': join_graph.nodes[root_table]['real_name'],
            'parent': None,
            'join_condition': None
        }]

        while len(visited) < len(aliases):
            candidates = []

            for u in sorted(visited):
                for v in sorted(join_graph.neighbors(u)):
                    if v not in visited:
                        real_name = join_graph.nodes[v]['real_name']
                        card = TABLE_CARD.get(real_name, float("inf"))
                        candidates.append((card, u, v))

            if not candidates:
                raise RuntimeError("Join graph is not connected")
            
            _, parent, child = min(candidates)
            visited.add(child)
            edge_data = join_graph.get_edge_data(parent, child)
            raw_condition = edge_data.get("join_condition")
            norm_cond = normalize_condition(raw_condition)

            if not norm_cond:
                raise ValueError(f"Missing join condition between {parent} and {child}")
            
            join_execution_plan.append({
                'alias': child,
                'real_name': join_graph.nodes[child]['real_name'],
                'parent': parent,
                'join_condition': norm_cond
            })
        
        if len(join_execution_plan) != len(aliases):
            print(f"WARNING: Subgraph might be disconnected. Expected {len(aliases)} nodes, got {len(join_execution_plan)}.")

        return join_execution_plan, root_table


    # 划分可以先改成对第一个表平均划分，随机分吧
    def partition_root_table(self, root_table, m_partitions, template_data):
        """
        将root划分为 m 个分区，返回每个分区的 id 列表。
        """
        real_name = template_data['real_names'][root_table]

        try:
            self.cursor.execute(f"SELECT id FROM {real_name} ORDER BY RANDOM();")
            all_ids = [row[0] for row in self.cursor.fetchall()]

            total_rows = len(all_ids)

            # 行数少于分区数时，允许重复选id
            if total_rows < m_partitions:
                partitions = []
                for i in range(m_partitions):
                    pid = all_ids[i % total_rows]
                    partitions.append([pid])

            if total_rows > m_partitions:
                base_size = total_rows // m_partitions
                remainder = total_rows % m_partitions

                partitions = []
                start_idx = 0
                for i in range(m_partitions):
                    part_size = base_size + (1 if i < remainder else 0)
                    end_idx = start_idx + part_size
                    partition_ids = all_ids[start_idx:end_idx]
                    partitions.append(partition_ids)
                    start_idx = end_idx

            return partitions

        except Exception as e:
            print(f"Error partitioning root table '{real_name}': {e}")
            raise e

    def create_annotation_tables(self):
        """
        [One-Time Setup]
        不修改原表，而是创建 '{table}_anno_idx' 伴生表存储 PID Bitmap。
        表结构: (id PRIMARY KEY, anno BIT VARYING)
        """
        print(f"    [Sidecar Setup] Creating annotation tables...")

        ROW_BATCH_SIZE = 100000
        UNIQUE_PRED_BATCH_SIZE = 500

        start_total = time.time()
        for real_name in self.all_involved_tables:
            if self.workload_name:
                sidecar_name = f"{real_name}_anno_idx_{self.workload_name}"
            else:
                sidecar_name = f"{real_name}_anno_idx"
            print(f"        Processing '{real_name}' -> '{sidecar_name}'...", flush=True)
            preds_dict = self.global_predicate_map.get(real_name, {})
            total_preds = len(preds_dict)
            if total_preds == 0:
                total_preds = 1  # 至少要有一位，避免 SQL 错误
            zero_string = '0' * total_preds

            try:
                self.cursor.execute(f"SELECT to_regclass('{sidecar_name}');")
                if self.cursor.fetchone()[0] is not None:
                    print(f"        Skipping '{real_name}': Sidecar '{sidecar_name}' already exists.", flush=True)
                    continue

                # self.cursor.execute(f"DROP TABLE IF EXISTS {sidecar_name}")
                # self.conn.commit()

                create_sql = f"""
                    CREATE TABLE {sidecar_name} AS
                    SELECT id AS query_anno_id, B'{zero_string}'::{f"BIT VARYING({total_preds})"} AS query_anno
                    FROM {real_name}
                """
                self.cursor.execute(create_sql)
                self.cursor.execute(f"ALTER TABLE {sidecar_name} ADD PRIMARY KEY (query_anno_id)")
                self.conn.commit()

                self.cursor.execute(f"SELECT min(id), max(id) FROM {real_name}")
                min_id, max_id = self.cursor.fetchone()
                if min_id is None: min_id = 0
                if max_id is None: max_id = 0
                
                total_row_batches = (max_id - min_id) // ROW_BATCH_SIZE + 1

                if not preds_dict:
                    print(f"            No predicates for table '{real_name}'. Created anno column with default zeros.")
                    continue

                sorted_items = sorted(preds_dict.items(), key=lambda x: x[1])
                sql_chunks = []
                for i in range(0, total_preds, UNIQUE_PRED_BATCH_SIZE):
                    batch_items = sorted_items[i : i + UNIQUE_PRED_BATCH_SIZE]
                    expr_parts = []
                    where_conditions = []

                    for pred_sql, pid in batch_items:
                        where_conditions.append(pred_sql)

                        mask_list = ['0'] * total_preds
                        mask_list[pid] = '1'
                        current_mask_str = "".join(mask_list)

                        expr = f"(CASE WHEN {pred_sql} THEN B'{current_mask_str}' ELSE B'{zero_string}' END)"
                        expr_parts.append(expr)
                    
                    pred_where = f"({' OR '.join(where_conditions)})" if where_conditions else ""
                    full_expr = " | ".join(expr_parts)
                    sql_chunks.append((full_expr, pred_where))

                # 外层循环：遍历 ID 范围
                current_id = min_id
                row_batch_count = 0

                while current_id <= max_id:
                    next_id = current_id + ROW_BATCH_SIZE
                    row_batch_count += 1
                    t0 = time.time()

                    for full_expr, pred_where in sql_chunks:
                        update_sql = f"""
                            UPDATE {sidecar_name} s
                            SET query_anno = query_anno | ({full_expr})
                            FROM {real_name} o
                            WHERE s.query_anno_id = o.id
                                AND s.query_anno_id >= {current_id} AND s.query_anno_id < {next_id}
                                {"AND " + pred_where if pred_where else ""}
                        """
                        self.cursor.execute(update_sql)
                    
                    self.conn.commit()

                    if row_batch_count % 10 == 1 or row_batch_count == total_row_batches:
                        print(f"            Processed row batch {row_batch_count}/{total_row_batches} (IDs [{current_id}, {next_id})). Time: {time.time() - t0:.2f}s")

                    current_id = next_id

                old_isolation_level = self.conn.isolation_level
                self.conn.set_isolation_level(0)  # 设置为 autocommit 模式
                try:
                    self.cursor.execute(f"VACUUM ANALYZE {sidecar_name};")
                finally:
                    self.conn.set_isolation_level(old_isolation_level)  # 恢复原始隔离级别

            except Exception as e:
                print(f"Error creating annotation table to table '{real_name}': {e}")
                self.conn.rollback()
                raise e
            
        print(f"    [Sidecar Setup] Finished. Total time: {time.time() - start_total:.2f}s", flush=True)

    def _parse_join_condition_for_engine(self, raw_cond, child_alias, parent_alias):
        """
        辅助函数：解析 SQL 连接条件，提取 (my_col, target_alias, target_col)。
        用于 Engine 的 get_candidates。
        """
        # raw_cond的结构例如 "t.id = mi.movie_id"
        parts = raw_cond.split('=')
        if len(parts) != 2: return []
        
        left = parts[0].strip().split('.')
        right = parts[1].strip().split('.')
        
        child_col = None
        target_col = None
        target_alias_res = None
        
        if left[0] == child_alias:
            child_col = left[1]
            if right[0] == parent_alias:
                target_col = right[1]
                target_alias_res = right[0]
        elif right[0] == child_alias:
            child_col = right[1]
            if left[0] == parent_alias:
                target_col = left[1]
                target_alias_res = left[0]
                
        if child_col and target_col:
            return [(child_col, target_alias_res, target_col)]
        return []

    def _build_lookahead_forest(self, join_execution_plan, start_index, k_steps):
        """
        基于 join_execution_plan 构建 Lookahead Tree。
        
        Args:
            join_execution_plan: 完整的执行计划列表
            start_index: 当前 Rj 在 plan 中的下标
            k_steps: 向后看多少步 (不包括 Rj 自己)
        
        Returns:
            list: [tree_root_structure] (通常只有一个元素，因为 Rj 是唯一的根)
        """
        # 截取 Rj 及其后面的 K 个表
        lookahead_slice = join_execution_plan[start_index : start_index + k_steps + 1]
        
        if not lookahead_slice:
            return []
            
        # 初始化 Rj 是根
        rj_node_data = lookahead_slice[0]
        rj_alias = rj_node_data['alias']
        
        # 这里的 conds 是 Rj 连向 T 的条件，得到(rj_col, parent_alias, parent_col)
        rj_conds = self._parse_join_condition_for_engine(
            rj_node_data['join_condition'], 
            rj_alias, 
            rj_node_data['parent']
        )

        # key: alias, value: node_structure (dict)
        tree_nodes = {}
        
        root_struct = {
            'alias': rj_alias,
            'conds': rj_conds,
            'children': []
        }
        tree_nodes[rj_alias] = root_struct
        
        # 线性遍历后续表，尝试挂载
        for step in lookahead_slice[1:]:
            child_alias = step['alias']
            parent_alias = step['parent']
            raw_cond = step['join_condition']
            
            # 只有当 Parent 已经在树中时，才挂载
            if parent_alias in tree_nodes:
                conds = self._parse_join_condition_for_engine(raw_cond, child_alias, parent_alias)
                
                child_struct = {
                    'alias': child_alias,
                    'conds': conds,
                    'children': []
                }
                
                tree_nodes[parent_alias]['children'].append(child_struct)
                
                tree_nodes[child_alias] = child_struct
            else:
                # Parent 不在树中 (说明直接连 T 的更早节点)，跳过 
                pass
                
        # 返回根节点
        return [root_struct]


# 辅助函数：将 Partition ID 转化为初始的 T (Beam)
    def _initialize_root_beam(self, root_alias, partition_ids, pid_map, global_map, uncovered_mask_int, limit_x):
        """
        从 Partition IDs 构建初始的 Beam (T)。
        完全在内存中进行，不查数据库。
        """
        candidates = []
        
        if root_alias not in self.engine.global_cache:
            print(f"Error: Root table {root_alias} not loaded in Engine.")
            return []
            
        root_cache = self.engine.global_cache[root_alias]['rows']
        
        for pid in partition_ids:
            pk_str = str(pid)
            if pk_str not in root_cache:
                continue
            
            row_data = root_cache[pk_str]
            current_bmp = self._translate_pid_bitmap(row_data['pid_bmp'], pid_map, global_map)

            score = bin(current_bmp & uncovered_mask_int).count('1')
            
            candidate = {
                'ids': {root_alias: pk_str},
                'bmp': current_bmp,
                'score': score # 仅用于排序
            }
            candidates.append(candidate)
            
        # 初始筛选 Top-K
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:limit_x]

    def greedy_join_selection(self, partition_ids, partition_idx, root_table, join_tree, template_data, global_covered_mask, limit_x):
        """
        Algorithm 4: Pure Memory Version
        完全移除 SQL Join，逻辑由 Engine 和 Python 接管。
        """
        join_graph = template_data['join_graph']
        total_queries = template_data['queries_count']
        
        # 计算 Uncovered Mask
        all_query_ids = set(range(total_queries))
        uncovered_ids = list(all_query_ids - global_covered_mask)
        uncovered_mask_int = 0
        for qid in uncovered_ids:
            uncovered_mask_int |= (1 << qid)

        # Root Beam (Step 0)
        current_beam = self._initialize_root_beam(
            root_table, 
            partition_ids, 
            template_data['pid_map'].get(root_table, {}),
            template_data['global_map'].get(root_table, 0),
            uncovered_mask_int, 
            limit_x
        )
        
        if not current_beam:
            return None

        # Step 1 to N
        # join_tree[0] 是 root, 循环从 1 开始
        for step_idx, step in enumerate(join_tree[1:]):
            next_alias = step['alias']
            
            # 构建 Lookahead Tree
            current_plan_idx = step_idx + 1
            
            lookahead_tree_list = self._build_lookahead_forest(
                join_tree, 
                current_plan_idx, 
                self.lookahead_k
            )
            
            lookahead_tree = lookahead_tree_list[0] if lookahead_tree_list else None
            
            if not lookahead_tree:
                # 不应该发生
                print(f"Warning: Lookahead tree empty at step {step_idx} for alias {next_alias}. Using single node.")
                lookahead_tree = {
                    'alias': next_alias, 
                    'conds': self._parse_join_condition_for_engine(
                        step['join_condition'], next_alias, step['parent']
                    ),
                    'children': []
                }

            # 扩展 Beam 中的每一个元组
            next_candidates_heap = [] # Min-heap for Top-K

            self.engine.memo.clear() # 清空权重缓存
            
            for t_tuple in current_beam:
                current_ids = t_tuple['ids'] # dict {alias: pk}
                current_base_bmp = t_tuple['bmp']

                extensions, rj_final_bmps = self.engine.sample_extensions(
                    current_ids, 
                    lookahead_tree, 
                    k_samples=self.lookahead_samples,
                    pid_map=template_data['pid_map'],
                    global_map=template_data['global_map']
                )
                
                if not extensions:
                    continue

                # 处理扩展结果
                for rj_id, lookahead_bmp in extensions.items():
                    rj_real_bmp = rj_final_bmps.get(rj_id, 0)

                    new_real_bmp = current_base_bmp & rj_real_bmp

                    # CHECK: 打分逻辑还要再确认
                    final_potential_bmp = current_base_bmp & lookahead_bmp
                    
                    score = bin(final_potential_bmp & uncovered_mask_int).count('1')
                    
                    # 构建新元组
                    new_ids = current_ids.copy()
                    new_ids[next_alias] = rj_id
                    
                    new_candidate = (score, next(self.tie_breaker), new_ids, new_real_bmp)

                    # print(f"        Candidate Extension: ID={rj_id}, Score={score}")
                    
                    if len(next_candidates_heap) < limit_x:
                        heapq.heappush(next_candidates_heap, new_candidate)
                    else:
                        if score > next_candidates_heap[0][0]:
                            heapq.heapreplace(next_candidates_heap, new_candidate)
            
            sorted_candidates = sorted(next_candidates_heap, key=lambda x: x[0], reverse=True)
            
            if not sorted_candidates:
                return None # 死路
            
            current_beam = []
            for score, counter, ids, real_bmp in sorted_candidates:
                current_beam.append({
                    'ids': ids,
                    'bmp': real_bmp,
                    'score': score
                })
                
        best_tuple = current_beam[0]
        
        covered_indices = set()
        temp_mask = best_tuple['bmp']
        qid = 0
        while temp_mask > 0:
            if temp_mask & 1: covered_indices.add(qid)
            temp_mask >>= 1
            qid += 1
            
        return {
            'full_row': best_tuple['ids'], # ids 是 {alias: pk}
            'covered_indices': covered_indices
        }

    def sample_for_one_template(self, partitions, root_table, join_order_tree, template_data):
        join_graph = template_data['join_graph']
        for node, info in join_graph.nodes(data=True):
            sels = []
            edges = join_graph.edges(node)
            for edge in edges:
                # edge_data = join_graph.get_edge_data(edge[0], edge[1])
                edge_data = join_graph[edge[0]][edge[1]]
                if "!" in edge_data["join_condition"]:
                    jconds = edge_data["join_condition"].split("!=")
                else:
                    jconds = edge_data["join_condition"].split("=")
                for jc in jconds:
                    jc = jc.strip()
                    if node == jc[0:len(node)]:
                        if jc not in sels:
                            sels.append(jc)
                    jc_node = jc.split(".")[0]
                    join_graph[edge[0]][edge[1]][jc_node] = jc
            
            # 如果没有主键就加上主键
            if f"{node}.id" not in sels and f"{node}.Id" not in sels:
                sels.append(f"{node}.id")
            
            join_graph.nodes()[node]["sels"] = sels

        template_data['join_graph'] = join_graph
        generated_samples = []
        global_covered_queries = set()
        total_queries = template_data['queries_count']
        
        pid_map, global_map = self.prepare_pid_to_qid_map(template_data)
        template_data['pid_map'] = pid_map
        template_data['global_map'] = global_map

        # === Preload Data into Engine ===
        engine_tables_info = []
        # 加载所有涉及的表，因为 Engine 现在负责所有的 Join
        for node, info in join_graph.nodes(data=True):
            raw_sels = info["sels"]
            clean_cols = set()
            for s in raw_sels:
                col = s.split('.')[-1]
                clean_cols.add(col)

            if "id" in clean_cols:
                clean_cols.remove("id")
                final_cols_list = ["id"] + sorted(list(clean_cols))
            elif "Id" in clean_cols:
                clean_cols.remove("Id")
                final_cols_list = ["Id"] + sorted(list(clean_cols))
            else:
                final_cols_list = ["id"] + sorted(list(clean_cols))

            engine_tables_info.append({
                'alias': node,
                'real_name': template_data['real_names'][node],
                'join_keys': final_cols_list
            })
        
        # Preload 自动处理 Translation 和 Indexing
        preload_start_time = time.time()
        print(f"    --> Preloading data into Engine...")
        self.engine.preload_data(engine_tables_info, template_data, self.workload_name)
        print(f"        Preloading completed in {time.time() - preload_start_time:.2f}s.")

        for k in range(self.k_bitmaps):
            bitmap_start_time = time.time()
            print(f"    --> Constructing Bitmap {k+1}/{self.k_bitmaps}...")
            current_bitmap = []

            if len(global_covered_queries) == total_queries:
                print("        All queries already covered. Ending early.")
                break
            
            for p_idx, partition_ids in enumerate(partitions):
                partition_select_time_start = time.time()
                best_tuple_info = self.greedy_join_selection(
                    partition_ids, 
                    p_idx, 
                    root_table, 
                    join_order_tree, 
                    template_data, 
                    global_covered_queries, 
                    self.limit_x
                )

                if best_tuple_info:
                    current_bitmap.append(best_tuple_info['full_row'])
                    newly_covered = best_tuple_info['covered_indices']
                    incremental_gain = len(newly_covered - global_covered_queries)
                    global_covered_queries.update(newly_covered)
                    print(f"          Partition {p_idx+1}/{len(partitions)}: Selected tuple covers {incremental_gain} new queries. Time: {time.time() - partition_select_time_start:.2f}s")
                else:
                    print(f"          Partition {p_idx+1}/{len(partitions)}: No valid tuple found. Time: {time.time() - partition_select_time_start:.2f}s")

                self.conn.commit()
            
            generated_samples.append(current_bitmap)
            coverage_pct = (len(global_covered_queries) / total_queries * 100) if total_queries > 0 else 0
            print(f"        Bitmap {k+1} constructed. Size: {len(current_bitmap)}. Global Coverage: {coverage_pct:.2f}%. Total Time: {time.time() - bitmap_start_time:.2f}s")

            if coverage_pct > 95.0:
                print("        Coverage threshold reached (>95%). Ending bitmap construction early.")
                break
        
        return generated_samples

    def sample(self):
        print("Starting Join Sampling Process...")
        start_time = time.time()
        self.load_and_parse_workload() 
        print(f"Total Join Templates Loaded: {len(self.join_templates)}, Workload Parsing Time: {time.time() - start_time:.2f}s")
        self.create_annotation_tables()
        if not self.join_templates:
            print("No join templates found. Exiting.")
            return
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

        SAVE_BATCH_SIZE = 10
        current_batch_samples = {}
        processed_count = 0
        batch_index = 0

        for template_id, template_data in self.join_templates.items():
            if len(template_data['aliases']) < 2:
                # print(f"\n=== Skipping Template: {template_id} (only {len(template_data['aliases'])} table) ===")
                continue

            if "mi1" in template_data['aliases'] or "mi2" in template_data['aliases'] or "ci" in template_data['aliases']:
                # print(f"\n=== Skipping Template: {template_id} (contains problematic table) ===")
                continue

            # if len(template_data['aliases']) < 3 or len(template_data['aliases']) >= 5:
            #     continue

            print(f"\n=== Processing Template: {template_id} ===")
            print(f"    Tables: {template_data['aliases']}")
            print(f"    Total Queries in Group: {template_data['queries_count']}")

            join_graph = template_data['join_graph']
            aliases = template_data['aliases']

            try:
                join_order_tree, root_table = self.build_join_tree_structure(join_graph, aliases)
                print(f"    Join Root: {root_table}")
                partitions = self.partition_root_table(root_table=root_table, 
                                                        m_partitions=self.m_partitions, 
                                                        template_data=template_data)
                print(f"    Root table partitioned into {len(partitions)} segments.")
                time_sampling_start = time.time()

                template_samples = self.sample_for_one_template(
                        partitions=partitions,
                        root_table=root_table,
                        join_order_tree=join_order_tree,
                        template_data=template_data
                    )
                
                current_batch_samples[template_id] = template_samples
                processed_count += 1
                print(f"    Sampling completed in {time.time() - time_sampling_start:.2f}s.")

                if processed_count % SAVE_BATCH_SIZE == 0:
                    batch_filename = f"samples_batch_{batch_index}.json"
                    batch_full_path = os.path.join(self.output_path, batch_filename)
                    self.save_samples(current_batch_samples, batch_full_path)
                    print(f"    >>> Saved batch {batch_index} (Templates {processed_count - SAVE_BATCH_SIZE + 1} to {processed_count}), Used Time: {time.time() - start_time:.2f}s")
                    batch_index += 1
                    current_batch_samples = {}

            except Exception as e:
                print(f"ERROR processing template {template_id}: {e}")
                import traceback
                traceback.print_exc()
        
        if current_batch_samples:
            batch_filename = f"samples_batch_{batch_index}.json"
            batch_full_path = os.path.join(self.output_path, batch_filename)
            self.save_samples(current_batch_samples, batch_full_path)
            print(f"    >>> Saved final batch {batch_index} (Templates {processed_count - len(current_batch_samples) + 1} to {processed_count})...")
        print(f"\nAll tasks finished. Results saved to directory: {self.output_path}. Total Time: {time.time() - start_time:.2f}s")

    def save_samples(self, samples_dict, output_path):
        formatted_output = {}
        for template_id, k_bitmaps in samples_dict.items():
            formatted_bitmaps = []
            for bitmap in k_bitmaps:
                if not bitmap:
                    formatted_bitmaps.append([])
                    continue
                aliases = sorted(bitmap[0].keys())
                header = [f"{alias}.id" for alias in aliases]
                formatted_rows = [header]
                for row_dict in bitmap:
                    row = [row_dict[alias] for alias in aliases]
                    formatted_rows.append(row)
                formatted_bitmaps.append(formatted_rows)
            formatted_output[template_id] = formatted_bitmaps

        def default_serializer(obj):
            import decimal
            import datetime
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return str(obj)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(formatted_output, f, default=default_serializer, indent=2)
        except Exception as e:
            print(f"Error saving samples to {output_path}: {e}")
            raise e
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python join_sampler.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]

    sampler = JoinSampler(config_path)
    try:
        sampler.sample()
    finally:
        sampler.close()


# 算法4 采完元组之后按照Rj分组，选每个组内的最大值当作分数
# 把采样算法改成wander join，还是用sql试一下，不要load到内存里
# 还是遍历T吧，尤其第一个表要遍历T