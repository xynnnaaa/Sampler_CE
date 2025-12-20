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

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def normalize_condition(cond_str):
    """
    标准化连接条件字符串，确保 'a=b' 和 'b=a' 被视为相同。
    去掉空格并按字典序排列等号两边。
    """
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

class TrieNode:
    def __init__(self, token):
        self.token = token # (parent, child, condition)
        self.children = {} # Dict[StepToken, TrieNode]

        # Template Info
        self.is_template_end = False
        self.pending_instances = [] # 暂存
        self.end_template_keys = [] 

        # QID Info
        self.relevant_qids = set()
        self.qid_map = {}   # {qid: {alias: pid}}
        self.subtree_total_queries = 0 # 只有 Root 节点这个值有意义

        # Graph Info
        self.child_alias = None
        self.real_name = None
        self.join_condition = None
        self.parent_alias = None
        self.sels = []

        self.parent_temp_table_ref = None

class JoinPathTrie:
    def __init__(self):
        self.root = TrieNode("ROOT_SENTRY")

    def insert(self, template_key, path_signature, join_graph, instances):
        node = self.root
        for step_token in path_signature:
            if step_token not in node.children:
                new_node = TrieNode(step_token)

                # 填充 Graph Info 部分

                parent, child, cond = step_token
                new_node.parent_alias = parent
                new_node.child_alias = child
                new_node.join_condition = cond

                if child in join_graph.nodes:
                    new_node.real_name = join_graph.nodes[child]['real_name']

                    new_node.sels = join_graph.nodes[child].get('sels', [])

                node.children[step_token] = new_node
            
            node = node.children[step_token]

        # 填充 Template Info 部分
        node.is_template_end = True
        node.pending_instances.extend(instances)
        node.end_template_keys.append(template_key)


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

        print("Configuration Loaded:")
        print(f"  Base Query Directory: {self.base_query_dir}")
        print(f"  Workload Name: {self.workload_name}")
        print(f"  Output Path: {self.output_path}")
        print(f"  Skip Template '7a': {self.skip_7a}")
        print(f"  M Partitions: {self.m_partitions}")
        print(f"  K Bitmaps: {self.k_bitmaps}")
        print(f"  Limit X: {self.limit_x}")

        # 初始化 Trie
        self.trie = JoinPathTrie()

        # 用于缓存每个 Template 的 Join Graph
        self.template_graphs = {} 

        self.join_templates = {}
        self.global_predicate_map = defaultdict(lambda: {}) # {real_table: {pred_sql: pid}}
        self.global_pid_counters = defaultdict(int)         # {real_table: next_pid}

        db_conf = self.config.get("database", {})
        try:
            self.conn = psycopg2.connect(
                host=db_conf.get("host", "localhost"),
                port=db_conf.get("port", 5432),
                dbname=db_conf.get("dbname", "imdb"),
                user=db_conf.get("user", "your_username")
            )
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
        
    def close(self):
        print("Closing database connection...")
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        print("Resources released.")


    def get_deterministic_execution_plan(self, join_graph, aliases):
        def root_score(alias):
            real_name = join_graph.nodes[alias]['real_name']
            return (TABLE_CARD.get(real_name, float("inf")), alias)

        root_table = max(aliases, key=root_score)

        visited = {root_table}

        path_signature = [("ROOT_SENTRY", root_table, "None")]

        while len(visited) < len(aliases):
            candidates = []

            for u in visited:
                for v in join_graph.neighbors(u):
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

            path_signature.append((parent, child, norm_cond))

            if not raw_condition:
                raise ValueError(f"Missing join condition between {parent} and {child}")
        
        if len(path_signature) != len(aliases):
            print(f"WARNING: Subgraph might be disconnected. Expected {len(aliases)} nodes, got {len(path_signature)}.")

        return path_signature
    

    def load_and_parse_workload(self):
        self.all_involved_tables = set()

        self.temp_template_data = defaultdict(lambda: {'instances': [], 'graph': None})

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


                    if template_key not in self.temp_template_data:
                        self.temp_template_data[template_key]['graph'] = sub_graph.copy()
                    self.temp_template_data[template_key]['instances'].append(current_query_pids)

        print(f"Parsing complete. Found {len(self.temp_template_data)} distinct join templates.")

        print("Global Predicate Map Summary:")
        for real_table, pred_map in self.global_predicate_map.items():
            print(f"  Table '{real_table}': {len(pred_map)} unique predicates.")


    def build_global_trie(self):
        print("\nBuilding Trie from Join Templates...")
        self.trie = JoinPathTrie()

        for template_key, data in self.temp_template_data.items():
            aliases_tuple = template_key[0]
            join_graph = data['graph']
            instances = data['instances'] # List[Dict{alias: pid}]

            try:
                path_signature = self.get_deterministic_execution_plan(join_graph, list(aliases_tuple))
                self.trie.insert(template_key, path_signature, join_graph, instances)

            except Exception as e:
                print(f"Error building trie for {aliases_tuple}: {e}")
                continue

        print("Assigning QIDs per Root Subtree...", flush=True)

        for root_token, root_node in self.trie.root.children.items():
            current_qid_counter = 0

            def assign_and_aggregate(node):
                nonlocal current_qid_counter

                my_relevant = set()

                if node.is_template_end:
                    for inst in node.pending_instances:
                        qid = current_qid_counter
                        current_qid_counter += 1

                        node.qid_map[qid] = inst
                        my_relevant.add(qid)
                    node.pending_instances = [] 

                for child in node.children.values():
                    child_relevant = assign_and_aggregate(child)
                    my_relevant.update(child_relevant)

                node.relevant_qids = my_relevant
                return my_relevant
            
            assign_and_aggregate(root_node)

            root_node.subtree_total_queries = current_qid_counter

            print(f"  Root '{root_node.child_alias}' ({root_node.real_name}): {current_qid_counter} queries.", flush=True)


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

    def partition_root_table(self, real_name, m_partitions):
        """
        将root划分为 m 个分区，返回每个分区的 id 列表。
        """
        try:
            self.cursor.execute(f"SELECT id FROM {real_name} ORDER BY RANDOM();")
            all_ids = [row[0] for row in self.cursor.fetchall()]

            total_rows = len(all_ids)

            # 行数少于分区数时，允许重复选id
            if total_rows < m_partitions:
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
        
    def prepare_subtree_pid_map(self, root_node):
        """
        为整个 Root 子树构建统一的 PID -> QID 映射。
        """
        # 收集所有 QID -> Instance
        subtree_instances = {} # { qid: {alias: pid} }
        
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.is_template_end:
                subtree_instances.update(node.qid_map)
            for child in node.children.values():
                stack.append(child)
        
        # 构建 Map
        pid_map = defaultdict(dict)
        global_map = defaultdict(int)
        
        for qid, inst in subtree_instances.items():
            for alias, pid in inst.items():
                if pid == -1:
                    global_map[alias] |= (1 << qid)
                else:
                    if pid not in pid_map[alias]:
                        pid_map[alias][pid] = 0
                    pid_map[alias][pid] |= (1 << qid)
                    
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

        pos = pid_bitmap_str.find('1')
        while pos != -1:
            result_mask |= pid_map.get(pos, 0)
            pos = pid_bitmap_str.find('1', pos + 1)
        
        return result_mask
    
    def sample_subtree_recursive(self, current_node, partition_ids, global_covered, pid_map, global_map):
        """
        DFS 遍历 Trie 进行采样。
        利用 TrieNode.current_temp_table 传递中间结果。
        """

        tmp_table_name = f"temp_{id(current_node)}_{int(time.time()*100000)}"

        parent_temp_table = getattr(current_node, 'parent_temp_table_ref', None)


    def sample_trie_root(self, root_node):
        """
        处理以 root_node 为起点的整个 Join 树。
        """

        # 对root进行分区
        real_name = root_node.real_name
        partitions = self.partition_root_table_simple(real_name, self.m_partitions)

        # 构建子树级 pid--qid 映射
        pid_map, global_map = self.prepare_subtree_pid_map(root_node)

        print(f"    Root partitioned into {len(partitions)} segments.", flush=True)

        subtree_covered = set()
        total_qids = root_node.subtree_total_queries

        for k in range(self.k_bitmaps):
            print(f"    --> Bitmap {k+1}/{self.k_bitmaps}...", flush=True)

            for p_idx, partition_ids in enumerate(partitions):
                root_temp_table = f"temp_root_{id(root_node)}_{p_idx}"

                success = self.execute_root_step(
                    node=root_node,
                    partition_ids=partition_ids,
                    output_table=root_temp_table,
                    pid_map=pid_map,
                    global_map=global_map,
                    covered_mask=subtree_covered,
                    total_queries=total_qids
                )

                if success:
                    try:
                        if root_node.is_template_end:
                            self.collect_samples_for_node(root_node, root_temp_table, subtree_covered)

                        # DFS
                        for child in root_node.children.values():
                            self.dfs_join_recursive(
                                current_node=child,
                                parent_table=root_temp_table,
                                pid_map=pid_map,
                                global_map=global_map,
                                covered_mask=subtree_covered,
                                total_queries=total_qids
                            )
                    finally:
                        self.cursor.execute(f"DROP TABLE IF EXISTS {root_temp_table}")

                self.conn.commit()

    def dfs_join_recursive(self, current_node, parent_table, pid_map, global_map, covered_mask, total_queries):
        """
        DFS 递归 Join。
        """
        current_temp_table = f"temp_{id(current_node)}_{int(time.time()*1000)}"
        
        # --- Join Sampling---
        success = self.execute_join_step(
            node=current_node,
            input_table=parent_table,
            output_table=current_temp_table,
            pid_map=pid_map,
            global_map=global_map,
            covered_mask=covered_mask,
            total_queries=total_queries
        )
        
        if not success:
            return

        try:
            # 收集样本
            if current_node.is_template_end:
                self.collect_samples_for_node(current_node, current_temp_table, covered_mask)
            
            # 递归子节点
            for child in current_node.children.values():
                self.dfs_join_recursive(
                    current_node=child,
                    parent_table=current_temp_table,
                    pid_map=pid_map,
                    global_map=global_map,
                    covered_mask=covered_mask,
                    total_queries=total_queries
                )
                
        finally:
            self.cursor.execute(f"DROP TABLE IF EXISTS {current_temp_table}")

    def execute_root_step(self, node, partition_ids, output_table, pid_map, global_map, covered_mask, total_queries):
        """
        [Root Processing]
        1. Fetch: 从 Root Sidecar 表读取 (id, pid_bitmap)。
        2. Compute: 在 Python 中翻译为 QID Bitmap，并根据 relevant_mask 打分。
        3. Store: 将 Top-K 写入 output_table。
        """
        try:
            # Uncovered Mask = (Relevant - Covered)
            target_qids = node.relevant_qids - covered_mask
            target_mask_int = 0
            for qid in target_qids:
                target_mask_int |= (1 << qid)
                
            real_name = node.real_name
            alias = node.child_alias
            sidecar_name = f"{real_name}_anno_idx"
            if self.workload_name:
                sidecar_name += f"_{self.workload_name}"
            
            ids_values = ",".join(f"({uid})" for uid in partition_ids)

            current_id_cols = []

            root_sels = node.sels
            root_cols = []
            id_idx = -1
            for i, sel in enumerate(root_sels):
                # 在视图中的列重命名为 Alias_Column 形式
                col_pure = sel.split('.')[-1] 
                if col_pure == "id" or col_pure == "Id":
                    id_idx = i
                root_cols.append(f"{col_pure} AS {alias}_{col_pure}")
                current_id_cols.append(f"{alias}_{col_pure}")

            if id_idx == -1:
                print(f"        Warning: can't find id column for table {alias}.")
                return None

            root_select_str = ", ".join(root_cols)
            
            r_map = pid_map.get(alias, {})
            r_global = global_map.get(alias, 0)

            # --- 2. Fetch---
            fetch_sql = f"""
                WITH partition_filter(pid) AS ( VALUES {ids_values} )
                SELECT t.query_anno_id, t.query_anno::text
                FROM {sidecar_name} t
                JOIN partition_filter pf ON t.query_anno_id = pf.pid
            """
            self.cursor.execute(fetch_sql)
            rows = self.cursor.fetchall()
            
            # --- 3. Python Compute ---
            top_k_heap = []
            for row in rows:
                pk_id = row[0]
                pid_str = row[1]
                qid_mask = self._translate_pid_bitmap(pid_str, r_map, r_global)
                score = bin(qid_mask & target_mask_int).count('1')
                item = (score, pk_id, qid_mask)
                if len(top_k_heap) < self.limit_x:
                    heapq.heappush(top_k_heap, item)
                else:
                    if score > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, item)
            
            # --- 4. Sort & Prune ---
            top_k = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
            
            if not top_k:
                return False
            
            top_ids = [item[1] for item in top_k]
            top_ids_str = ",".join(map(str, top_ids))

            id_to_mask = {item[1]: item[2] for item in top_k}

            # --- 5. Store (Create output_table) ---
            refetch_sql = f"""
                SELECT {root_select_str}
                FROM {real_name} AS {alias}
                WHERE {alias}.id IN ({top_ids_str})
            """
            self.cursor.execute(refetch_sql)
            rows = self.cursor.fetchall()

            dummy_anno = f"B'{'0'*total_queries}'::bit varying({total_queries})"

            create_sql = f"""
                CREATE TEMP TABLE {output_table} AS
                SELECT {root_select_str}, {dummy_anno} AS anno
                FROM {real_name} AS {alias}
                LIMIT 0
            """

            self.cursor.execute(create_sql)

            insert_values = []
            for row in rows:
                id_val = row[id_idx]
                qid_str = format(id_to_mask[id_val], f'0{total_queries}b')
                insert_values.append(tuple(list(row) + [qid_str]))

            insert_sql = f"INSERT INTO {output_table} VALUES %s"
            execute_values(self.cursor, insert_sql, insert_values)

            return True

        except Exception as e:
            print(f"Error in execute_root_step for {node.child_alias}: {e}")
            self.conn.rollback()
            return False


    def sample(self):
        """
        [主入口]
        执行完整的 Join 采样流程，并保存结果。
        """

        print("Starting Join Sampling Process...")
        start_time = time.time()

        self.load_and_parse_workload() 
        self.create_annotation_tables()
        self.build_global_trie()    # 构建 Trie 并分配 QID

        print(f"Total Workload Analysis Time: {time.time() - start_time:.2f}s", flush=True)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

        # 遍历 Trie 的每个 Root 子树
        for root_token, root_node in self.trie.root.children.items():
            print(f"\n=== Processing Root Subtree: {root_node.child_alias} ({root_node.real_name}) ===", flush=True)
            self.sample_trie_root(root_node)
