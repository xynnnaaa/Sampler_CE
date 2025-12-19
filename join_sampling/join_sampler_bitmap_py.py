# anno列存谓词id
# 从小表开始，因为只会对第一个表扫描一遍做映射，单表可以直接用单表采样的方法
# 在内存中做映射，把M的内容select出来到内存里，在内存做映射和排序，然后直接把排好的作为T插回去

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
        self.join_templates = dict(sorted(self.join_templates.items(), key=lambda item: len(item[1]['aliases'])))


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


    # TODO: root不要选基数太大的表
    def build_join_tree_structure(self, join_graph, aliases):
        def root_score(alias):
            real_name = join_graph.nodes[alias]['real_name']
            return (-TABLE_CARD.get(real_name, float("inf")), alias)

        root_table = max(aliases, key=root_score)

        visited = {root_table}

        join_execution_plan = [{
            'alias': root_table,
            'real_name': join_graph.nodes[root_table]['real_name'],
            'parent': None,
            'join_condition': None
        }]

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

            if not raw_condition:
                raise ValueError(f"Missing join condition between {parent} and {child}")
            
            join_execution_plan.append({
                'alias': child,
                'real_name': join_graph.nodes[child]['real_name'],
                'parent': parent,
                'join_condition': raw_condition
            })

            # neighbors = list(join_graph.neighbors(current_node))
            # neighbors.sort()

            # for neighbor in neighbors:
            #     if neighbor not in visited:
            #         visited.add(neighbor)
            #         queue.append(neighbor)

            #         # check
            #         edge_data = join_graph.get_edge_data(current_node, neighbor)
            #         raw_condition = edge_data.get("join_condition")

            #         if not raw_condition:
            #             raise ValueError(f"Missing join condition between {current_node} and {neighbor}")
                    
            #         join_execution_plan.append({
            #             'alias': neighbor,
            #             'real_name': join_graph.nodes[neighbor]['real_name'],
            #             'parent': current_node,
            #             'join_condition': raw_condition
            #         })
        
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


    def greedy_join_selection(self, partition_ids, partition_idx, root_cache, root_table, join_tree, template_data, global_covered_mask, limit_x):
        """
        Algorithm 4
        """
        join_graph = template_data['join_graph']
        total_queries = template_data['queries_count']

        pid_map = template_data.get('pid_map', {})
        global_map = template_data.get('global_map', {})

        all_query_ids = set(range(total_queries))
        uncovered_ids = list(all_query_ids - global_covered_mask)
        uncovered_mask_int = 0
        for qid in uncovered_ids:
            uncovered_mask_int |= (1 << qid)

        try:
            root_real_name = template_data['real_names'][root_table]
            if self.workload_name:
                temp_T_name = "temp_T_" + self.workload_name
                temp_T_new_name = "temp_T_new_" + self.workload_name
                root_sidecar = f"{root_real_name}_anno_idx_{self.workload_name}"
            else:
                temp_T_name = "temp_T"
                temp_T_new_name = "temp_T_new"
                root_sidecar = f"{root_real_name}_anno_idx"
            self.cursor.execute(f"DROP TABLE IF EXISTS {temp_T_name};")

            if not partition_ids: return None

            current_id_cols = []

            root_sels = join_graph.nodes[root_table]['sels']

            root_cols = []
            id_idx = -1
            for i, sel in enumerate(root_sels):
                # 在视图中的列重命名为 Alias_Column 形式
                col_pure = sel.split('.')[-1] 
                if col_pure == "id" or col_pure == "Id":
                    id_idx = i
                root_cols.append(f"{col_pure} AS {root_table}_{col_pure}")
                current_id_cols.append(f"{root_table}_{col_pure}")

            if id_idx == -1:
                print(f"        Warning: can't find id column for table {root_table}.")
                return None

            root_select_str = ", ".join(root_cols)

            cached_pairs = None
            if root_cache is not None and partition_idx in root_cache:
                print(f"            Load id->qid map from cache.")
                cached_pairs = root_cache[partition_idx]
            else:
                print(f"            Compute id->qid map.")

                self.cursor.execute("TRUNCATE temp_partition_filter")
                buf = io.StringIO()
                for pid in partition_ids:
                    buf.write(f"{pid}\n")
                buf.seek(0)
                self.cursor.copy_from(
                    buf,
                    "temp_partition_filter",
                    columns=("pid",),
                )
                # ids_values = ",".join(f"({uid})" for uid in partition_ids)
                # fetch_sql = f"""
                #     WITH partition_filter(pid) AS ( VALUES {ids_values} )
                #     SELECT t.query_anno_id, t.query_anno::text
                #     FROM {root_sidecar} t
                #     JOIN partition_filter pf ON t.query_anno_id = pf.pid
                # """
                fetch_sql = f"""
                    SELECT t.query_anno_id, t.query_anno::text
                    FROM {root_sidecar} t
                    JOIN temp_partition_filter pf
                    ON t.query_anno_id = pf.pid;
                """
                t1_execute = time.time()
                self.cursor.execute(fetch_sql)
                print(f"                Execute root table select query in {time.time() - t1_execute:.2f}s.")
                t1_fetchall = time.time()
                rows = self.cursor.fetchall()
                print(f"                Fetched {len(rows)} rows from root table '{root_real_name}' in {time.time() - t1_fetchall:.2f}s.")

                cached_pairs = []
                r_map = pid_map.get(root_table, {})
                r_global = global_map.get(root_table, 0)

                t2 = time.time()
                for row in rows:
                    pk_id = row[0]
                    anno_str = row[-1]
                    qid_mask = self._translate_pid_bitmap(anno_str, r_map, r_global)
                    cached_pairs.append((pk_id, qid_mask))
                print(f"                Translated PID to QID in {time.time() - t2:.2f}s.")

                if root_cache is not None:
                    root_cache[partition_idx] = cached_pairs

            scored_candidates = []
            for pk_id, qid_mask in cached_pairs:
                score = bin(qid_mask & uncovered_mask_int).count('1')
                scored_candidates.append((score, pk_id, qid_mask))
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_k = scored_candidates[:limit_x]

            if not top_k:
                return None
            
            top_ids = [item[1] for item in top_k]
            top_ids_str = ",".join(map(str, top_ids))

            id_to_mask = {item[1]: item[2] for item in top_k}

            refetch_sql = f"""
                SELECT {root_select_str}
                FROM {root_real_name} AS {root_table}
                WHERE {root_table}.id IN ({top_ids_str})
            """
            t_refetch = time.time()
            self.cursor.execute(refetch_sql)
            rows = self.cursor.fetchall()
            print(f"            Refetched {len(rows)} rows from root table '{root_real_name}' in {time.time() - t_refetch:.2f}s.")
            
            # check
            dummy_anno = f"B'{'0'*total_queries}'::bit varying({total_queries})"

            create_sql = f"""
                CREATE TEMP TABLE {temp_T_name} AS
                SELECT {root_select_str}, {dummy_anno} AS anno
                FROM {root_real_name} AS {root_table}
                LIMIT 0
            """
            t3 = time.time()
            self.cursor.execute(create_sql)
            print(f"            Created temp table '{temp_T_name}' in {time.time() - t3:.2f}s.")

            insert_values = []

            for row in rows:
                id_val = row[id_idx]
                qid_str = format(id_to_mask[id_val], f'0{total_queries}b')
                insert_values.append(tuple(list(row) + [qid_str]))

            insert_sql = f"INSERT INTO {temp_T_name} VALUES %s"
            t4 = time.time()
            execute_values(self.cursor, insert_sql, insert_values)
            print(f"            Inserted top-{len(top_k)} candidates into '{temp_T_name}' in {time.time() - t4:.2f}s.")

            for step in join_tree[1:]:
                next_alias = step['alias']
                real_name = step['real_name']
                if self.workload_name:
                    next_sidecar = f"{real_name}_anno_idx_{self.workload_name}"
                else:
                    next_sidecar = f"{real_name}_anno_idx"

                parent_alias = step['parent']
                raw_join_cond = step['join_condition'] # e.g., "t.id = mi.movie_id"

                prev_cols_sql = [f"T_tmp.{col}" for col in current_id_cols]
                prev_select_str = ", ".join(prev_cols_sql)

                next_sels = join_graph.nodes[next_alias]['sels']
                next_cols_sql = []
                for sel in next_sels:
                    col_pure = sel.split('.')[-1]
                    next_cols_sql.append(f"{next_alias}.{col_pure} AS {next_alias}_{col_pure}")
                    current_id_cols.append(f"{next_alias}_{col_pure}")

                next_select_str = ", ".join(next_cols_sql)

                pattern = fr"\b{parent_alias}\."
                replacement = f"T_tmp.{parent_alias}_"
                fixed_join_cond = re.sub(pattern, replacement, raw_join_cond)

                fetch_join_sql = f"""
                    SELECT
                        {prev_select_str},
                        {next_select_str},
                        sa.query_anno::text,
                        T_tmp.anno::text
                    FROM {temp_T_name} T_tmp
                    JOIN {real_name} AS {next_alias} ON {fixed_join_cond}
                    JOIN {next_sidecar} sa ON {next_alias}.id = sa.query_anno_id;
                """
                t5_execute = time.time()
                self.cursor.execute("SET enable_hashjoin = OFF;")
                self.cursor.execute("SET enable_mergejoin = OFF;")
                self.cursor.execute(fetch_join_sql)
                print(f"                Execute {next_alias} join select query in {time.time() - t5_execute:.2f}s.")
                t5_fetchall = time.time()
                rows = self.cursor.fetchall()
                print(f"                Fetched joined rows with '{next_alias}' in {time.time() - t5_fetchall:.2f}s. Total rows: {len(rows)}")

                self.cursor.execute("SET enable_hashjoin = ON;")
                self.cursor.execute("SET enable_mergejoin = ON;")

                scored_candidates = []
                n_map = pid_map.get(next_alias, {})
                n_global = global_map.get(next_alias, 0)

                # TODO: 这里内存占用很大，可以不要append了，改成维护一个limit_x大小的有序数组？

                t6 = time.time()
                for row in rows:
                    t_qid_str = row[-1]
                    n_pid_str = row[-2]
                    row_data = row[:-2]
                    n_qid_mask = self._translate_pid_bitmap(n_pid_str, n_map, n_global)
                    new_anno_mask = n_qid_mask & int(t_qid_str, 2)
                    score = bin(new_anno_mask & uncovered_mask_int).count('1')
                    scored_candidates.append((score, row_data, new_anno_mask))
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = scored_candidates[:limit_x]
                print(f"            Processed and scored joined rows in {time.time() - t6:.2f}s.")

                if not top_k:
                    return None
                
                # 如果已经是最后一步，直接返回best_tuple
                if step == join_tree[-1]:
                    best_score, best_row_data, best_qid_mask = top_k[0]
                    full_row_ids = {}
                    for i, col_name in enumerate(current_id_cols):
                        if col_name.endswith("_id") or col_name.endswith("_Id"):
                            for alias in join_graph.nodes:
                                if col_name == f"{alias}_id" or col_name == f"{alias}_Id":
                                    full_row_ids[alias] = best_row_data[i]
                                    break
                    for alias in join_graph.nodes:
                        if alias not in full_row_ids:
                            print(f"            Warning: ID for alias '{alias}' not found in selected row.")
                            return None
                    covered_indices = set()
                    temp_mask = best_qid_mask
                    qid = 0
                    while temp_mask > 0:
                        if temp_mask & 1:
                            covered_indices.add(qid)
                        temp_mask >>= 1
                        qid += 1
                    return {
                        'full_row': full_row_ids,
                        'covered_indices': covered_indices
                    }

                create_sql = f"""
                    CREATE TEMP TABLE {temp_T_new_name} AS
                    SELECT
                        {prev_select_str},
                        {next_select_str},
                        {dummy_anno} AS anno
                    FROM {temp_T_name} T_tmp
                    JOIN {real_name} AS {next_alias}
                    ON {fixed_join_cond}
                    LIMIT 0;
                """
                t7 = time.time()
                self.cursor.execute(create_sql)
                self.cursor.execute(f"DROP TABLE IF EXISTS {temp_T_name};")
                self.cursor.execute(f"ALTER TABLE {temp_T_new_name} RENAME TO {temp_T_name};")
                print(f"            Created new temp table '{temp_T_name}' after joining '{next_alias}' in {time.time() - t7:.2f}s.")

                insert_values = []
                for score, row_data, anno_mask in top_k:
                    anno_str = format(anno_mask, f'0{total_queries}b')
                    insert_values.append(tuple(list(row_data) + [anno_str]))
                insert_sql = f"INSERT INTO {temp_T_name} VALUES %s"
                t8 = time.time()
                execute_values(self.cursor, insert_sql, insert_values)
                print(f"            Inserted top-{len(top_k)} candidates into '{temp_T_name}' after joining '{next_alias}' in {time.time() - t8:.2f}s.")
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error during greedy join selection: {e}")
            return None
        finally:
            self.cursor.execute(f"DROP TABLE IF EXISTS {temp_T_name};")

    def sample_for_one_template(self, partitions, root_table, join_order_tree, template_data):
        """
        Algorithm 3
        参考 sampler.py 的 sample 函数结构
        针对单个 Template，构建 k 个样本集。
        """

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

        root_partition_cache = {}

        for k in range(self.k_bitmaps):
            bitmap_start_time = time.time()
            print(f"    --> Constructing Bitmap {k+1}/{self.k_bitmaps}...")
            current_bitmap = []

            if len(global_covered_queries) == total_queries and total_queries > 0:
                print("        All queries covered. Stop sampling.")
                break

            for p_idx, partition_ids in enumerate(partitions):
                partition_select_time_start = time.time()
                best_tuple_info = self.greedy_join_selection(
                    partition_ids=partition_ids,
                    partition_idx=p_idx,
                    root_cache=root_partition_cache,
                    root_table=root_table,
                    join_tree=join_order_tree,
                    template_data=template_data,
                    global_covered_mask=global_covered_queries, # 当前已覆盖的集合
                    limit_x=self.limit_x
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
        
        return generated_samples


    def sample(self):
        """
        [主入口]
        执行完整的 Join 采样流程，并保存结果。
        """

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
                print(f"\n=== Skipping Template: {template_id} (only {len(template_data['aliases'])} table) ===")
                continue

            print(f"\n=== Processing Template: {template_id} ===")
            print(f"    Tables: {template_data['aliases']}")
            print(f"    Total Queries in Group: {template_data['queries_count']}")

            join_graph = template_data['join_graph']
            aliases = template_data['aliases']

            try:
                join_order_tree, root_table = self.build_join_tree_structure(join_graph, aliases)
                print(f"    Join Root: {root_table}")

                # print("    Computing root weights...")
                # root_weights = self.compute_root_weights(root_table, join_order_tree, template_data)

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

                # 批量保存
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
        
        # 保存剩余样本
        if current_batch_samples:
            batch_filename = f"samples_batch_{batch_index}.json"
            batch_full_path = os.path.join(self.output_path, batch_filename)
            self.save_samples(current_batch_samples, batch_full_path)
            print(f"    >>> Saved final batch {batch_index} (Templates {processed_count - len(current_batch_samples) + 1} to {processed_count})...")
        print(f"\nAll tasks finished. Results saved to directory: {self.output_path}. Total Time: {time.time() - start_time:.2f}s")


    def save_samples(self, samples_dict, output_path):
        """
        保存采样结果+格式转换
        List[List[Dict{'alias': id}]] -> List[List[List[Header, Row1, Row2...]]]
        Header 按别名字典序排列 (e.g., ["mi.id", "t.id"])
        """

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

        # 序列化保存
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
