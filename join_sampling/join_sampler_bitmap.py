# 每个template单独加anno列，存qid

import pickle
import glob
import os
from collections import defaultdict
from networkx.readwrite import json_graph
import networkx as nx
import time
import json
import psycopg2
from psycopg2.extras import execute_values
import sys

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

class JoinSampler:
    def __init__(self, config_path: str):
        print(f"Initializing JoinSampler with config: {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        samp_conf = self.config.get("sampling", {})
        self.base_query_dir = samp_conf.get("base_query_dir", "./qrep")
        self.output_path = samp_conf.get("output_path", "./samples.json")

        self.m_partitions = samp_conf.get("m_partitions", 10)
        self.k_bitmaps = samp_conf.get("k_bitmaps", 5)
        self.limit_x = samp_conf.get("limit_x", 50)
        self.batch_size = samp_conf.get("batch_size", 1000)

        self.join_templates = {}

        db_conf = self.config.get("database", {})
        try:
            self.conn = psycopg2.connect(
                host=db_conf.get("host", "localhost"),
                port=db_conf.get("port", 5432),
                dbname=db_conf.get("dbname", "imdb"),
                user=db_conf.get("user", "your_username")
            )
            self.cursor = self.conn.cursor()
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


    def load_and_parse_workload(self):
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
            if template_name == "7a":
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

                for subplan_tuple in subset_graph.nodes():
                    if len(subplan_tuple) < 1:
                        continue

                    sorted_aliases = sorted(list(subplan_tuple))

                    join_conditions_sig = []

                    sub_graph = join_graph.subgraph(subplan_tuple)

                    edges_info = []

                    for u, v, data in sub_graph.edges(data=True):
                        # 确保 u, v 顺序一致，比如字母序
                        if u > v:
                            u, v = v, u

                        cond = data.get("join_condition", "")
                        if not cond:
                            print(f"Warning: No join condition found between {u} and {v} in {pkl_file}")
                            continue

                        cond_clean = cond.replace(" ", "") 

                        edges_info.append(f"{u}|{v}|{cond_clean}")

                    edges_info.sort()
                    join_sig_str = "||".join(edges_info)

                    template_key = (tuple(sorted_aliases), join_sig_str)

                    current_query_predicates = {}

                    for alias in subplan_tuple:
                        node_data = join_graph.nodes[alias]
                        preds_list = node_data.get("predicates", [])
                        if preds_list:
                            combined_pred = " AND ".join(preds_list)
                            current_query_predicates[alias] = combined_pred
                        else:
                            current_query_predicates[alias] = "TRUE"

                    temp_groups[template_key].append(current_query_predicates)

                    if template_key not in temp_graphs:
                        # 创建子图副本，保留边上的连接条件信息
                        temp_graphs[template_key] = join_graph.subgraph(subplan_tuple).copy()

        print(f"Parsing complete. Found {len(temp_groups)} distinct join templates.")

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

            # predicates 结构: {table: {query_idx: pred_str}}
            table_predicates_map = defaultdict(dict)

            for q_idx, preds_dict in enumerate(query_list):
                for alias, pred_sql in preds_dict.items():
                    table_predicates_map[alias][q_idx] = pred_sql
            
            self.join_templates[template_id] = {
                "aliases": list(aliases_tuple),
                "real_names": real_names,
                "join_graph": sub_graph,
                "queries_count": num_queries,
                "table_predicates": dict(table_predicates_map)
            }


    def compute_root_weights(self):
        pass


    # TODO: root不要选基数太大的表
    def build_join_tree_structure(self, join_graph, aliases):
        degrees = dict(join_graph.degree())
        root_table = max(aliases, key=lambda n: (degrees[n], n))

        visited = {root_table}
        queue = [root_table]

        join_execution_plan = [{
            'alias': root_table,
            'real_name': join_graph.nodes[root_table]['real_name'],
            'parent': None,
            'join_condition': None
        }]

        parent_map = {root_table: None}

        while queue:
            current_node = queue.pop(0)

            neighbors = list(join_graph.neighbors(current_node))
            neighbors.sort()

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent_map[neighbor] = current_node

                    # check
                    edge_data = join_graph.get_edge_data(current_node, neighbor)
                    raw_condition = edge_data.get("join_condition")

                    if not raw_condition:
                        raise ValueError(f"Missing join condition between {current_node} and {neighbor}")
                    
                    join_execution_plan.append({
                        'alias': neighbor,
                        'real_name': join_graph.nodes[neighbor]['real_name'],
                        'parent': current_node,
                        'join_condition': raw_condition
                    })
        
        if len(join_execution_plan) != len(aliases):
            print(f"WARNING: Subgraph might be disconnected. Expected {len(aliases)} nodes, got {len(join_execution_plan)}.")

        return join_execution_plan, root_table


    # 划分可以先改成对第一个表平均划分，先随机分吧
    def partition_root_table(self, root_table, m_partitions, template_data):
        """
        将root划分为 m 个分区，返回每个分区的 id 列表。
        """
        real_name = template_data['real_names'][root_table]

        try:
            self.cursor.execute(f"SELECT id FROM {real_name} ORDER BY RANDOM();")
            all_ids = [row[0] for row in self.cursor.fetchall()]

            total_rows = len(all_ids)

            # 行数少于分区数时，调整分区数，每个分区一个 ID
            if total_rows < m_partitions:
                m_partitions = total_rows

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


    def add_annotation_columns(self, template_data):
        """
        [Algorithm 4 Pre-computation]
        为当前 Template 中的每个表物理添加 'anno' 列并计算填充

        优化：
        1. 忽略 TRUE 谓词，只将有具体条件的 Query ID 写入数据库。
        2. 返回一个 always_true_ids 列表，供后续查询时动态拼装。
        """
        print(f"    [Annotation] Start.")

        aliases = template_data['aliases']
        real_names = template_data['real_names']
        table_predicates = template_data['table_predicates']
        total_queries = template_data['queries_count']

        # 用于存储每个表的“全表覆盖查询ID”
        # 结构: { 'mi': [0, 5], 't': [1] }
        template_data['always_true_ids_map'] = defaultdict(list)

        zero_string = '0' * total_queries

        ROW_BATCH_SIZE = 100000
        UNIQUE_PRED_BATCH_SIZE = 500

        start_total = time.time()
        for alias in aliases:
            print(f"        Adding annotation column to table '{alias}'...")
            real_name = real_names[alias]
            preds_map = table_predicates.get(alias, {})
            anno_col = f"anno_{alias}"

            always_true_ids = []
            unique_preds_map = defaultdict(list)

            for q_idx in range(total_queries):
                pred = preds_map.get(q_idx, "TRUE")
                clean_pred = pred.strip()
                if not clean_pred or clean_pred.upper() == "TRUE":
                    always_true_ids.append(q_idx)
                else:
                    unique_preds_map[clean_pred].append(q_idx)

            template_data['always_true_ids_map'][alias] = always_true_ids
            sorted_unique_items = list(unique_preds_map.items())
            total_preds = len(sorted_unique_items)

            print(f"            Total Queries: {total_queries}, Always True: {len(always_true_ids)}, Unique Predicates: {total_preds}")

            try:
                self.cursor.execute(f"SELECT min(id), max(id) FROM {real_name}")
                min_id, max_id = self.cursor.fetchone()
                if min_id is None: min_id = 0
                if max_id is None: max_id = 0
                
                total_row_batches = (max_id - min_id) // ROW_BATCH_SIZE + 1

                self.cursor.execute(f"ALTER TABLE {real_name} DROP COLUMN IF EXISTS {anno_col};")
                self.cursor.execute(f"ALTER TABLE {real_name} ADD COLUMN {anno_col} BIT VARYING({total_queries}) DEFAULT B'{zero_string}';")
                self.conn.commit()

                if not unique_preds_map:
                    continue

                print("            Pre-compiling SQL fragments...")
                sql_chunks = []
                for i in range(0, total_preds, UNIQUE_PRED_BATCH_SIZE):
                    batch_items = sorted_unique_items[i : i + UNIQUE_PRED_BATCH_SIZE]
                    expr_parts = []
                    where_conditions = []

                    for pred_sql, q_ids in batch_items:
                        where_conditions.append(pred_sql)

                        mask_list = ['0'] * total_queries
                        for qid in q_ids:
                            mask_list[qid] = '1'
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

                    for i, (full_expr, pred_where) in enumerate(sql_chunks):
                        update_sql = f"""
                            UPDATE {real_name} {alias}
                            SET {anno_col} = {anno_col} | ({full_expr})
                            WHERE {alias}.id >= {current_id} AND {alias}.id < {next_id}
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
                    self.cursor.execute(f"ANALYZE {real_name};")
                finally:
                    self.conn.set_isolation_level(old_isolation_level)  # 恢复原始隔离级别

            except Exception as e:
                print(f"Error adding annotation column to table '{real_name}': {e}")
                self.conn.rollback()
                raise e
            
        print(f"    [Annotation] Finished. Total time: {time.time() - start_total:.2f}s")

    def remove_annotation_columns(self, template_data):
        """
        移除当前 Template 中每个表的 'anno' 列。
        """
        aliases = template_data['aliases']
        real_names = template_data['real_names']

        for alias in aliases:
            real_name = real_names[alias]
            anno_col = f"anno_{alias}"

            try:
                self.cursor.execute(f"ALTER TABLE {real_name} DROP COLUMN IF EXISTS {anno_col};")
                self.conn.commit()
            except Exception as e:
                print(f"Error removing annotation column from table '{real_name}': {e}")
                self.conn.rollback()
                raise e


    def greedy_join_selection(self, partition_ids, root_table, join_tree, template_data, global_covered_mask, limit_x):
        """
        Algorithm 4
        """
        join_graph = template_data['join_graph']
        total_queries = template_data['queries_count']
        all_query_ids = set(range(total_queries))
        uncovered_ids = list(all_query_ids - global_covered_mask)
        mask_list = ['0'] * total_queries
        for qid in uncovered_ids:
            mask_list[qid] = '1'
        uncovered_mask_str = "".join(mask_list)

        # if not uncovered_ids:
        #     order_clause = "bit_count(anno) DESC"
        # else:
        #     order_clause = f"bit_count(anno & B'{uncovered_mask_str}') DESC"

        # Effective_Length = Cardinality( Anno INTERSECT Uncovered_Set )

        try:
            self.cursor.execute("DROP TABLE IF EXISTS temp_T")
            self.cursor.execute("DROP TABLE IF EXISTS temp_M")

            if not partition_ids: return None
            ids_values = ",".join(f"({uid})" for uid in partition_ids)

            always_true_map = template_data.get('always_true_ids_map', {})

            current_id_cols = []

            root_real_name = template_data['real_names'][root_table]
            root_sels = join_graph.nodes[root_table]['sels']

            root_cols_sql = []
            for sel in root_sels:
                # 在视图中的列重命名为 Alias_Column 形式
                col_pure = sel.split('.')[-1] 
                root_cols_sql.append(f"{col_pure} AS {root_table}_{col_pure}")
                current_id_cols.append(f"{root_table}_{col_pure}")

            root_select_str = ", ".join(root_cols_sql)

            # 辅助：构建 Global Mask
            def get_global_mask_str(alias):
                g_ids = always_true_map.get(alias, [])
                g_list = ['0'] * total_queries
                for qid in g_ids: 
                    if qid < total_queries:
                        g_list[qid] = '1'
                return "".join(g_list)
            
            def get_order_sql(anno_expression):
                if not uncovered_ids:
                    return f"bit_count({anno_expression}) DESC"
                else:
                    return f"bit_count({anno_expression} & B'{uncovered_mask_str}') DESC"
            
            root_g_mask = get_global_mask_str(root_table)
            root_anno_col = f"anno_{root_table}"
            root_anno_expr = f"({root_anno_col} | B'{root_g_mask}')"

            order_sql_step1 = get_order_sql(root_anno_expr)

            sql_step1 = f"""
                CREATE TEMP TABLE temp_T AS
                WITH partition_filter(pid) AS ( VALUES {ids_values} )
                SELECT {root_select_str}, {root_anno_expr} AS anno
                FROM {root_real_name} AS {root_table}
                JOIN partition_filter pf ON {root_table}.id = pf.pid
                ORDER BY {order_sql_step1}
                LIMIT {limit_x};
            """

            # with open("temp_sql_debug.sql", "w") as debug_f:
            #     debug_f.write(sql_step1 + "\n\n")

            self.cursor.execute(sql_step1)

            for step in join_tree[1:]:
                next_alias = step['alias']
                real_name = step['real_name']
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

                next_g_mask = get_global_mask_str(next_alias)
                next_anno_col = f"{next_alias}.anno_{next_alias}"
                next_anno_expr = f"({next_anno_col} | B'{next_g_mask}')"

                fixed_join_cond = raw_join_cond.replace(f"{parent_alias}.", f"T_tmp.{parent_alias}_")

                self.cursor.execute("DROP TABLE IF EXISTS temp_M;")

                sql_join = f"""
                    CREATE TEMP TABLE temp_M AS
                    SELECT
                        {prev_select_str},
                        {next_select_str},
                        (T_tmp.anno & {next_anno_expr}) AS anno
                    FROM temp_T T_tmp
                    JOIN {real_name} AS {next_alias}
                    ON {fixed_join_cond};
                """

                # with open("temp_sql_debug.sql", "a") as debug_f:
                #     debug_f.write(sql_join + "\n\n")

                self.cursor.execute(sql_join)
                self.cursor.execute("DROP TABLE IF EXISTS temp_T;")

                order_sql_prune = get_order_sql("anno")
                sql_prune = f"""
                    CREATE TEMP TABLE temp_T AS
                    SELECT * FROM temp_M
                    ORDER BY {order_sql_prune}
                    LIMIT {limit_x};
                """

                # with open("temp_sql_debug.sql", "a") as debug_f:
                #     debug_f.write(sql_prune + "\n\n")

                self.cursor.execute(sql_prune)

            self.cursor.execute("SELECT * FROM temp_T LIMIT 1;")
            row = self.cursor.fetchone()

            if row is None:
                return None
            
            # 提取所有id列，把“_id”结尾的列都当成 id 列
            # check 这个列名提取方式对吗？
            full_row_ids = {}
            col_names = [desc[0] for desc in self.cursor.description]
            anno_bits = row[-1] # 最后一列是 anno

            for i, col_name in enumerate(col_names[:-1]):
                if col_name.endswith("_id") or col_name.endswith("_Id"):
                    for alias in join_graph.nodes:
                        if col_name == f"{alias}_id" or col_name == f"{alias}_Id":
                            full_row_ids[alias] = row[i]
                            break

            covered_indices = set()
            if anno_bits:
                for idx, bit in enumerate(anno_bits):
                    if bit == '1':
                        covered_indices.add(idx)
            
            return {
                'full_row': full_row_ids,
                'covered_indices': covered_indices
            }
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error during greedy join selection: {e}")
            sys.exit(1)
            return None
        finally:
            self.cursor.execute("DROP TABLE IF EXISTS temp_T;")
            self.cursor.execute("DROP TABLE IF EXISTS temp_M;")

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

        try:
            self.add_annotation_columns(template_data)
        except Exception as e:
            print(f"Error during annotation column addition: {e}")
            self.remove_annotation_columns(template_data)
            return generated_samples

        global_covered_queries = set()
        total_queries = template_data['queries_count']

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

        if not self.join_templates:
            print("No join templates found. Exiting.")
            return
        
        final_samples = {}

        for template_id, template_data in self.join_templates.items():
            # if template_id != "ci_rt_t_9527a4":
            #     continue  # 仅处理特定模板以进行测试

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
                
                final_samples[template_id] = template_samples

                print(f"    Sampling completed in {time.time() - time_sampling_start:.2f}s.")

            except Exception as e:
                print(f"ERROR processing template {template_id}: {e}")
                import traceback
                traceback.print_exc()
        
        self.save_samples(final_samples, self.output_path)
        print(f"\nAll tasks finished. Samples saved to {self.output_path}.")


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
                print(f"Samples successfully saved to {output_path}.")
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
