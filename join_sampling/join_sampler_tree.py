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
        self.template_ids = [] # template_key (alias_tuple, sig)
        self.pending_instances = [] # 暂存
        self.end_template_keys = [] # 暂存

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

class JoinPathTrie:
    def __init__(self):
        self.root = TrieNode("ROOT_SENTRY")

    def insert(self, template_key, path_signature, qid_list, join_graph):
        node = self.root
        for qid in qid_list:
            node.relevant_qids.add(qid)

        for step_token in path_signature:
            # step_token: (parent, child, condition)
            if step_token not in node.children:
                new_node = TrieNode(step_token)

                parent, child, cond = step_token
                new_node.parent_alias = parent
                new_node.child_alias = child
                new_node.join_condition = cond

                if child in join_graph.nodes:
                    new_node.real_name = join_graph.nodes[child]['real_name']

                node.children[step_token] = new_node

            node = node.children[step_token]

            for qid in qid_list:
                node.relevant_qids.add(qid)

        node.is_template_end = True
        node.template_ids.append(template_key)


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
                node = self.trie.root
                for step_token in path_signature:
                    if step_token not in node.children:
                        new_node = TrieNode(step_token)

                        parent, child, cond = step_token
                        new_node.parent_alias = parent
                        new_node.child_alias = child
                        new_node.join_condition = cond

                        if child in join_graph.nodes:
                            new_node.real_name = join_graph.nodes[child]['real_name']

                            new_node.sels = join_graph.nodes[child].get('sels', [])

                        node.children[step_token] = new_node
                    
                    node = node.children[step_token]

                node.is_template_end = True
                node.pending_instances.extend(instances)
                node.end_template_keys.append(template_key)

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