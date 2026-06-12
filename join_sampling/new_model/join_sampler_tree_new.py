import sys
import os
import datetime

# 获取当前脚本所在目录: .../Sampler/join_sampling/joblight-stats
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 定位到 Sampler 根目录 (../../)
project_root = os.path.abspath(os.path.join(curr_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 定位到 mscn 目录 (.../Sampler/mscn)
mscn_root = os.path.join(project_root, "mscn")
if mscn_root not in sys.path:
    sys.path.append(mscn_root)

from mscn.query_representation.utils import *

import pickle
import glob
import io
from collections import defaultdict
from networkx.readwrite import json_graph
import networkx as nx
import time
import json
import psycopg2
from psycopg2.extras import execute_values
import re
import heapq
import itertools
import hashlib

import sqlglot
from sqlglot import exp

from wander_join import WanderJoinEngine


# --- 工具函数 ---
def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)
    query["subset_graph"] = nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    return query

def normalize_condition(cond_str):
    if not cond_str or cond_str == "None": return "None"
    parts =[p.strip() for p in cond_str.split('=')]
    parts.sort()
    return "=".join(parts)

def timestamp_to_string(timestamp_str):
    unix_timestamp = int(timestamp_str)
    dt = datetime.datetime.fromtimestamp(unix_timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def convert_timestamps_in_sql(sql):
    pattern = r'(\w+\.(\w+))\s*([<>=!]+)\s*(\d+)'
    def replace_match(match):
        full_col = match.group(1)
        col_name = match.group(2)
        op = match.group(3)
        value = match.group(4)
        if col_name.endswith('date') or col_name.endswith('Date'):
            converted_value = timestamp_to_string(value)
            return f"{full_col} {op} '{converted_value}'::timestamp"
        else:
            return match.group(0)
    return re.sub(pattern, replace_match, sql)

def remove_unnecessary_parentheses(sql_str):
    match = re.search(r'\bWHERE\b', sql_str, re.IGNORECASE)
    if not match: return sql_str 
    where_idx = match.end()
    select_from_part = sql_str[:where_idx] 
    where_part = sql_str[where_idx:].strip()
    if where_part.endswith(";"): where_part = where_part[:-1].strip()
    conditions = re.split(r'\s+AND\s+', where_part, flags=re.IGNORECASE)
    clean_conditions =[]
    for cond in conditions:
        cond = cond.strip()
        while cond.startswith("(") and cond.endswith(")"):
            cond = cond[1:-1].strip()
        clean_conditions.append(cond)
    return select_from_part + " " + " AND ".join(clean_conditions)


TABLE_CARD_IMDB = {
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

TABLE_CARD_STATS = {
    "badges": 79851,
    "comments": 174305,
    "posts": 91976,
    "users": 40325,
    "votes": 328064,
    "posthistory": 303187,
    "postlinks": 11102,
    "tags": 1032,
}

TABLE_CARD_ERGASTF1 = {
    "laptimes": 420369,
    "driverstandings": 31578,
    "target": 31578,
    "results": 23657,
    "constructorstandings": 11836,
    "constructorresults": 11082,
    "qualifying": 7397,
    "pitstops": 6070,
    "races": 976,
    "drivers": 840,
    "constructors": 208,
    "status": 134,
    "circuits": 73,
    "seasons": 68,
}

TABLE_CARD_GENOME = {
    "img_obj": 1750617,
    "img_obj_att": 1074676,
    "img_rel": 763159,
    "att_classes": 699,
    "obj_classes": 300,
    "pred_classes": 150,
}



# ================= 数据结构: Trie 树 =================

class TrieNode:
    def __init__(self, token):
        self.token = token  # (parent_alias, child_alias, condition)
        self.children = {}  # Dict[StepToken, TrieNode]
        self.parent = None  

        # Template Info
        self.is_template_end = False
        self.end_template_keys = [] 
        self.pending_instances =[]

        # QID Info
        self.relevant_qids = set()   # 子树中所有查询的并集
        self.qid_map = {}            # {global_qid: {alias: pid}} 
        self.current_qids = set()    # 仅在该点结束的查询实例
        self.subtree_total_queries = 0 

        # Graph Info
        self.child_alias = None
        self.real_name = None
        self.join_condition = None
        self.parent_alias = None
        self.sels =[]

        # 采样状态: 缓存当前到达该节点的所有有效游走路径
        self.cached_paths =[]  # List of {'vals': dict, 'acc_bmp': int, 'alive': bool}

class JoinPathTrie:
    def __init__(self):
        self.root = TrieNode("ROOT_SENTRY")

    def insert(self, template_key, join_execution_plan, instances):
        node = self.root
        for step in join_execution_plan:
            step_token = (step['parent'], step['alias'], step['join_condition'])
            
            if step_token not in node.children:
                new_node = TrieNode(step_token)
                new_node.parent = node
                new_node.parent_alias = step['parent']
                new_node.child_alias = step['alias']
                new_node.join_condition = step['join_condition']
                new_node.real_name = step['real_name']
                new_node.sels = list(step['sels'])
                node.children[step_token] = new_node
            
            node = node.children[step_token]
            
            # 合并 Sels
            current_sels_set = set(node.sels)
            for s in step['sels']:
                if s not in current_sels_set:
                    node.sels.append(s)

        node.is_template_end = True
        node.pending_instances = instances # 暂存实例: List[(template_id, q_idx, pids_dict)]
        node.end_template_keys.append(template_key)


# ================= 主类: JoinSampler =================

class JoinSampler:
    def __init__(self, config_path: str):
        print(f"Initializing JoinSampler with config: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        samp_conf = self.config.get("sampling", {})
        self.sql_file = samp_conf.get("query_file", "./qrep")
        self.workload_name = samp_conf.get("workload_name", "")
        self.output_path = samp_conf.get("output_path", "./samples.json")
        self.skip_7a = samp_conf.get("skip_7a", True)

        self.m_partitions = samp_conf.get("m_partitions", 10)
        self.k_bitmaps = samp_conf.get("k_bitmaps", 5)
        
        # [修改] 替换 lookahead 配置为 Monte Carlo 采样大小 w
        self.w_samples = samp_conf.get("w_samples", 10) # 对应算法中的采样大小 w

        self.global_predicate_map = defaultdict(lambda: {}) 
        self.global_pid_counters = defaultdict(int)
        self.global_pid_to_pred = defaultdict(dict)  

        db_conf = self.config.get("database", {})
        self.db_config = {
            "host": db_conf.get("host", "localhost"),
            "port": db_conf.get("port", 5432),
            "dbname": db_conf.get("dbname", "imdb"),
            "user": db_conf.get("user", "your_username"),
            "password": db_conf.get("password", "123")
        }

        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise e
        
        self.engine = WanderJoinEngine(self.conn, self.cursor)
        self.tie_breaker = itertools.count()
        self.trie = JoinPathTrie()
        self.all_involved_tables = set()

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
        if self.engine: self.engine.close()
        print("Resources released.")

    def load_and_parse_workload(self):
        self.temp_template_data = defaultdict(lambda: {'instances':[], 'graph': None})
        try:
            with open(self.sql_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error loading {self.sql_file}: {e}")
            return
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            sql_str = line.split("||")[0].strip()
            if sql_str.startswith("/*"): sql_str = sql_str.split("*/", 1)[-1].strip()
            if "SELECT" not in sql_str.upper(): continue

            try:
                sql_str = remove_unnecessary_parentheses(sql_str)
                join_graph = extract_join_graph(sql_str)
            except Exception as e:
                continue

            sorted_aliases = sorted(list(join_graph.nodes()))
            if len(sorted_aliases) < 2: continue

            col_graph = nx.Graph()
            for u, v, data in join_graph.edges(data=True):
                if u > v: u, v = v, u
                cond = data.get("join_condition", "")
                if not cond: continue
                cond_clean = normalize_condition(cond)
                parts = cond_clean.split("=")
                if len(parts) == 2:
                    col_graph.add_edge(parts[0].strip(), parts[1].strip())

            eq_classes = list(nx.connected_components(col_graph))
            class_signatures =["=".join(sorted(list(eq_class))) for eq_class in eq_classes]
            class_signatures.sort()
            join_sig_str = "||".join(class_signatures)

            template_key = (tuple(sorted_aliases), join_sig_str)
            current_query_pids = {}

            for alias in sorted_aliases:
                real_name = join_graph.nodes[alias]["real_name"]
                self.all_involved_tables.add(real_name)
                preds_list = sorted(join_graph.nodes[alias].get("predicates", []))
                clean_pred_list =[]
                for pred in preds_list:
                    if "stats" in self.db_config.get("dbname", "").lower():
                        pred = convert_timestamps_in_sql(pred)
                    pred_clean = re.sub(fr"\b{alias}\.", "", pred).strip()
                    pred_clean = re.sub(r'\s*([<>=!]+)\s*', r'\1', pred_clean)
                    clean_pred_list.append(pred_clean)

                if clean_pred_list:
                    combined_pred = " AND ".join(clean_pred_list)
                    if combined_pred not in self.global_predicate_map[real_name]:
                        pid = self.global_pid_counters[real_name]
                        self.global_predicate_map[real_name][combined_pred] = pid
                        self.global_pid_counters[real_name] += 1
                        self.global_pid_to_pred[real_name][pid] = combined_pred
                    current_query_pids[alias] = self.global_predicate_map[real_name][combined_pred]
                else:
                    current_query_pids[alias] = -1 

            if template_key not in self.temp_template_data:
                self.temp_template_data[template_key]['graph'] = join_graph.copy()
                self.temp_template_data[template_key]['real_names'] = {a: join_graph.nodes[a]["real_name"] for a in sorted_aliases}
            
            # Store q_idx inherently via list length
            q_idx = len(self.temp_template_data[template_key]['instances'])
            self.temp_template_data[template_key]['instances'].append(current_query_pids)

    def add_sel_info_to_graph(self, join_graph):
        for node, info in join_graph.nodes(data=True):
            sels =[]
            for u, v, edge_data in join_graph.edges(node, data=True):
                conds = edge_data.get("join_condition", "").split("!=") if "!" in edge_data.get("join_condition", "") else edge_data.get("join_condition", "").split("=")
                for jc in conds:
                    jc = jc.strip()
                    jc_alias = jc.split(".")[0]
                    if node == jc_alias and jc not in sels:
                        sels.append(jc)
                    jc_node = jc.split(".")[0]
                    join_graph[u][v][jc_node] = jc
            
            if f"{node}.id" not in sels and f"{node}.Id" not in sels:
                sels = [f"{node}.id"] + sels
            else:
                sels =[f"{node}.id"] +[s for s in sels if s != f"{node}.id" and s != f"{node}.Id"]
            join_graph.nodes[node]["sels"] = sels

    def build_join_tree_structure(self, join_graph, aliases):
        aliases = sorted(aliases)
        scored =[]
        for a in aliases:
            real_name = join_graph.nodes[a]['real_name']
            if "imdb" in self.db_config.get("dbname", "").lower():
                card = TABLE_CARD_IMDB.get(real_name.lower(), float("inf"))
            elif "stats" in self.db_config.get("dbname", "").lower():
                card = TABLE_CARD_STATS.get(real_name.lower(), float("inf"))
            elif "ergastf1" in self.db_config.get("dbname", "").lower():
                card = TABLE_CARD_ERGASTF1.get(real_name.lower(), float("inf"))
            elif "genome" in self.db_config.get("dbname", "").lower():
                card = TABLE_CARD_GENOME.get(real_name.lower(), float("inf"))
            else:
                print(f"Warning: Unknown database '{self.db_config.get('dbname', '')}', defaulting cardinality to infinity.")
                card = float("inf")
            scored.append((card, a))
        scored.sort()

        root_table = aliases[0]
        for card, alias in scored:
            if card > 5000:
                root_table = alias
                break
        if "imdb" in self.db_config.get("dbname", "").lower():
            if root_table in ['imdb_ci', 'imdb_mi', 'imdb_pi']: root_table = scored[0][1]

        visited = {root_table}
        join_execution_plan =[{
            'alias': root_table, # 当前节点的alias
            'real_name': join_graph.nodes[root_table]['real_name'],
            'parent': None,
            'join_condition': None,
            'sels': join_graph.nodes[root_table].get('sels',[])
        }]

        while len(visited) < len(aliases):
            candidates =[]
            for u in sorted(visited):
                for v in sorted(join_graph.neighbors(u)):
                    if v not in visited:
                        real_name = join_graph.nodes[v]['real_name']
                        if "imdb" in self.db_config.get("dbname", "").lower():
                            card = TABLE_CARD_IMDB.get(real_name.lower(), float("inf"))
                        elif "stats" in self.db_config.get("dbname", "").lower():
                            card = TABLE_CARD_STATS.get(real_name.lower(), float("inf"))
                        elif "ergastf1" in self.db_config.get("dbname", "").lower():
                            card = TABLE_CARD_ERGASTF1.get(real_name.lower(), float("inf"))
                        elif "genome" in self.db_config.get("dbname", "").lower():
                            card = TABLE_CARD_GENOME.get(real_name.lower(), float("inf"))
                        else:
                            print(f"Warning: Unknown database '{self.db_config.get('dbname', '')}', defaulting cardinality to infinity.")
                            card = float("inf")
                        candidates.append((card, u, v))

            if not candidates: raise RuntimeError("Join graph is not connected")
            _, parent, child = min(candidates)
            visited.add(child)
            norm_cond = normalize_condition(join_graph.get_edge_data(parent, child).get("join_condition"))
            
            join_execution_plan.append({
                'alias': child,
                'real_name': join_graph.nodes[child]['real_name'],
                'parent': parent,
                'join_condition': norm_cond,
                'sels': join_graph.nodes[child].get('sels',[])
            })
        return join_execution_plan

    def build_global_trie_for_tasks(self, my_tasks):
        print("\nBuilding Trie from assigned tasks...")
        self.trie = JoinPathTrie()

        for template_key, data in my_tasks:
            join_graph = data['graph'].copy()
            self.add_sel_info_to_graph(join_graph)
            
            aliases_tuple = template_key[0]
            try:
                join_execution_plan = self.build_join_tree_structure(join_graph, list(aliases_tuple))
                self.trie.insert(template_key, join_execution_plan, data['instances'])
            except Exception as e:
                print(f"Error building trie for {aliases_tuple}: {e}")

        # QID 分配 (按 Root 子树隔离)
        for root_token, root_node in self.trie.root.children.items():
            current_qid_counter = 0

            def assign_and_aggregate(node):
                nonlocal current_qid_counter
                my_relevant = set()
                my_query = set()

                if node.is_template_end:
                    for inst in node.pending_instances:
                        qid = current_qid_counter
                        current_qid_counter += 1
                        node.qid_map[qid] = inst
                        my_relevant.add(qid)
                        my_query.add(qid)
                    node.pending_instances =[] 

                for child in node.children.values():
                    my_relevant.update(assign_and_aggregate(child))

                node.relevant_qids = my_relevant
                node.current_qids = my_query
                return my_relevant
            
            assign_and_aggregate(root_node)
            root_node.subtree_total_queries = current_qid_counter
            print(f"  Root '{root_node.child_alias}' ({root_node.real_name}): {current_qid_counter} queries.")

    def prepare_subtree_pid_map(self, root_node):
        subtree_instances = {}
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.is_template_end:
                subtree_instances.update(node.qid_map)
            stack.extend(node.children.values())
        
        pid_map = defaultdict(dict)
        global_map = defaultdict(int)
        for qid, inst in subtree_instances.items():
            for alias, pid in inst.items():
                if pid == -1:
                    global_map[alias] |= (1 << qid)
                else:
                    if pid not in pid_map[alias]: pid_map[alias][pid] = 0
                    pid_map[alias][pid] |= (1 << qid)
        return pid_map, global_map

    def partition_root_table(self, real_name, m_partitions):
        try:
            self.cursor.execute(f"SELECT id FROM {real_name} ORDER BY RANDOM();")
            all_ids =[row[0] for row in self.cursor.fetchall()]
            total_rows = len(all_ids)
            if total_rows < m_partitions:
                return [[all_ids[i % total_rows]] for i in range(m_partitions)]

            base_size = total_rows // m_partitions
            remainder = total_rows % m_partitions
            partitions, start_idx =[], 0
            for i in range(m_partitions):
                part_size = base_size + (1 if i < remainder else 0)
                partitions.append(all_ids[start_idx:start_idx + part_size])
                start_idx += part_size
            return partitions
        except Exception as e:
            raise e

    # ================= 核心执行逻辑 =================

    def _get_uncovered_mask(self, relevant_qids, covered_mask_int):
        mask = 0
        for qid in relevant_qids:
            if not (covered_mask_int & (1 << qid)):
                mask |= (1 << qid)
        return mask
    
    def execute_root_step(self, root_node, partition_ids, pid_map, global_map, subtree_covered_mask):
        """
        Root初始化：获取Root元组，并按照 Monte Carlo 大小 w 进行复制，建立初始的平行路径。
        """
        if not partition_ids: return False

        uncovered_mask_int = self._get_uncovered_mask(root_node.relevant_qids, subtree_covered_mask)
        if uncovered_mask_int == 0: return False

        neighbors, _ = self.engine._batch_fetch_neighbors(
            root_node.real_name, "id", partition_ids, root_node.sels, root_node.child_alias
        )

        translated_map, _ = self.engine._batch_fetch_translate_bitmaps(
            root_node.real_name, partition_ids,
            workload_name=self.workload_name, alias=root_node.child_alias,
            pid_map=pid_map.get(root_node.child_alias, {}),
            global_map=global_map.get(root_node.child_alias, 0)
        )

        root_paths =[]
        for pid in set(partition_ids):
            p_val = str(pid)
            rows = neighbors.get(p_val,[])
            if not rows: continue
            
            row_data = rows[0]
            qid_mask = translated_map.get(p_val, global_map.get(root_node.child_alias, 0))

            # 剪枝
            if (qid_mask & uncovered_mask_int) == 0:
                continue

            clean_data = {k: v for k, v in row_data.items() if k != '_bmp_str'}

            # [核心逻辑] Monte Carlo ：每个 Root 元组复制 w 份，形成 w 条平行路径
            for _ in range(self.w_samples):
                root_paths.append({
                    'vals': clean_data.copy(),
                    'acc_bmp': qid_mask,
                    'alive': True
                })

        root_node.cached_paths = root_paths
        return len(root_node.cached_paths) > 0

    def dfs_join_recursive(self, current_node, subtree_covered_mask, pid_map, global_map):
        """
        深度优先遍历: 向下执行 1 步 Wander Join，并将结果缓存在当前节点。
        """
        collected_samples = defaultdict(list)
        newly_covered_global = 0

        uncovered_mask_int = self._get_uncovered_mask(current_node.relevant_qids, subtree_covered_mask)
        if uncovered_mask_int == 0: 
            current_node.cached_paths =[]
            return collected_samples, newly_covered_global

        parent_node = current_node.parent
        if not parent_node.cached_paths: 
            print(f"Warning: No cached paths to extend at node {current_node.child_alias}. This should not happen if the algorithm is correct. Returning empty results for this branch.")
            current_node.cached_paths =[]
            return collected_samples, newly_covered_global

        step_info = {
            'alias': current_node.child_alias,
            'real_name': current_node.real_name,
            'parent': current_node.parent_alias,
            'join_condition': current_node.join_condition,
            'sels': current_node.sels
        }

        # [核心逻辑] 从父节点缓存的路径出发，做且只做 1 步 Wander Join
        current_node.cached_paths = self.engine.extend_paths_one_step(
            parent_node.cached_paths, step_info, pid_map, global_map, self.workload_name
        )
        
        if not current_node.cached_paths: 
            return collected_samples, newly_covered_global

        # 如果是模版终点，在当前的所有缓存路径中打分，挑出最高分
        if current_node.is_template_end:
            sample_dict, covered_mask_update = self.collect_best_sample_for_node(current_node, subtree_covered_mask)
            if sample_dict:
                newly_covered_global |= covered_mask_update
                subtree_covered_mask |= covered_mask_update 
                for tmpl_key in current_node.end_template_keys:
                    collected_samples[tmpl_key].append(sample_dict)

        # 递归遍历所有子分支，子分支会自动复用当前节点的 cached_paths
        for child in current_node.children.values():
            child_results, child_covered = self.dfs_join_recursive(
                child, subtree_covered_mask, pid_map, global_map
            )
            newly_covered_global |= child_covered
            subtree_covered_mask |= child_covered
            for k, v in child_results.items():
                collected_samples[k].extend(v)

        current_node.cached_paths =[]

        return collected_samples, newly_covered_global

    def collect_best_sample_for_node(self, node, current_covered_mask):
        """
        在所有存活的 Monte Carlo 路径中，寻找对当前模板的未覆盖查询带来最大收益的1个Tuple
        """
        if not node.cached_paths: return None, 0

        best_path = None
        best_score = -1
        best_newly_covered_mask = 0

        # 打分选优 (Identify t'_{join} among the w samples with maximal coverage)
        for path in node.cached_paths:
            anno_bits = path['acc_bmp']
            score = 0
            newly_covered_mask = 0
            
            # 只关心当前模板所包含的查询 (current_qids)
            for qid in node.current_qids:
                if (anno_bits & (1 << qid)) and not (current_covered_mask & (1 << qid)):
                    score += 1
                    newly_covered_mask |= (1 << qid)
            
            if score > best_score:
                best_score = score
                best_path = path
                best_newly_covered_mask = newly_covered_mask

        # 如果分数都是0(没覆盖新查询)，我们依然返回其中一条(比如第一条)作为样本，以保证即使没新覆盖也有数据
        # 或者你也可以选择 return None, 0。这里采用返回最好结果（哪怕是0分）。
        if not best_path:
            best_path = node.cached_paths[0] 

        sample_dict = {}
        for k, v in best_path['vals'].items():
            if k.endswith(".id") or k.endswith(".Id"):
                alias = k.split('.')[0]
                sample_dict[alias] = v
        
        return sample_dict, best_newly_covered_mask

    
    # ================= 采样控制与保存 =================

    def sample_trie_root(self, root_node, cur_output_path, batch_index):
        partitions = self.partition_root_table(root_node.real_name, self.m_partitions)
        pid_map, global_map = self.prepare_subtree_pid_map(root_node)

        self.engine.bitmap_cache.clear()
        
        total_qids = root_node.subtree_total_queries
        target_full_mask = (1 << total_qids) - 1
        
        root_samples_dict = defaultdict(list) # {tmpl_key: [[sample1..], [sample2..] ] }

        subtree_covered_mask = 0

        for k_idx in range(self.k_bitmaps):
            print(f"    --> Bitmap {k_idx+1}/{self.k_bitmaps}...")
            
            # 本轮 K 所有 Template 的样本集合
            k_round_samples = defaultdict(list)

            for p_idx, partition_ids in enumerate(partitions):
                t_p = time.time()
                success = self.execute_root_step(root_node, partition_ids, pid_map, global_map, subtree_covered_mask)
                
                partition_newly_covered = 0
                
                if success:
                    if root_node.is_template_end:
                        s_dict, cov_mask = self.collect_best_sample_for_node(root_node, subtree_covered_mask)
                        if s_dict:
                            partition_newly_covered |= cov_mask
                            subtree_covered_mask |= cov_mask
                            for key in root_node.end_template_keys:
                                k_round_samples[key].append(s_dict)
                    
                    for child in root_node.children.values():
                        c_res, c_cov = self.dfs_join_recursive(child, subtree_covered_mask, pid_map, global_map)
                        partition_newly_covered |= c_cov
                        subtree_covered_mask |= c_cov
                        for key, v in c_res.items():
                            k_round_samples[key].extend(v)

                cov_count = bin(subtree_covered_mask).count('1')
                print(f"        Partition {p_idx+1}/{len(partitions)} done in {time.time()-t_p:.2f}s. Subtree Coverage: {cov_count}/{total_qids}")

                # if subtree_covered_mask == target_full_mask:
                #     break

                if bin(subtree_covered_mask).count('1') / total_qids >= 0.99:
                    print(f"        Coverage reached 99% after partition {p_idx+1}. Stop sampling further partitions for this bitmap.")
                    break
            
            # 将这一轮的样本存入总字典
            for tmpl_key, samples in k_round_samples.items():
                root_samples_dict[tmpl_key].append(samples)

            # 覆盖率达到99%以上就提前停止当前 Root 的采样
            if bin(subtree_covered_mask).count('1') / total_qids >= 0.99:
                print(f"        Coverage reached 99%. Stop sampling further bitmaps for this root.")
                break
                
            # if subtree_covered_mask == target_full_mask:
            #     print("        All queries covered. Stop sampling.")
            #     break
                
        # 收集完所有的 K，保存文件
        self.save_trie_samples(root_samples_dict, root_node.child_alias, cur_output_path, batch_index)

    def save_trie_samples(self, root_samples_dict, root_alias, output_path, batch_index):
        if not root_samples_dict: return
        formatted_output = {}
        for tmpl_key, k_bitmaps in root_samples_dict.items():
            aliases_tuple = tmpl_key[0]
            sig_hash = hashlib.md5(tmpl_key[1].encode('utf-8')).hexdigest()[:6]
            template_id = f"{'_'.join(aliases_tuple)}_{sig_hash}"
            
            formatted_bitmaps =[]
            for bitmap in k_bitmaps:
                if not bitmap:
                    formatted_bitmaps.append([])
                    continue
                aliases = sorted(bitmap[0].keys())
                header =[f"{alias}.id" for alias in aliases]
                formatted_rows = [header]
                for row_dict in bitmap:
                    formatted_rows.append([row_dict[alias] for alias in aliases])
                formatted_bitmaps.append(formatted_rows)
            formatted_output[template_id] = formatted_bitmaps

        filename = f"samples_batch_{batch_index}_{root_alias}.json"
        full_path = os.path.join(output_path, filename)
        
        def default_serializer(obj):
            if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
            return str(obj)
            
        with open(full_path, 'w') as f:
            json.dump(formatted_output, f, default=default_serializer, indent=2)
        print(f"    >>> Saved Trie Root '{root_alias}' to {filename}")

    # ================= 调度入口 =================

    def sample(self, worker_id=0, num_workers=1):
        print(f"Starting Join Sampling Process (Worker {worker_id}/{num_workers})...")
        t_start = time.time()
        
        self.load_and_parse_workload()
        if not self.temp_template_data:
            print("No templates parsed. Exiting.")
            return

        cur_output_path = os.path.join(self.output_path, str(worker_id))
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)

        # 任务分配
        all_items = list(self.temp_template_data.items())
        all_items.sort(key=lambda x: x[0][1]) # 按签名排序确保稳定
        my_tasks =[item for i, item in enumerate(all_items) if i % num_workers == worker_id]
        
        print(f"Worker {worker_id} assigned {len(my_tasks)} templates.")
        if not my_tasks: return

        # 构建局部 Trie 树
        self.build_global_trie_for_tasks(my_tasks)

        # 逐个 Root 处理并保存
        for batch_index, (root_token, root_node) in enumerate(self.trie.root.children.items()):
            print(f"\n=== Processing Trie Root: {root_node.child_alias} ({root_node.real_name}) ===")
            self.sample_trie_root(root_node, cur_output_path, batch_index)
            
        print(f"Worker {worker_id} finished in {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python join_sampler.py <config_path> <worker_id> <num_workers>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    worker_id = int(sys.argv[2])
    num_workers = int(sys.argv[3])

    sampler = JoinSampler(config_path)
    try:
        sampler.sample(worker_id=worker_id, num_workers=num_workers)
    finally:
        sampler.close()