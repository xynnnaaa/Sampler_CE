import sys
import os
import datetime

# 获取当前脚本所在目录
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

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
import re
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


# ================= 主类: JoinSampler (Baseline 线性版) =================

class JoinSampler:
    def __init__(self, config_path: str):
        print(f"Initializing JoinSampler (Linear Baseline) with config: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        samp_conf = self.config.get("sampling", {})
        self.sql_file = samp_conf.get("query_file", "./qrep")
        self.workload_name = samp_conf.get("workload_name", "")
        self.output_path = samp_conf.get("output_path", "./samples_linear.json")
        self.skip_7a = samp_conf.get("skip_7a", True)

        self.m_partitions = samp_conf.get("m_partitions", 10)
        self.k_bitmaps = samp_conf.get("k_bitmaps", 5)
        
        # 对应算法中的采样大小 w
        self.w_samples = samp_conf.get("w_samples", 10) 

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
        """在这里它生成的是一条线性的执行计划 (Join Path)"""
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

    def prepare_template_pid_map(self, instances):
        """为独立的 Template 构建 PID -> QID 映射"""
        pid_map = defaultdict(dict)
        global_map = defaultdict(int)
        
        total_queries = len(instances)
        
        for qid, inst in enumerate(instances):
            for alias, pid in inst.items():
                if pid == -1:
                    global_map[alias] |= (1 << qid)
                else:
                    if pid not in pid_map[alias]: pid_map[alias][pid] = 0
                    pid_map[alias][pid] |= (1 << qid)
                    
        return pid_map, global_map, total_queries


    # ================= 核心独立执行逻辑 =================

    def sample_for_one_template(self, template_id, template_data):
        """
        对单个 Template 执行 Monte Carlo Random Walk。
        严格不使用缓存和树结构。
        """

        self.engine.bitmap_cache.clear()

        
        join_graph = template_data['graph']
        self.add_sel_info_to_graph(join_graph)
        aliases = list(template_id[0])
        instances = template_data['instances']
        
        join_execution_plan = self.build_join_tree_structure(join_graph, aliases)
        root_info = join_execution_plan[0]
        
        partitions = self.partition_root_table(root_info['real_name'], self.m_partitions)
        pid_map, global_map, total_qids = self.prepare_template_pid_map(instances)
        
        target_full_mask = (1 << total_qids) - 1
        template_covered_mask = 0
        
        all_k_samples = []

        for k_idx in range(self.k_bitmaps):
            print(f"    --> Bitmap {k_idx+1}/{self.k_bitmaps}...", flush=True)
            
            current_bitmap_samples = []

            for p_idx, partition_ids in enumerate(partitions):
                if not partition_ids: continue
                
                t_p = time.time()
                
                # 1. 剪枝检查：看当前还缺多少覆盖
                uncovered_mask_int = target_full_mask & ~template_covered_mask
                if uncovered_mask_int == 0:
                    break

                # 2. Root 抓取
                neighbors, _ = self.engine._batch_fetch_neighbors(
                    root_info['real_name'], "id", partition_ids, root_info['sels'], root_info['alias']
                )

                translated_map, _ = self.engine._batch_fetch_translate_bitmaps(
                    root_info['real_name'], partition_ids,
                    workload_name=self.workload_name, alias=root_info['alias'],
                    pid_map=pid_map.get(root_info['alias'], {}),
                    global_map=global_map.get(root_info['alias'], 0)
                )

                # 3. 构建初始平行路径
                active_paths = []
                for pid in set(partition_ids):
                    p_val = str(pid)
                    rows = neighbors.get(p_val, [])
                    if not rows: continue
                    
                    row_data = rows[0]
                    qid_mask = translated_map.get(p_val, global_map.get(root_info['alias'], 0))
                    
                    # [极速剪枝]
                    if (qid_mask & uncovered_mask_int) == 0:
                        continue
                        
                    clean_data = {k: v for k, v in row_data.items() if k != '_bmp_str'}

                    # Monte Carlo 裂变 w 次
                    for _ in range(self.w_samples):
                        active_paths.append({
                            'vals': clean_data.copy(),
                            'acc_bmp': qid_mask,
                            'alive': True
                        })

                if not active_paths:
                    continue

                # 4. 线性向下走 (Wander Join 直到模板末尾)
                for step in join_execution_plan[1:]:
                    step_info = {
                        'alias': step['alias'],
                        'real_name': step['real_name'],
                        'parent': step['parent'],
                        'join_condition': step['join_condition'],
                        'sels': step['sels']
                    }
                    
                    active_paths = self.engine.extend_paths_one_step(
                        active_paths, step_info, pid_map, global_map, self.workload_name
                    )
                    
                    if not active_paths:
                        break # 此分区全军覆没
                
                # 5. 打分挑最好
                if not active_paths:
                    continue

                best_path = None
                best_score = -1
                best_new_cov = 0
                
                for path in active_paths:
                    anno_bits = path['acc_bmp']
                    score = bin(anno_bits & uncovered_mask_int).count('1')
                    if score > best_score:
                        best_score = score
                        best_path = path
                        best_new_cov = anno_bits & uncovered_mask_int
                        
                if not best_path:
                    best_path = active_paths[0]
                    
                # 提取最佳结果
                sample_dict = {}
                for k, v in best_path['vals'].items():
                    if k.endswith(".id") or k.endswith(".Id"):
                        alias = k.split('.')[0]
                        sample_dict[alias] = v
                
                current_bitmap_samples.append(sample_dict)
                template_covered_mask |= best_new_cov
                
                cov_count = bin(template_covered_mask).count('1')
                print(f"        Partition {p_idx+1}/{len(partitions)} done in {time.time()-t_p:.2f}s. Coverage: {cov_count}/{total_qids}")

                if bin(template_covered_mask).count('1') / total_qids >= 0.99:
                    print(f"        Coverage reached 99% after partition {p_idx+1}. Stop sampling partitions.")
                    break

            all_k_samples.append(current_bitmap_samples)
            
            if bin(template_covered_mask).count('1') / total_qids >= 0.99:
                print(f"        Coverage reached 99%. Stop sampling further bitmaps.")
                break
                
        return all_k_samples

    # ================= 调度与存储 =================

    def save_samples(self, samples_dict, output_path, batch_index):
        if not samples_dict: return
        formatted_output = {}
        for tmpl_key, k_bitmaps in samples_dict.items():
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

        filename = f"samples_batch_{batch_index}.json"
        full_path = os.path.join(output_path, filename)
        
        def default_serializer(obj):
            if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
            return str(obj)
            
        with open(full_path, 'w') as f:
            json.dump(formatted_output, f, default=default_serializer, indent=2)
        print(f"    >>> Saved batch {batch_index} to {filename}")

    def sample(self, worker_id=0, num_workers=1):
        print(f"Starting Join Sampling Process (Linear Baseline) (Worker {worker_id}/{num_workers})...")
        t_start = time.time()
        
        self.load_and_parse_workload()
        if not self.temp_template_data:
            print("No templates parsed. Exiting.")
            return

        cur_output_path = os.path.join(self.output_path, str(worker_id))
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)

        all_items = list(self.temp_template_data.items())
        all_items.sort(key=lambda x: x[0][1]) 
        my_tasks =[item for i, item in enumerate(all_items) if i % num_workers == worker_id]
        
        print(f"Worker {worker_id} assigned {len(my_tasks)} templates.")
        if not my_tasks: return

        # 线性逐个 Template 执行
        SAVE_BATCH_SIZE = 20
        current_batch_results = {}
        processed = 0

        for template_key, template_data in my_tasks:
            aliases_tuple = template_key[0]
            print(f"\n=== Processing Template: {aliases_tuple} ===")
            
            samples = self.sample_for_one_template(template_key, template_data)
            current_batch_results[template_key] = samples
            processed += 1
            
            if processed % SAVE_BATCH_SIZE == 0:
                batch_index = processed // SAVE_BATCH_SIZE
                self.save_samples(current_batch_results, cur_output_path, batch_index)
                current_batch_results = {}
                
        if current_batch_results:
            batch_index = (processed // SAVE_BATCH_SIZE) + 1
            self.save_samples(current_batch_results, cur_output_path, batch_index)
            
        print(f"Worker {worker_id} finished in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python join_sampler_linear.py <config_path> <worker_id> <num_workers>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    worker_id = int(sys.argv[2])
    num_workers = int(sys.argv[3])

    sampler = JoinSampler(config_path)
    try:
        sampler.sample(worker_id=worker_id, num_workers=num_workers)
    finally:
        sampler.close()