import os
import glob
import pickle
import json
import hashlib
import time
import networkx as nx
import psycopg2
from collections import defaultdict
from networkx.readwrite import json_graph

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

class Config:
    BASE_QUERY_DIR = "/data2/xuyining/Sampler/mscn/queries/ceb-imdb"
    
    OUTPUT_DIR = "/data2/xuyining/Sampler/join_sampling/sample_results/full_join"

    DB_CONFIG = {
        "host": "localhost",
        "port": "5433",
        "database": "imdb_new", 
        "user": "xuyining",
        "options": "-c statement_timeout=300000"
    }

    SKIP_TEMPLATE_7A = True
    SAMPLE_SIZE = 100


def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)
    query["subset_graph"] = nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    return query

def normalize_condition(cond_str):
    if not cond_str: return "None"
    parts = [p.strip() for p in cond_str.split('=')]
    parts.sort()
    return "=".join(parts)


class SimpleWorkloadSampler:
    def __init__(self, config):
        self.base_query_dir = config.BASE_QUERY_DIR
        self.output_dir = config.OUTPUT_DIR
        self.db_config = config.DB_CONFIG
        self.skip_7a = config.SKIP_TEMPLATE_7A
        self.sample_size = config.SAMPLE_SIZE
        
        # 存储唯一的模板结构
        # Key: Template ID (string)
        # Value: { 'aliases': [], 'join_graph': nx.Graph, 'real_names': {} }
        self.join_templates = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_parse_workload(self):
        """解析 Workload，提取唯一的 Join 模板"""
        print("Parsing workload...")
        
        # 临时存储用于去重: Key=(aliases_tuple, join_sig) -> Value=sub_graph
        temp_templates = {}

        try:
            template_dirs = sorted([d for d in os.listdir(self.base_query_dir) 
                                    if os.path.isdir(os.path.join(self.base_query_dir, d))])
        except FileNotFoundError:
            print(f"Error: Directory not found {self.base_query_dir}")
            return

        for template_name in template_dirs:
            if self.skip_7a and template_name == "7a":
                continue
            
            dir_path = os.path.join(self.base_query_dir, template_name)
            pkl_files = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
            
            for pkl_file in pkl_files:
                try:
                    qrep = load_qrep(pkl_file)
                except Exception as e:
                    print(f"Error loading {pkl_file}: {e}")
                    continue

                join_graph = qrep["join_graph"]
                subset_graph = qrep["subset_graph"]

                for subplan_tuple in sorted(subset_graph.nodes()):
                    if len(subplan_tuple) < 2:
                        continue # 跳过单表

                    # if "ci" in subplan_tuple or "mi1" in subplan_tuple or "mi2" in subplan_tuple:
                    #     continue # 跳过大表

                    all_small = True
                    for alias in subplan_tuple:
                        real_name = join_graph.nodes[alias]['real_name']
                        card = TABLE_CARD.get(real_name, float("inf"))
                        if card >= 3000000:
                            all_small = False
                            break
                    if not all_small:
                        continue

                    sorted_aliases = sorted(list(subplan_tuple))
                    sub_graph = join_graph.subgraph(subplan_tuple)
                    
                    # 生成 Join 签名
                    edges_info = []
                    for u, v, data in sub_graph.edges(data=True):
                        if u > v: u, v = v, u
                        cond = data.get("join_condition", "")
                        if not cond:
                            print(f"Warning: No join condition found between {u} and {v} in {pkl_file}")
                            continue

                        cond_clean = normalize_condition(cond)
                        edges_info.append(f"{u}|{v}|{cond_clean}")

                    edges_info.sort()
                    join_sig = "||".join(edges_info)
                    
                    template_key = (tuple(sorted_aliases), join_sig)
                    
                    if template_key not in temp_templates:
                        temp_templates[template_key] = sub_graph.copy()

        for (aliases_tuple, join_sig), sub_graph in temp_templates.items():
            import hashlib
            sig_hash = hashlib.md5(join_sig.encode('utf-8')).hexdigest()[:6]
            template_id = f"{'_'.join(aliases_tuple)}_{sig_hash}"
            
            real_names = {alias: sub_graph.nodes[alias]["real_name"] for alias in aliases_tuple}
            
            self.join_templates[template_id] = {
                "aliases": list(aliases_tuple),
                "real_names": real_names,
                "join_graph": sub_graph
            }

        self.join_templates = dict(sorted(self.join_templates.items(), key=lambda x: len(x[1]['aliases'])))
        print(f"Parsing complete. Found {len(self.join_templates)} distinct templates.")

    def generate_clean_sql(self, template_id, data):
        """
        生成 SQL：
        1. 复制图并去除所有谓词 (WHERE条件)。
        2. 使用 BFS 构建合法的 JOIN 顺序。
        3. 包裹采样语句。
        """
        aliases = data['aliases']
        real_names = data['real_names']
        original_graph = data['join_graph']
        
        # 去除谓词
        graph = original_graph.copy()
        for node in graph.nodes():
            if 'predicates' in graph.nodes[node]:
                graph.nodes[node]['predicates'] = []
        
        root = sorted(aliases)[0]
        root_real = real_names[root]
        
        from_clause = f"FROM {root_real} AS {root}"
        join_clauses = []
        
        try:
            edges = list(nx.bfs_edges(graph, source=root))
        except:
            edges = graph.edges()

        visited = {root}
        
        for u, v in edges:
            if v in visited: continue

            edge_data = graph.get_edge_data(u, v)
            if not edge_data: edge_data = graph.get_edge_data(v, u)
            
            cond = edge_data.get("join_condition")
            if not cond: return None
            
            target = v
            target_real = real_names[target]
            
            join_clauses.append(f"JOIN {target_real} AS {target} ON {cond}")
            visited.add(target)

        if len(visited) < len(aliases):
            print(f"Warning: Incomplete join for template {template_id}")
            
        base_joins = f"{from_clause} {' '.join(join_clauses)}"

        select_cols = ", ".join([f'{alias}.id AS "{alias}"' for alias in aliases])
        
        sql = f"""
            SELECT {select_cols}
            {base_joins}
        """
        return sql

    def execute_sampling(self):
        """执行 SQL 并获取结果"""
        if not self.join_templates:
            return {}

        conn = None
        print("\nStarting DB Sampling in batches of 100...")

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            current_batch_samples = {}
            batch_count = 0
            
            total = len(self.join_templates)
            for i, (tid, data) in enumerate(self.join_templates.items()):
                print(f"[{i+1}/{total}] {tid} ... ", end="", flush=True)
                
                sql = self.generate_clean_sql(tid, data)
                if not sql:
                    print("    Error generating SQL")
                    continue
                
                start_t = time.time()
                try:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]

                    current_samples = [dict(zip(cols, row)) for row in rows]
                    
                    if current_samples:
                        current_batch_samples[tid] = [current_samples]
                        print(f"    Done ({len(current_samples)} rows, {time.time()-start_t:.2f}s)")
                    else:
                        print("    Empty Result")
                        
                except psycopg2.errors.QueryCanceled:
                    conn.rollback()
                    print("TIMEOUT (>5min)")
                except Exception as e:
                    conn.rollback()
                    print(f"Error: {e}")

                if (i + 1) % 2 == 0 or (i + 1) == total:
                    if current_batch_samples:
                        batch_count += 1
                        self.save_samples(current_batch_samples, batch_count)
                        current_batch_samples = {}
                    
        except Exception as e:
            print(f"DB Connection Error: {e}")
        finally:
            if conn: conn.close()


    def save_samples(self, samples_dict, batch_idx):
        """保存为指定格式的 JSON"""
        if not samples_dict:
            print("No samples to save.")
            return

        file_name = f"samples_batch_{batch_idx}.json"
        full_path = os.path.join(self.output_dir, file_name)
        
        print(f" --> Saving batch {batch_idx} to {file_name}...")

        formatted_output = {}

        for template_id, bitmaps in samples_dict.items():
            formatted_bitmaps = []
            for bitmap in bitmaps: # 这里只有一个 bitmap
                if not bitmap:
                    formatted_bitmaps.append([])
                    continue
                
                # 提取 Header
                aliases = sorted(bitmap[0].keys())
                header = [f"{alias}.id" for alias in aliases]
                
                # 提取 Rows
                rows = [header]
                for row_dict in bitmap:
                    rows.append([row_dict[a] for a in aliases])
                
                formatted_bitmaps.append(rows)
            
            formatted_output[template_id] = formatted_bitmaps

        with open(full_path, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        print("Save complete.")

if __name__ == "__main__":
    conf = Config()
    sampler = SimpleWorkloadSampler(conf)

    sampler.load_and_parse_workload()

    samples = sampler.execute_sampling()

    print("\nAll batches processed.")