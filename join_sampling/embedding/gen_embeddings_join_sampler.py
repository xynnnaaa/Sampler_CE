import os
import json
import pickle
import glob
import re
import hashlib
import networkx as nx
import pandas as pd
import duckdb
import psycopg2
import torch  # 假设使用 PyTorch
from collections import defaultdict

class Config:
    # 路径配置
    BASE_QUERY_DIR = "/data1/xuyining/CEB-default/queries/ceb-imdb"      # 存放 QRep .pkl 文件的根目录 (按 template 分文件夹)
    SAMPLE_OUTPUT_DIR = "/data1/xuyining/Sampler/join_sampling/sample_results/small_skip7a_tree/smallest_first"   # 存放采样结果 json 文件的目录
    EMBEDDING_OUTPUT_DIR = "/data1/xuyining/Sampler/join_sampling/embedding/results/small_skip7a" # 存放生成的 embedding 结果
    
    # 数据库配置
    DB_CONFIG = {
        "host": "localhost",
        "port": "5433",
        "database": "imdb_new",
        "user": "xuyining"
    }

    # 其他配置
    SKIP_TEMPLATE_7A = True # 是否跳过 7a


def generate_table_embedding(df):
    if df.empty:
        return torch.zeros(128) 

    id_cols = [c for c in df.columns if c.endswith('_id')]

    return None


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

def get_template_id(sorted_aliases, join_graph):
    """
    根据别名和 Join Graph 生成唯一的 Template ID。
    逻辑必须与 save_single_bitmap 中的生成逻辑完全一致。
    """
    sub_graph = join_graph.subgraph(sorted_aliases)
    edges_info = []

    for u, v, data in sub_graph.edges(data=True):
        if u > v:
            u, v = v, u
        cond = data.get("join_condition", "")
        if not cond:
            continue
        cond_clean = normalize_condition(cond)
        edges_info.append(f"{u}|{v}|{cond_clean}")

    edges_info.sort()
    join_sig_str = "||".join(edges_info)
    
    sig_hash = hashlib.md5(join_sig_str.encode('utf-8')).hexdigest()[:6]

    template_id = f"{'_'.join(sorted_aliases)}_{sig_hash}"
    return template_id

def transform_predicate_for_duckdb(predicate_str, alias):
    """
    将谓词中的列名从 "alias.col" 转换为 "alias_col"，以适应 DuckDB 的列命名规则。
    """
    escaped_alias = re.escape(alias)
    pattern = rf"(?<![\w\.]){escaped_alias}\.(\w+)"
    return re.sub(pattern, rf"{alias}_\1", predicate_str)


class SampleHydrator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.duck_con = duckdb.connect(database=':memory:')
        self.full_sample_cache = {} # template_id -> True

    def _get_pg_connection(self):
        return psycopg2.connect(**self.db_config)

    def fetch_and_register_samples(self, template_id, join_graph, formatted_rows):
        """
        从数据库 Fetch 完整行并注册到 DuckDB
        """
        view_name = f"sample_{template_id}"
        if template_id in self.full_sample_cache:
            return view_name
        
        if not formatted_rows or len(formatted_rows) < 2:
            return None

        header = formatted_rows[0] # e.g. ["t.id", "mc.id"]
        data = formatted_rows[1:]
        
        base_df = pd.DataFrame(data, columns=header)
        final_df = base_df
        
        aliases = [col.split('.')[0] for col in header]
        
        conn = self._get_pg_connection()
        try:
            for i, alias in enumerate(aliases):
                id_col_name = header[i]
                unique_ids = base_df[id_col_name].unique()
                if len(unique_ids) == 0:
                    continue

                real_table_name = join_graph.nodes[alias]["real_name"]
                
                ids_str = ",".join(map(str, unique_ids))
                sql = f"SELECT * FROM {real_table_name} WHERE id IN ({ids_str})"
                
                partial_df = pd.read_sql(sql, conn)
                
                # 重命名: col -> alias_col
                rename_map = {col: f"{alias}_{col}" for col in partial_df.columns}
                partial_df.rename(columns=rename_map, inplace=True)
                
                join_key_in_base = id_col_name
                join_key_in_partial = f"{alias}_id"
                
                final_df = final_df.merge(
                    partial_df,
                    left_on=join_key_in_base,
                    right_on=join_key_in_partial,
                    how='inner'
                )
                del partial_df
        except Exception as e:
            print(f"Error hydrating {template_id}: {e}")
            return None
        finally:
            conn.close()

        self.full_sample_cache[template_id] = True

        # 注册 View
        self.duck_con.execute(f"DROP VIEW IF EXISTS {view_name}")
        self.duck_con.register(view_name, final_df)
        
        return view_name

    def query_sample(self, view_name, where_clause):
        if not where_clause:
            query = f"SELECT * FROM {view_name}"
        else:
            query = f"SELECT * FROM {view_name} WHERE {where_clause}"
        
        try:
            return self.duck_con.execute(query).fetchdf()
        except Exception as e:
            print(f"DuckDB Query Error: {e} | Query: {query}")
            return pd.DataFrame()

class SampleManager:
    """
    负责扫描目录加载所有的 json 样本，并提供按 TemplateID 查询的功能
    """
    def __init__(self, sample_dir):
        self.sample_dir = sample_dir
        self.sample_cache = {} # template_id -> formatted_rows
        self._load_all_samples()

    def _load_all_samples(self):
        print(f"Loading samples from {self.sample_dir}...")
        json_files = glob.glob(os.path.join(self.sample_dir, "*.json"))
        count = 0
        for jf in json_files:
            # 只收集“_1”结尾的文件
            if not jf.endswith("_1.json"):
                continue
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    # data 的结构是 { template_id: [rows] }
                    for tid, rows in data.items():
                        self.sample_cache[tid] = rows
                        count += 1
            except Exception as e:
                print(f"Error loading {jf}: {e}")
        print(f"Loaded {count} sample bitmaps into memory.")

    def get_sample(self, template_id):
        return self.sample_cache.get(template_id)

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.hydrator = SampleHydrator(config.DB_CONFIG)
        self.sample_manager = SampleManager(config.SAMPLE_OUTPUT_DIR)
        
        if not os.path.exists(config.EMBEDDING_OUTPUT_DIR):
            os.makedirs(config.EMBEDDING_OUTPUT_DIR)

    def extract_filter_predicates(self, sub_graph, sorted_aliases):
        """
        从子图提取 WHERE 条件。
        注意：**只提取 Filter Predicates** (如 t.kind_id=1)，
        **不** 提取 Join Conditions (如 t.id=mc.movie_id)，因为 Sample 已经是 Join 过的了。
        """
        conds = []

        for alias in sorted_aliases:
            if alias not in sub_graph.nodes: 
                continue
            node_data = sub_graph.nodes[alias]
            preds = node_data.get("predicates", [])
            for pred in preds:
                # 转换: t.kind_id = 1  -->  t_kind_id = 1
                clean_pred = transform_predicate_for_duckdb(pred, alias)
                conds.append(clean_pred)

        return " AND ".join(conds) if conds else None

    def process_all(self):
        template_dirs = sorted([d for d in os.listdir(self.config.BASE_QUERY_DIR) 
                                if os.path.isdir(os.path.join(self.config.BASE_QUERY_DIR, d))])
        
        for tmpl_dir_name in template_dirs:
            if self.config.SKIP_TEMPLATE_7A and tmpl_dir_name == "7a":
                continue
                
            input_dir = os.path.join(self.config.BASE_QUERY_DIR, tmpl_dir_name)
            output_dir = os.path.join(self.config.EMBEDDING_OUTPUT_DIR, tmpl_dir_name)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
            
            print(f"Processing template dir: {tmpl_dir_name}, found {len(pkl_files)} queries.")
            
            for pkl_path in pkl_files:
                self.process_single_qrep(pkl_path, output_dir)

    def process_single_qrep(self, pkl_path, output_dir):
        try:
            with open(pkl_path, 'rb') as f:
                qrep = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {pkl_path}: {e}")
            return

        join_graph = qrep["join_graph"]
        subset_graph = qrep["subset_graph"]
        
        # 结果字典: { sorted_aliases_tuple: embedding_tensor }
        query_embeddings = {}

        for subplan_tuple in sorted(subset_graph.nodes()):
            if len(subplan_tuple) < 2:
                continue

            sorted_aliases = sorted(list(subplan_tuple))

            template_id = get_template_id(sorted_aliases, join_graph)

            formatted_rows = self.sample_manager.get_sample(template_id)
            
            if not formatted_rows:
                query_embeddings[tuple(sorted_aliases)] = torch.zeros(128) 
                continue

            view_name = self.hydrator.fetch_and_register_samples(template_id, join_graph, formatted_rows)
            
            if not view_name:
                 query_embeddings[tuple(sorted_aliases)] = torch.zeros(128)
                 continue

            sub_graph = join_graph.subgraph(subplan_tuple)
            where_clause = self.extract_filter_predicates(sub_graph, sorted_aliases)

            filtered_df = self.hydrator.query_sample(view_name, where_clause)

            emb = generate_table_embedding(filtered_df)

            query_embeddings[tuple(sorted_aliases)] = emb

        file_name = os.path.basename(pkl_path) # e.g., query_0.pkl
        save_path = os.path.join(output_dir, file_name)
        
        with open(save_path, 'wb') as f:
            pickle.dump(query_embeddings, f)
            
        print(f"Saved embeddings for {file_name}")


if __name__ == "__main__":
    conf = Config()
    
    print("Starting Embedding Generation Process...")
    generator = EmbeddingGenerator(conf)
    generator.process_all()
    print("All Done.")