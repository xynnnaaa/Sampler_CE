import os
import json
import re
import hashlib
import sys
import time
import networkx as nx

import pandas as pd
import datetime
import duckdb
import psycopg2

import warnings
warnings.filterwarnings('ignore', 
                       message='pandas only support SQLAlchemy connectable',
                       module='pandas.io.sql')

parser_module_dir = "/home/Sampler_CE/mscn"
if parser_module_dir not in sys.path:
    sys.path.insert(0, parser_module_dir)

# ================= 配置区域 =================
class Config:
    # SQL 工作负载文件路径
    TRAIN_SQL_FILE = "/home/PRICE/datas/workloads/finetune/imdb/workloads.sql" # 替换为实际训练集路径
    TEST_SQL_FILE = "/home/PRICE/datas/workloads/test/imdb/workloads.sql"   # 替换为实际测试集路径
    
    # 样本文件夹路径 (由训练集生成的样本)
    SAMPLE_JSON_FILE = "/home/Sampler_CE/join_sampling/workloads/imdb/v1/results_padded" 

    # 数据库配置 (Postgres - 用于 Hydration 和 通用元组探测)
    DB_CONFIG = {
        "host": "localhost",
        "port": "5432",
        "database": "ergastf1", 
        "user": "xuyining",
        "password": "123"
    }

# ================= 1. SQL Parser (复用) =================

def normalize_condition(cond_str):
    if not cond_str:
        return "None"
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
    where_part = sql_str[where_idx:]       
    
    where_part = where_part.strip()
    has_semicolon = where_part.endswith(";")
    if has_semicolon:
        where_part = where_part[:-1].strip()
        
    conditions = re.split(r'\s+AND\s+', where_part, flags=re.IGNORECASE)
    
    clean_conditions =[]
    for cond in conditions:
        cond = cond.strip()
        while cond.startswith("(") and cond.endswith(")"):
            cond = cond[1:-1].strip()
        clean_conditions.append(cond)
        
    new_where = " AND ".join(clean_conditions)
    return select_from_part + " " + new_where

from query_representation.utils import extract_join_graph

class WorkloadParser:
    def __init__(self):
        pass

    def parse(self, sql_file, is_train=False):
        queries =[] 
        try:
            with open(sql_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error loading {sql_file}: {e}")
            return queries

        valid_query_idx = 0 
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            parts = line.split("||")
            sql_str = parts[0].strip()

            if sql_str.startswith("/*"):
                sql_str = sql_str.split("*/", 1)[-1].strip()

            if "SELECT" not in sql_str.upper(): continue

            try:
                sql_str = remove_unnecessary_parentheses(sql_str)
                join_graph = extract_join_graph(sql_str)
            except Exception as e:
                print(f"Error parsing SQL at line {line_idx}: {e}")
                continue

            sorted_aliases = sorted(list(join_graph.nodes()))
            if len(sorted_aliases) < 2: continue

            # 生成 Template ID
            col_graph = nx.Graph()
            for u, v, data in join_graph.edges(data=True):
                cond = data.get("join_condition", "")
                if not cond: continue
                cond_clean = normalize_condition(cond)
                parts = cond_clean.split("=")
                if len(parts) == 2:
                    col_graph.add_edge(parts[0].strip(), parts[1].strip())

            eq_classes = list(nx.connected_components(col_graph))
            class_signatures =[]
            for eq_class in eq_classes:
                sorted_cols = sorted(list(eq_class))
                class_signatures.append("=".join(sorted_cols))
            class_signatures.sort()
            join_sig_str = "||".join(class_signatures)

            sig_hash = hashlib.md5(join_sig_str.encode('utf-8')).hexdigest()[:6]
            template_id = f"{'_'.join(sorted_aliases)}_{sig_hash}"

            # 生成 DuckDB where_clause (用于本地命中测试)
            duckdb_conds =[]
            for alias in sorted_aliases:
                preds_list = join_graph.nodes[alias].get("predicates",[])
                for pred in preds_list:
                    if "stats" in Config.DB_CONFIG.get("database", ""):
                        pred = convert_timestamps_in_sql(pred) 
                    pattern = fr"\b{alias}\."
                    pred_duckdb = re.sub(pattern, f"{alias}__", pred).strip()
                    pred_duckdb = re.sub(r'\s*([<>=!]+)\s*', r'\1', pred_duckdb)
                    duckdb_conds.append(pred_duckdb)

            where_clause = " AND ".join(duckdb_conds) if duckdb_conds else None

            queries.append({
                'q_idx': valid_query_idx, 
                'sql': sql_str,
                'template_id': template_id,
                'aliases': sorted_aliases,
                'join_graph': join_graph,
                'where_clause': where_clause
            })
            valid_query_idx += 1

        label = "Train" if is_train else "Test"
        print(f"Parsed {len(queries)} valid join queries from {label} set.")
        return queries

# ================= 2. 样本与数据层 (复用) =================

class SampleManager:
    def __init__(self, sample_file):
        self.sample_cache = {}
        self._load_samples(sample_file)

    def _load_samples(self, sample_file):
        print(f"Loading merged JSON samples from {sample_file}...")
        if os.path.isdir(sample_file):
            for root, dirs, files in os.walk(sample_file):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                for template_id, samples in data.items():
                                    if template_id not in self.sample_cache:
                                        self.sample_cache[template_id] = []
                                    self.sample_cache[template_id].extend(samples)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        else:
            print(f"{sample_file} is not a directory.")

    def get_sample_list(self, template_id):
        return self.sample_cache.get(template_id,[])

class SampleHydrator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.duck_con = duckdb.connect(database=':memory:')
        self.registered_views = {} 

    def _get_pg_connection(self):
        return psycopg2.connect(**self.db_config)

    def fetch_and_register_samples(self, template_id, join_graph, formatted_samples_list):
        if template_id in self.registered_views:
            return self.registered_views[template_id]
        
        if not formatted_samples_list:
            return[]

        conn = self._get_pg_connection()
        view_names =[]
        try:
            for sample_idx, formatted_rows in enumerate(formatted_samples_list):
                if not formatted_rows or len(formatted_rows) < 2: continue

                header = formatted_rows[0]
                raw_data = formatted_rows[1:]
                expected_cols = len(header)
                
                clean_data =[row for row in raw_data if isinstance(row, list) and len(row) == expected_cols]
                if not clean_data: continue

                base_df = pd.DataFrame(clean_data, columns=header)
                base_df[header] = base_df[header].apply(pd.to_numeric, errors='coerce')

                final_df = base_df
                aliases = [col.split('.')[0] for col in header]
                
                for i, alias in enumerate(aliases):
                    id_col_name = header[i]
                    unique_ids = base_df[id_col_name].dropna().unique()
                    if len(unique_ids) == 0: continue
                    
                    real_table_name = join_graph.nodes[alias]["real_name"]
                    ids_str = ",".join(map(str, unique_ids))
                    
                    sql = f"SELECT * FROM {real_table_name} WHERE id IN ({ids_str})"
                    partial_df = pd.read_sql(sql, conn)
                    partial_df.rename(columns={col: f"{alias}__{col}" for col in partial_df.columns}, inplace=True)
                    
                    final_df = final_df.merge(partial_df, left_on=id_col_name, right_on=f"{alias}__id", how='inner')
                    del partial_df

                view_name = f"sample_{template_id}_{sample_idx}"
                self.duck_con.execute(f"DROP VIEW IF EXISTS {view_name}")
                self.duck_con.register(view_name, final_df)
                view_names.append(view_name)
                
        except Exception as e:
            print(f"Hydration Error for {template_id}: {e}")
        finally:
            conn.close()

        self.registered_views[template_id] = view_names
        return view_names

    def query_sample(self, view_name, where_clause):
        query = f"SELECT 1 FROM {view_name}" # 只需要探测是否有命中，不用 select *
        if where_clause:
            query += f" WHERE {where_clause}"
        try:
            return self.duck_con.execute(query).fetchdf()
        except Exception as e:
            return pd.DataFrame()


# ================= 3. 分析控制器 =================

class CoverageAnalyzer:
    def __init__(self, config):
        self.config = config
        self.parser = WorkloadParser()
        self.sample_manager = SampleManager(config.SAMPLE_JSON_FILE)
        self.hydrator = SampleHydrator(config.DB_CONFIG)
        
        # 维护一个常驻的 PG 连接用于探测 Universal Tuple
        self.pg_conn = psycopg2.connect(**config.DB_CONFIG)
        self.pg_cursor = self.pg_conn.cursor()

    def build_intersection_sql(self, test_query, train_queries):
        """
        [核心逻辑] 构造在数据库中探测通用元组的 SQL。
        查询满足 (Test Query Predicates) AND (Train_1_Preds OR Train_2_Preds ...) 的元组是否存在。
        """
        join_graph = test_query['join_graph']
        aliases = test_query['aliases']
        
        # 1. 构造 FROM clause (e.g., "title AS t, movie_info AS mi")
        from_parts = [f"{join_graph.nodes[a]['real_name']} AS {a}" for a in aliases]
        from_clause = ", ".join(from_parts)
        
        # 2. 构造 JOIN 关系
        join_conds =[]
        for u, v, data in join_graph.edges(data=True):
            if 'join_condition' in data and data['join_condition']:
                join_conds.append(data['join_condition'])
        join_clause = " AND ".join(join_conds) if join_conds else "TRUE"
        
        # 3. 构造 Test Query 过滤条件
        test_filters =[]
        for a in aliases:
            test_filters.extend(join_graph.nodes[a].get("predicates",[]))
        test_filter_clause = " AND ".join(test_filters) if test_filters else "TRUE"
        
        # 4. 构造 Train Queries 过滤条件的 OR 集合
        train_or_parts =[]
        for tr_q in train_queries:
            tr_filters =[]
            tr_join_graph = tr_q['join_graph']
            for a in aliases:
                tr_filters.extend(tr_join_graph.nodes[a].get("predicates",[]))
            tr_filter_clause = " AND ".join(tr_filters) if tr_filters else "TRUE"
            train_or_parts.append(f"({tr_filter_clause})")
            
        train_or_clause = " OR ".join(train_or_parts) if train_or_parts else "FALSE"
        
        # 组合终极 SQL (LIMIT 1 短路求值保证极致性能)
        sql = f"""
            SELECT 1 
            FROM {from_clause} 
            WHERE ({join_clause}) 
              AND ({test_filter_clause}) 
              AND ({train_or_clause}) 
            LIMIT 1;
        """
        return sql

    def analyze(self):
        print("\n=== Stage 1: Parsing Workloads ===")
        train_queries = self.parser.parse(self.config.TRAIN_SQL_FILE, is_train=True)
        test_queries = self.parser.parse(self.config.TEST_SQL_FILE, is_train=False)
        
        if not train_queries or not test_queries:
            print("Missing queries. Exiting.")
            return

        # 整理训练集 Templates
        train_templates = set()
        train_queries_by_template = {}
        for q in train_queries:
            tid = q['template_id']
            train_templates.add(tid)
            if tid not in train_queries_by_template:
                train_queries_by_template[tid] = []
            train_queries_by_template[tid].append(q)

        # 统计容器
        hit_queries = []
        cat1_unseen_template =[]  # Template 在训练集未出现
        cat2_missed_by_sample =[] # Template 出现过，但没被样本命中

        print("\n=== Stage 2: Evaluating Test Queries on Samples ===")
        for i, tq in enumerate(test_queries):
            tid = tq['template_id']
            if i % 100 == 0:
                print(f"  Evaluating query {i}/{len(test_queries)}...")
                
            if tid not in train_templates:
                cat1_unseen_template.append(tq)
            else:
                formatted_samples_list = self.sample_manager.get_sample_list(tid)
                view_names = self.hydrator.fetch_and_register_samples(tid, tq['join_graph'], formatted_samples_list)
                
                is_hit = False
                if view_names:
                    for view_name in view_names:
                        filtered_df = self.hydrator.query_sample(view_name, tq['where_clause'])
                        if not filtered_df.empty:
                            is_hit = True
                            break 
                
                if is_hit:
                    hit_queries.append(tq)
                else:
                    cat2_missed_by_sample.append(tq)

        print("\n=== Stage 3: Checking Universal Tuples for Missed Queries ===")
        universal_hits = []
        no_universal_hits =[]
        
        for i, tq in enumerate(cat2_missed_by_sample):
            if i % 10 == 0:
                print(f"  Checking DB for missed query {i}/{len(cat2_missed_by_sample)}...")
            
            tid = tq['template_id']
            corresponding_train_qs = train_queries_by_template[tid]
            
            check_sql = self.build_intersection_sql(tq, corresponding_train_qs)
            
            try:
                self.pg_cursor.execute(check_sql)
                result = self.pg_cursor.fetchone()
                if result:
                    universal_hits.append(tq)
                else:
                    no_universal_hits.append(tq)
            except Exception as e:
                print(f"  [DB Error] Failed to check universal tuple for query {tq['q_idx']}: {e}")
                self.pg_conn.rollback() # 防止事务卡死
                no_universal_hits.append(tq)

        # === 打印最终报告 ===
        total_test = len(test_queries)
        total_missed = len(cat1_unseen_template) + len(cat2_missed_by_sample)
        
        print("\n=======================================================")
        print("                 COVERAGE ANALYSIS REPORT                ")
        print("=======================================================")
        print(f"Total Test Queries:        {total_test}")
        print(f"Hit by Generated Samples:  {len(hit_queries)} ({(len(hit_queries)/total_test)*100:.2f}%)")
        print(f"Missed by Samples:         {total_missed} ({(total_missed/total_test)*100:.2f}%)")
        print("-------------------------------------------------------")
        print("Breakdown of Missed Queries:")
        print(f"  1. OOD: Join Template NOT in Train Set:")
        print(f"     Count: {len(cat1_unseen_template)}")
        print(f"     Ratio (of missed): {(len(cat1_unseen_template)/max(1, total_missed))*100:.2f}%")
        print(f"     Ratio (of total test): {(len(cat1_unseen_template)/total_test)*100:.2f}%")
        print("")
        print(f"  2. Distribution Shift: Template IN Train Set, but Missed by Sample:")
        print(f"     Count: {len(cat2_missed_by_sample)}")
        print(f"     Ratio (of missed): {(len(cat2_missed_by_sample)/max(1, total_missed))*100:.2f}%")
        print(f"     Ratio (of total test): {(len(cat2_missed_by_sample)/total_test)*100:.2f}%")
        print("")
        print("     --- Deep Dive into Category 2 (Teacher's Hypothesis) ---")
        if len(cat2_missed_by_sample) > 0:
            print(f"     -> Universal Tuples EXIST (Can be fixed by better sampling strategy):")
            print(f"        Count: {len(universal_hits)}")
            print(f"        Ratio (of Cat 2): {(len(universal_hits)/len(cat2_missed_by_sample))*100:.2f}%")
            print("")
            print(f"     -> NO Universal Tuples (Hard isolation between train and test distribution):")
            print(f"        Count: {len(no_universal_hits)}")
            print(f"        Ratio (of Cat 2): {(len(no_universal_hits)/len(cat2_missed_by_sample))*100:.2f}%")
        print("=======================================================\n")
        
        self.pg_cursor.close()
        self.pg_conn.close()

        # 把未覆盖的查询写到uncovered_test.sql文件中，方便后续分析
        uncovered_file = "uncovered_test.sql"
        with open(uncovered_file, 'w') as f:
            for tq in cat1_unseen_template + cat2_missed_by_sample:
                f.write(tq['sql'] + ";\n")

if __name__ == "__main__":
    start_time = time.time()
    
    # 初始化并运行分析
    analyzer = CoverageAnalyzer(Config)
    analyzer.analyze()

    print(f"All tasks finished in {time.time() - start_time:.3f}s")