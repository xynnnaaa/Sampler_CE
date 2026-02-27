import json
import psycopg2
import os
import sys
import glob
import pickle
import time
from typing import List, Dict, Any, Tuple
from sqlglot import parse_one, exp
from collections import defaultdict, Counter
import random
from networkx.readwrite import json_graph
import networkx as nx

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

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

class JoinSampler:
    def __init__(self, config_path: str, join_results_path: str):
        print("Initializing JoinSampler...")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.join_results_path = join_results_path
        self.base_query_dir = self.config["sampling"]["base_query_dir"]
        self.k_bitmaps = self.config["sampling"]["k_bitmaps"]
        self.m_partitions = self.config["sampling"]["m_partitions"]
        
        # 结果存储
        # self.samples: Dict[str, List[List[Any]]] = defaultdict(list)
        self.query_counter = 0

        print("Connecting to database...")
        self.conn = psycopg2.connect(
            dbname=self.config["db"]["dbname"],
            user=self.config["db"]["user"],
            host=self.config["db"]["host"],
            port=self.config["db"]["port"],
        )
        self.cursor = self.conn.cursor()

    def normalize_condition(self, cond_str):
        """
        标准化连接条件字符串，确保 'a=b' 和 'b=a' 被视为相同。
        去掉空格并按字典序排列等号两边。
        """
        if not cond_str:
            return "None"
        parts = [p.strip() for p in cond_str.split('=')]
        parts.sort()
        return "=".join(parts)

    def load_join_results(self) -> Dict[str, Dict]:
        """
        加载预先计算好的完整连接结果 (JSON)。
        格式预期: { "template_id": [ [header...], [row1...], ... ] }
        如果是分批的文件，这里只演示加载单个合并后的文件，或者是目录下的所有json。
        """
        print(f"Loading join results from {self.join_results_path}...")
        join_data = {}
        
        # 支持传入目录或单个文件
        if os.path.isdir(self.join_results_path):
            files = glob.glob(os.path.join(self.join_results_path, "*.json"))
        else:
            files = [self.join_results_path]

        total_rows = 0
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    for tid, content in data.items():
                        # 处理可能的嵌套格式
                        rows_list = content
                        if not rows_list: continue
                        
                        # 兼容性处理：如果是 [[header, row...]] 这种多套一层的
                        if isinstance(rows_list[0][0], list) or isinstance(rows_list[0][0], str) is False:
                             rows_list = rows_list[0]

                        join_data[tid] = {
                            "header": rows_list[0],
                            "rows": rows_list[1:]
                        }
                        total_rows += len(rows_list) - 1
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        
        print(f"Loaded join results for {len(join_data)} templates (Total {total_rows} rows).")
        return join_data

    def parse_workload_by_template(self) -> Dict[str, Dict]:
        """
        遍历每个查询的 subset_graph，按子查询（Sub-plan）粒度聚合谓词。
        """
        print("Parsing workload by Sub-plan Templates...")
        import networkx as nx
        
        # 结果存储结构：
        # { 
        #   template_id: {
        #       "alias_map": {alias: real_name},
        #       "queries": [ 
        #           {"file_name": "1a.pkl", "predicate_sql": "t.kind_id=1 AND ..."}, 
        #           ... 
        #       ]
        #   }
        # }
        parsed_templates = defaultdict(lambda: {"alias_map": {}, "queries": []})
        
        # 统计计数
        total_subplans = 0
        
        try:
            template_dirs = [d for d in os.listdir(self.base_query_dir) 
                             if os.path.isdir(os.path.join(self.base_query_dir, d))]
        except FileNotFoundError:
            print(f"Base query dir not found: {self.base_query_dir}")
            return {}

        for template_dir_name in sorted(template_dirs):
            if template_dir_name == "7a": 
                continue 

            dir_path = os.path.join(self.base_query_dir, template_dir_name)
            pkl_files = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
            
            for pkl_file in pkl_files:
                try:
                    qrep = load_qrep(pkl_file)
                    
                    join_graph = qrep["join_graph"]
                    subset_graph = qrep["subset_graph"]
                    
                    for subplan_tuple in subset_graph.nodes():
                        if len(subplan_tuple) < 2:
                            continue

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
                        edges_info = []

                        for u, v, data in sub_graph.edges(data=True):
                            if u > v: u, v = v, u
                            cond = data.get("join_condition", "")
                            if not cond: continue
                            cond_clean = self.normalize_condition(cond)
                            edges_info.append(f"{u}|{v}|{cond_clean}")

                        edges_info.sort()
                        join_sig_str = "||".join(edges_info)

                        # 3. 生成 Template ID
                        import hashlib
                        sig_hash = hashlib.md5(join_sig_str.encode('utf-8')).hexdigest()[:6]
                        tid = f"{'_'.join(sorted_aliases)}_{sig_hash}"
                        
                        # 4. 提取当前子查询对应的谓词和别名映射
                        current_predicates = []
                        current_alias_map = {}
                        
                        for alias in sorted_aliases:
                            node_data = join_graph.nodes[alias]
                            real_name = node_data["real_name"]
                            current_alias_map[alias] = real_name
                            
                            # 提取谓词
                            preds = node_data.get("predicates", [])
                            if isinstance(preds, str): preds = [preds]
                            for p in preds:
                                if p and p.strip():
                                    current_predicates.append(f"({p})")

                        full_predicate_sql = " AND ".join(current_predicates) if current_predicates else "1=1"
                        
                        # 5. 存入字典
                        # 如果是第一次遇到这个 Template ID，初始化 alias_map
                        if not parsed_templates[tid]["alias_map"]:
                            parsed_templates[tid]["alias_map"] = current_alias_map
                        
                        # 将当前查询实例添加到该 Template 下
                        existing_preds = {q["predicate_sql"] for q in parsed_templates[tid]["queries"]}
                        if full_predicate_sql not in existing_preds:
                            parsed_templates[tid]["queries"].append({
                                "file_name": os.path.basename(pkl_file),
                                "predicate_sql": full_predicate_sql
                            })
                            total_subplans += 1

                except Exception as e:
                    print(f"Error parsing {pkl_file}: {e}")

        print(f"Parsed workload: Found {len(parsed_templates)} unique templates covering {total_subplans} sub-query instances.")
        return parsed_templates

    def setup_temp_table(self, tid: str, header: List[str], rows: List[List[Any]]) -> str:
        """
        在数据库中创建一个临时表来存储该 Template 的所有 Join 结果 ID。
        返回: 临时表名
        """
        temp_table_name = f"temp_ids_{tid.replace('-', '_')}"
        # 防止 SQL 注入，清理表名
        temp_table_name = "".join(c for c in temp_table_name if c.isalnum() or c == '_')

        # 1. 构建建表语句
        # header 格式如 ["t.id", "mi.id"] -> 列名变成 "t_id", "mi_id"
        cols_def = ["virtual_id SERIAL PRIMARY KEY"]
        col_mapping = {} # "t.id" -> "t_id"
        
        for col in header:
            safe_col = col.replace(".", "_") # t.id -> t_id
            cols_def.append(f"{safe_col} INTEGER")
            col_mapping[col] = safe_col
        
        create_sql = f"CREATE TEMPORARY TABLE IF NOT EXISTS {temp_table_name} ({', '.join(cols_def)}) ON COMMIT DROP;"
        
        try:
            # 先 Drop 防止残留
            self.cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
            self.cursor.execute(create_sql)
            
            # 2. 批量插入数据
            insert_cols = [col_mapping[c] for c in header]
            placeholders = ",".join(["%s"] * len(insert_cols))
            insert_sql = f"INSERT INTO {temp_table_name} ({','.join(insert_cols)}) VALUES ({placeholders})"
            
            # 批量执行
            from psycopg2.extras import execute_batch
            execute_batch(self.cursor, insert_sql, rows, page_size=1000)
            
            # 创建索引加速
            self.cursor.execute(f"CREATE INDEX ON {temp_table_name} (virtual_id)")
            
            return temp_table_name, col_mapping

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to setup temp table for {tid}: {e}")

    def get_join_context_sql(self, temp_table: str, alias_map: Dict[str, str], col_mapping: Dict[str, str]) -> str:
        """
        构建 FROM ... JOIN ... JOIN ... 字符串
        连接 Temp 表和真实表
        """
        # temp_table 类似于主表
        clauses = [f"FROM {temp_table}"]
        
        # 遍历 header 中的列，例如 t.id 对应 temp_table.t_id
        # 我们需要把 t (title) JOIN 到 temp_table.t_id 上
        
        # 反转 col_mapping: t_id -> t.id
        # 并解析出 alias
        
        for raw_col, safe_col in col_mapping.items():
            # raw_col = "t.id"
            if "." not in raw_col: continue
            alias, _ = raw_col.split(".")
            real_table = alias_map.get(alias)
            
            if not real_table:
                print(f"Warning: Alias {alias} not found in map, skipping join.")
                continue
                
            # JOIN title AS t ON temp_table.t_id = t.id
            join_clause = f"JOIN {real_table} AS {alias} ON {temp_table}.{safe_col} = {alias}.id"
            clauses.append(join_clause)
            
        return " ".join(clauses)

    def partition_ids(self, temp_table: str) -> List[List[int]]:
        """获取所有 virtual_id 并分区"""
        self.cursor.execute(f"SELECT virtual_id FROM {temp_table} ORDER BY virtual_id")
        ids = [r[0] for r in self.cursor.fetchall()]
        
        n = len(ids)
        if n == 0: return []
        
        # 分区逻辑
        base, rem = divmod(n, self.m_partitions)
        partitions = []
        start = 0
        for i in range(self.m_partitions):
            size = base + (1 if i < rem else 0)
            partitions.append(ids[start:start+size])
            start += size
            
        return partitions

    def select_best_tuple(self, temp_table: str, join_ctx: str, partition: List[int], queries: List[Dict], covered: set) -> int:
        """
        在 Join 上下文中选择最佳 tuple。
        Query: SELECT virtual_id, SUM(...) FROM temp JOIN ... WHERE virtual_id IN (...)
        """
        if not partition: return None
        
        uncovered = [(i, q) for i, q in enumerate(queries) if i not in covered]
        if not uncovered:
            return random.choice(partition)
        
        BATCH_SIZE = 50 
        scores = Counter()
        part_str = ",".join(map(str, partition))
        
        for i in range(0, len(uncovered), BATCH_SIZE):
            batch = uncovered[i:i+BATCH_SIZE]
            
            sum_cases = []
            for _, q in batch:
                pred = q['predicate_sql']
                sum_cases.append(f"CASE WHEN ({pred}) THEN 1 ELSE 0 END")
            
            score_expr = " + ".join(sum_cases)
            
            # 核心查询改造
            sql = f"""
                SELECT {temp_table}.virtual_id, ({score_expr}) as score
                {join_ctx}
                WHERE {temp_table}.virtual_id IN ({part_str})
            """
            
            try:
                self.cursor.execute(sql)
                for vid, score in self.cursor.fetchall():
                    if score > 0:
                        scores[vid] += score
            except Exception as e:
                print(f"Error in scoring: {e}")
                self.conn.rollback()
                continue
                
        if scores:
            return scores.most_common(1)[0][0]
        else:
            return random.choice(partition)

    def update_covered(self, temp_table: str, join_ctx: str, selected_vid: int, queries: List[Dict], covered: set):
        uncovered = [(i, q) for i, q in enumerate(queries) if i not in covered]
        if not uncovered: return
        
        BATCH_SIZE = 50
        newly_covered = set()
        
        for i in range(0, len(uncovered), BATCH_SIZE):
            batch = uncovered[i:i+BATCH_SIZE]
            
            union_parts = []
            for idx, q in batch:
                pred = q['predicate_sql']
                # 检查特定 ID 是否满足谓词
                sub_q = f"""
                    SELECT {idx} 
                    WHERE EXISTS (
                        SELECT 1 {join_ctx} 
                        WHERE {temp_table}.virtual_id = {selected_vid} AND ({pred})
                    )
                """
                union_parts.append(sub_q)
            
            full_sql = " UNION ALL ".join(union_parts)
            
            try:
                self.cursor.execute(full_sql)
                for (res_idx,) in self.cursor.fetchall():
                    newly_covered.add(res_idx)
            except Exception as e:
                print(f"Error updating coverage: {e}")
                self.conn.rollback()
        
        covered.update(newly_covered)

    def retrieve_full_row(self, temp_table: str, vid: int, header: List[str], col_mapping: Dict) -> List[Any]:
        """根据 virtual_id 取回原始的 join ID 组合"""
        # 构造查询列，例如 t_id, mi_id
        cols = [col_mapping[h] for h in header]
        sql = f"SELECT {','.join(cols)} FROM {temp_table} WHERE virtual_id = %s"
        self.cursor.execute(sql, (vid,))
        return list(self.cursor.fetchone())

    def run(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 加载数据
        join_data = self.load_join_results()
        parsed_workload = self.parse_workload_by_template()

        batch_counter = 0
        current_batch_samples = {}
        
        # 2. 遍历 Template
        for tid, data in join_data.items():
            template_key = tid
            if tid not in parsed_workload:
                print(f"Skipping {tid}: No matching queries found.")
                continue
                
            template_info = parsed_workload[template_key]
            queries = template_info['queries']
            alias_map = template_info['alias_map']
            
            print(f"\n=== Processing Template {tid} ({len(queries)} queries) ===")
            
            rows = data['rows']
            header = data['header']

            temp_table = None
            
            # 3. 创建临时表环境
            try:
                temp_table, col_mapping = self.setup_temp_table(tid, header, rows)
                join_ctx = self.get_join_context_sql(temp_table, alias_map, col_mapping)
                
                # 4. 执行分区和采样
                partitions = self.partition_ids(temp_table)
                covered_queries = set()
                
                tid_results = [] # 存储结果
                
                print(f"  Data partitioned into {len(partitions)} chunks.")
                
                for k in range(self.k_bitmaps):
                    print(f"  Bitmap {k+1}/{self.k_bitmaps} ...")
                    if len(covered_queries) == len(queries):
                        print("  All queries covered. Stopping early.")
                        break
                    elif len(covered_queries) >= len(queries) * 0.95:
                        print("  >95% queries covered. Stopping early.")
                        break
                        
                    bitmap_rows = []
                    for p_idx, partition in enumerate(partitions):
                        best_vid = self.select_best_tuple(temp_table, join_ctx, partition, queries, covered_queries)
                        if best_vid:
                            self.update_covered(temp_table, join_ctx, best_vid, queries, covered_queries)
                            # 获取原始 ID 存入结果
                            original_row = self.retrieve_full_row(temp_table, best_vid, header, col_mapping)
                            bitmap_rows.append(original_row)
                            
                        if (p_idx+1) % 5 == 0:
                            print(f"    Partition {p_idx+1}/{len(partitions)} done. Coverage: {len(covered_queries)}/{len(queries)}")
                            
                    # 保存该 bitmap (包含 header 以防万一，或者统一格式)
                    # 这里我们只存 rows，外层统一加 header 或者按照你之前的格式
                    # 为了兼容你的格式： [[header, r1, r2...], [header, r1...]]
                    if bitmap_rows:
                        tid_results.append([header] + bitmap_rows)

                current_batch_samples[tid] = tid_results
                batch_counter += 1
                if batch_counter % 5 == 0:
                    batch_id = batch_counter // 5
                    batch_file = os.path.join(output_dir, f"samples_batch_{batch_id}.json")
                    print(f"Saving batch {batch_id} to {batch_file}...")
                    with open(batch_file, 'w') as f:
                        json.dump(current_batch_samples, f, indent=2)
                    current_batch_samples = {}

            except Exception as e:
                print(f"Error processing {tid}: {e}")
                self.conn.rollback()
            finally:
                # 清理临时表
                if temp_table:
                    self.cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    self.conn.commit()

        # 最后保存所有结果
        if current_batch_samples:
            final_file = os.path.join(output_dir, f"samples_batch_final.json")
            print(f"Saving final samples to {final_file}...")
            with open(final_file, 'w') as f:
                json.dump(current_batch_samples, f, indent=2)

        print("Sampling completed.")

    def close(self):
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    config_file = "sampler_full_join_config.json"
    join_results_dir = "/data2/xuyining/Sampler/join_sampling/sample_results/small_table_300w/full_join" 
    
    sampler = JoinSampler(config_file, join_results_dir)
    try:
        sampler.run("/data2/xuyining/Sampler/single_table/full_join_sampler_results")
    finally:
        sampler.close()