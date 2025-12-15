import json
import psycopg2
import os
import sys
import glob
import pickle
from typing import List, Dict, Any, Tuple
from sqlglot import parse_one, exp
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def get_table_alias_map(expression: exp.Expression) -> Dict[str, str]:
    table_alias_map = {}
    for table in expression.find_all(exp.Table):
        table_name = table.name
        alias = table.alias_or_name
        table_alias_map[alias] = table_name
    return table_alias_map

def parse_single_query_predicates(query_sql: str) -> Dict[Tuple[str, str], str]:
    predicates = {}
    try:
        expression = parse_one(query_sql, read="postgres")
        table_alias_map = get_table_alias_map(expression)
        where_clause = expression.find(exp.Where)
        alias_specific_filters: Dict[str, List[exp.Expression]] = defaultdict(list)

        if where_clause:
            conjuncts = list(where_clause.this.flatten()) if isinstance(where_clause.this, exp.And) else [where_clause.this]
            for condition in conjuncts:
                involved_table_aliases = set()
                for col_name in condition.find_all(exp.Column):
                    alias = col_name.table
                    if alias and alias not in table_alias_map:
                        for a, t in table_alias_map.items():
                            if t == alias:
                                alias = a
                                break
                    if alias:
                        involved_table_aliases.add(alias)

                if len(involved_table_aliases) == 1:
                    alias = list(involved_table_aliases)[0]
                    alias_specific_filters[alias].append(condition)

            for alias, table_name in table_alias_map.items():
                if alias_specific_filters[alias]:
                    conditions = alias_specific_filters[alias]
                    combined = exp.and_(*conditions) if len(conditions) > 1 else conditions[0]
                    combined_no_alias = combined.transform(
                        lambda node: exp.Column(this=node.this) if isinstance(node, exp.Column) and node.table else node
                    )
                    predicates[(table_name, alias)] = combined_no_alias.sql(dialect="postgres")
    except Exception as e:
        print(f"Warning: Could not parse query for vector generation: {query_sql[:100]}...\nError: {e}")
    return predicates


class QueryClusterer:
    def __init__(self, config_path: str):
        print("Initializing Query Clusterer...")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.db_conn = psycopg2.connect(
            dbname=self.config["db"]["name"],
            user=self.config["db"]["user"],
            password=self.config["db"]["password"],
            host=self.config["db"]["host"],
            port=self.config["db"]["port"],
        )
        self.cursor = self.db_conn.cursor()

        self.samples_data = None
        self.workload: List[Dict[str, str]] = []
        self.max_k = 0

    def _load_data(self):
        """加载 samples.json 和查询工作负载"""
        print("Loading samples and query workload...")
        samples_path = self.config["paths"]["samples_file"]
        with open(samples_path, 'r') as f:
            self.samples_data = json.load(f)
        
        if self.samples_data:
            self.max_k = max((len(samples) for samples in self.samples_data.values()), default=0)
        print(f"Max number of samples (max_k) found: {self.max_k}")

        base_query_dir = self.config["paths"]["base_query_dir"]
        try:
            template_names = sorted([d for d in os.listdir(base_query_dir) if os.path.isdir(os.path.join(base_query_dir, d))])
            for template_name in template_names:
                pkl_files = sorted(glob.glob(os.path.join(base_query_dir, template_name, "*.pkl")))
                for pkl_file in pkl_files:
                    with open(pkl_file, "rb") as f:
                        sql = pickle.load(f).get("sql")
                        query_name = os.path.splitext(os.path.basename(pkl_file))[0]
                        if sql:
                            self.workload.append({"name": query_name, "sql": sql})
        except FileNotFoundError:
            print(f"ERROR: Base query directory not found: {base_query_dir}")
            sys.exit(1)
        print(f"Loaded {len(self.workload)} queries from workload.")

    def manage_materialized_views(self, create=True):
        """根据 samples.json 创建或删除物化视图"""
        action = "Creating" if create else "Dropping"
        print(f"{action} materialized views...")
        
        for table_name, samples in self.samples_data.items():
            for i, pks in enumerate(samples):
                view_name = f"sample_{table_name}_b{i}"
                if create:
                    if not pks: continue
                    pks_str = ', '.join(map(str, pks))
                    
                    self.cursor.execute(f"""
                        SELECT kcu.column_name FROM information_schema.table_constraints AS tc 
                        JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table_name}';
                    """)
                    pk_col_result = self.cursor.fetchone()
                    if not pk_col_result:
                        print(f"Warning: No primary key found for table {table_name}. Cannot create view {view_name}.")
                        continue
                    pk_col = pk_col_result[0]
                    
                    # self.cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};")
                    sql = f"""
                        CREATE MATERIALIZED VIEW {view_name} AS
                        SELECT * FROM {table_name} WHERE {pk_col} IN ({pks_str})
                        IF NOT EXISTS;
                    """
                    self.cursor.execute(sql)
                else:
                    self.cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};")
        
        self.db_conn.commit()
        print(f"Materialized views {action.lower()} completed.")

    def _get_global_alias_order(self) -> List[Tuple[str, str]]:
        """解析整个工作负载，获取全局统一的 (表名, 别名) 排序列表"""
        print("Determining global alias order...")
        all_aliases = set()
        for query_info in self.workload:
            try:
                expression = parse_one(query_info["sql"], read="postgres")
                for table_node in expression.find_all(exp.Table):
                    if len(self.samples_data.get(table_node.name, [])) > 0:
                        # 只考虑在 samples.json 中有样本的表
                        all_aliases.add((table_node.name, table_node.alias_or_name))
            except Exception:
                continue
        
        sorted_aliases = sorted(list(all_aliases))
        print(f"Found {len(sorted_aliases)} unique (table, alias) pairs.")
        print(f"Global alias order: {sorted_aliases}")
        return sorted_aliases

    def _generate_sampling_vectors(self, global_alias_order: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """为每个查询生成采样向量，使用查询名作为键"""
        print("Generating sampling vectors for all queries...")
        query_vectors = {}

        for i, query_info in enumerate(self.workload):
            query_name = query_info["name"]
            query_sql = query_info["sql"]
            
            print(f"  Processing query {i+1}/{len(self.workload)} ({query_name})...", end='\r')
            
            query_predicates = parse_single_query_predicates(query_sql)
            full_vector = []

            for table_name, alias in global_alias_order:
                if (table_name, alias) in query_predicates:
                    predicate = query_predicates[(table_name, alias)]

                    k_actual = len(self.samples_data.get(table_name, []))
                    hit_vector = np.zeros(k_actual, dtype=int)

                    for sample_idx in range(k_actual):
                        view_name = f"sample_{table_name}_b{sample_idx}"
                        check_sql = f"SELECT 1 FROM {view_name} WHERE {predicate} LIMIT 1;"
                        try:
                            self.cursor.execute(check_sql)
                            if self.cursor.fetchone():
                                hit_vector[sample_idx] = 1
                        except psycopg2.Error:
                            self.db_conn.rollback()
                            pass
                    
                    padding = np.zeros(self.max_k - k_actual, dtype=int)
                    segment = np.concatenate([hit_vector, padding])
                    full_vector.append(segment)
                else:
                    full_vector.append(np.zeros(self.max_k, dtype=int))
            
            query_vectors[query_name] = np.concatenate(full_vector)
        
        print("\nSampling vector generation complete.")
        return query_vectors

    def _cluster_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        """使用K-Means对向量进行聚类"""
        print("length of one vector:", len(vectors[0]) if vectors else 0)
        n_clusters = self.config["clustering"]["n_clusters"]
        print(f"Clustering {len(vectors)} vectors into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(vectors)
        print("Clustering complete.")
        return kmeans.labels_

    def _find_optimal_samples(self, labels: np.ndarray, vectors: List[np.ndarray], global_alias_order: List[Tuple[str, str]]) -> Dict[int, Dict[str, int]]:
        """为每个类别计算最优样本编号"""
        print("Finding optimal samples for each cluster...")
        n_clusters = self.config["clustering"]["n_clusters"]
        cluster_optimal_samples = {}
        
        X = np.array(vectors)

        for i in range(n_clusters):
            cluster_vectors = X[labels == i]
            if len(cluster_vectors) == 0: continue

            cluster_hit_sums = np.sum(cluster_vectors, axis=0)
            
            optimal_samples_for_cluster = {}
            start_index = 0
            for table_name, alias in global_alias_order:
                segment = cluster_hit_sums[start_index : start_index + self.max_k]
                best_sample_idx = np.argmax(segment)
                alias_key = f"{table_name}_{alias}"
                optimal_samples_for_cluster[alias_key] = int(best_sample_idx)
                start_index += self.max_k
            
            cluster_optimal_samples[i] = optimal_samples_for_cluster
        
        print("Optimal sample selection complete.")
        return cluster_optimal_samples
    
    def _evaluate_clustering_effectiveness(
        self,
        query_vectors_map: Dict[str, np.ndarray],
        query_cluster_map: Dict[str, int],
        cluster_optimal_samples: Dict[int, Dict[str, int]],
        global_alias_order: List[Tuple[str, str]]
    ):
        """
        评估聚类和最优样本选择的效果。
        计算在最优样本上，所有查询的谓词命中和未命中总数。
        """
        print("\nEvaluating clustering effectiveness...")

        total_predicates = 0
        total_hits_on_optimal = 0
        total_misses_on_optimal = 0

        # 按类别组织评估结果
        cluster_stats = defaultdict(lambda: {"total_predicates": 0, "hits": 0, "misses": 0})

        for query_name, vector in query_vectors_map.items():
            cluster_label = query_cluster_map[query_name]
            optimal_samples_for_cluster = cluster_optimal_samples.get(str(cluster_label)) # JSON的键是字符串

            if not optimal_samples_for_cluster:
                continue

            # 遍历该查询向量的每个分段
            start_index = 0
            for table_name, alias in global_alias_order:
                # 提取该别名对应的向量分段
                segment = vector[start_index : start_index + self.max_k]
                
                # 检查这个分段是否全为0（即该查询在此别名上无谓词）
                if np.all(segment == 0):
                    start_index += self.max_k
                    continue
                
                # 这个分段代表一个有实际谓词的查询部分
                total_predicates += 1
                cluster_stats[cluster_label]["total_predicates"] += 1

                # 找到该别名在此类别下的最优样本编号
                alias_key = f"{table_name}_{alias}"
                optimal_sample_idx = optimal_samples_for_cluster.get(alias_key)

                if optimal_sample_idx is None:
                    start_index += self.max_k
                    continue

                # 检查在最优样本上的命中情况
                if segment[optimal_sample_idx] == 1:
                    total_hits_on_optimal += 1
                    cluster_stats[cluster_label]["hits"] += 1
                else:
                    total_misses_on_optimal += 1
                    cluster_stats[cluster_label]["misses"] += 1
                
                start_index += self.max_k
        
        print("\n--- Clustering Effectiveness Report ---")
        if total_predicates == 0:
            print("No predicates found to evaluate.")
            return

        overall_hit_rate = (total_hits_on_optimal / total_predicates) * 100
        print(f"Overall Predicate Hit Rate on Optimal Samples: {overall_hit_rate:.2f}%")
        print(f"  - Total Predicates with Conditions: {total_predicates}")
        print(f"  - Total Hits on Optimal Samples:    {total_hits_on_optimal}")
        print(f"  - Total Misses on Optimal Samples:  {total_misses_on_optimal}")

        print("\nPer-Cluster Statistics:")
        for label, stats in sorted(cluster_stats.items()):
            cluster_total = stats['total_predicates']
            if cluster_total == 0: continue
            
            hits = stats['hits']
            misses = stats['misses']
            hit_rate = (hits / cluster_total) * 100
            print(f"  - Cluster {label}:")
            print(f"    - Hit Rate: {hit_rate:.2f}% ({hits} hits / {misses} misses)")
            print(f"    - Total Predicates in Cluster: {cluster_total}")

        print("-------------------------------------\n")

    def _save_results(self, query_cluster_map, cluster_optimal_samples):
        """保存结果到JSON文件"""
        map_path = self.config["paths"]["query_cluster_map_output"]
        optimal_path = self.config["paths"]["cluster_optimal_samples_output"]
        n_clusters = self.config["clustering"]["n_clusters"]

        map_dir, map_filename = os.path.split(map_path)
        optimal_dir, optimal_filename = os.path.split(optimal_path)

        new_map_filename = f"{n_clusters}_{map_filename}"
        new_optimal_filename = f"{n_clusters}_{optimal_filename}"

        map_path_final = os.path.join(map_dir, new_map_filename)
        optimal_path_final = os.path.join(optimal_dir, new_optimal_filename)

        
        print(f"Saving query-to-cluster map to {map_path_final}...")
        with open(map_path_final, 'w') as f:
            json.dump(query_cluster_map, f, indent=4)

        print(f"Saving cluster-to-optimal-samples map to {optimal_path_final}...")
        with open(optimal_path_final, 'w') as f:
            json.dump(cluster_optimal_samples, f, indent=4)
        
        print("Results saved.")

    def run(self):
        """执行完整的聚类流程"""
        self._load_data()
        self.manage_materialized_views(create=True)
        
        global_alias_order = self._get_global_alias_order()
        query_vectors_map = self._generate_sampling_vectors(global_alias_order)

        queries_in_order = list(query_vectors_map.keys())
        vectors_in_order = list(query_vectors_map.values())
        
        labels = self._cluster_vectors(vectors_in_order)

        query_cluster_map = {query_name: int(label) for query_name, label in zip(queries_in_order, labels)}
        
        optimal_samples_int_keys = self._find_optimal_samples(labels, vectors_in_order, global_alias_order)
        optimal_samples_str_keys = {str(k): v for k, v in optimal_samples_int_keys.items()}

        self._save_results(query_cluster_map, optimal_samples_str_keys)

        self._evaluate_clustering_effectiveness(
            query_vectors_map, query_cluster_map, optimal_samples_str_keys, global_alias_order
        )

    def close(self):
        """关闭数据库连接"""
        if self.db_conn:
            self.cursor.close()
            self.db_conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    clusterer = QueryClusterer("cluster_config.json")
    try:
        clusterer.run()
        # clusterer._load_data()
        # global_alias_order = clusterer._get_global_alias_order()
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        clusterer.close()