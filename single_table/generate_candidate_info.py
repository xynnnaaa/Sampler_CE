import json
import psycopg2
import os
import sys
from typing import List, Dict, Any

class SampleMaterializer:
    def __init__(self, db_config_path: str, samples_path: str, output_info_path: str):
        self.db_config_path = db_config_path
        self.samples_path = samples_path
        self.output_info_path = output_info_path
        self.conn = None
        self.cursor = None

    def connect_db(self):
        """Connects to the database using the config file."""
        print(f"Loading database config from: {self.db_config_path}")
        try:
            with open(self.db_config_path, 'r') as file:
                config = json.load(file)
            
            # Support both structure types (direct keys or nested under "db")
            db_conf = config.get("db", config)
            
            print("Connecting to the database...")
            self.conn = psycopg2.connect(
                dbname=db_conf["name"],
                user=db_conf["user"],
                password=db_conf["password"],
                host=db_conf["host"],
                port=db_conf["port"],
            )
            self.conn.autocommit = True # Enable autocommit for DDL statements
            self.cursor = self.conn.cursor()
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)

    def load_samples(self) -> Dict[str, List[List[Any]]]:
        """Loads the samples.json file."""
        print(f"Reading samples from: {self.samples_path}")
        try:
            with open(self.samples_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading samples file: {e}")
            sys.exit(1)

    def create_materialized_tables(self, samples_data: Dict[str, List[List[Any]]]):
        """
        Iterates through samples and creates physical tables in the DB.
        Naming convention: {original_table}_s{sample_size}_{index}
        Example: movie_s100_0
        """
        print("\n--- Starting Materialization of Sample Tables ---")
        
        for table_name, samples_list in samples_data.items():
            print(f"Processing original table: {table_name}")
            
            for i, sample_ids in enumerate(samples_list):
                sample_size = 1000
                
                # 构造新表名：原表名 + _s样本大小 + _序号
                # 例如：title_s100_0
                new_table_name = f"{table_name}_s{sample_size}_{i}_with_partition"
                
                if sample_size == 0:
                    print(f"  Skipping {new_table_name} (empty sample).")
                    continue

                # 将 ID 列表转换为 SQL 友好的字符串 (例如: "1, 2, 3")
                ids_str = ", ".join(map(str, sample_ids))
                
                # 构造 SQL 语句
                # 假设主键列名为 'id'，这与你之前的 Sampler 脚本逻辑一致
                primary_key_col = 'id' 
                
                drop_query = f"DROP TABLE IF EXISTS {new_table_name};"
                create_query = f"""
                    CREATE TABLE {new_table_name} AS 
                    SELECT * FROM {table_name} 
                    WHERE {primary_key_col} IN ({ids_str});
                """
                
                try:
                    # 执行删表和建表
                    self.cursor.execute(drop_query)
                    self.cursor.execute(create_query)
                    
                    # 可选：执行 ANALYZE 以更新统计信息，这对于后续查询优化器很重要
                    self.cursor.execute(f"ANALYZE {new_table_name};")
                    
                    print(f"  [SUCCESS] Created table: {new_table_name} (Rows: {sample_size})")
                    
                except Exception as e:
                    print(f"  [ERROR] Failed to create {new_table_name}: {e}")
                    # 注意：由于设置了 autocommit，这里的失败不会回滚之前的成功操作
        
        print("--- Materialization Complete ---\n")

    def generate_candidate_info(self, samples_data: Dict[str, List[List[Any]]]):
        """
        Generates the candidate_samples_info.json file mapping tables to available indices.
        """
        print("Generating candidate sample info...")
        candidate_info = {}

        for table_name, samples_list in samples_data.items():
            k_actual = len(samples_list)
            
            # 生成索引列表：0, 1, ..., k-1
            # 同时也对应了数据库中 _0, _1 ... 的后缀
            if k_actual > 0:
                candidate_info[table_name] = list(range(k_actual))
                print(f"  - {table_name}: {k_actual} samples recorded.")
            else:
                print(f"  - {table_name}: No samples found.")

        print(f"Saving candidate info to: {self.output_info_path}")
        try:
            with open(self.output_info_path, 'w') as f:
                json.dump(candidate_info, f, indent=4)
            print("Successfully generated candidate_samples_info.json!")
        except IOError as e:
            print(f"ERROR: Could not write output file: {e}")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed.")

    def run(self):
        self.connect_db()
        try:
            samples_data = self.load_samples()
            
            # 1. 在数据库中创建物理表
            self.create_materialized_tables(samples_data)
            
            # 2. 生成信息文件
            self.generate_candidate_info(samples_data)
            
        finally:
            self.close()

if __name__ == '__main__':
    # --- 配置路径 ---
    # 请根据你的实际文件位置修改这些路径
    
    BASE_DIR = "/data2/xuyining/Sampler/single_table/ceb_imdb_results/1000"
    
    # 数据库配置文件 (包含 host, user, password, dbname 等)
    CONFIG_FILE = os.path.join(BASE_DIR, "sampler_config.json")
    
    # 采样结果文件 (输入)
    SAMPLES_FILE = os.path.join(BASE_DIR, "samples.json")
    
    # 候选信息文件 (输出)
    CANDIDATE_FILE = os.path.join(BASE_DIR, "candidate_samples_info.json")
    
    # --- 执行 ---
    materializer = SampleMaterializer(CONFIG_FILE, SAMPLES_FILE, CANDIDATE_FILE)
    materializer.run()