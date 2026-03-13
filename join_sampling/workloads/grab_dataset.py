# import pandas as pd
# from sqlalchemy import create_engine, text, inspect
# import psycopg2

# # --- 1. 配置参数 ---
# # 远程 MariaDB
# REMOTE_DB_NAME = "ErgastF1" 
# MARIADB_URI = f"mysql+pymysql://guest:ctu-relational@relational.fel.cvut.cz:3306/{REMOTE_DB_NAME}"

# # 本地 Postgres 配置
# PG_USER = "xuyining"
# PG_HOST = "localhost"
# PG_PORT = 5433
# PG_DBNAME = "ergastf1"  # 本地新建的数据库名

# # --- 2. 创建本地数据库 ---
# default_pg_uri = f"postgresql://{PG_USER}@{PG_HOST}:{PG_PORT}/postgres"
# engine_default = create_engine(default_pg_uri, isolation_level="AUTOCOMMIT")

# with engine_default.connect() as conn:
#     exists = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{PG_DBNAME}'")).fetchone()
#     if not exists:
#         print(f"正在创建数据库: {PG_DBNAME}...")
#         conn.execute(text(f"CREATE DATABASE {PG_DBNAME}"))

# # --- 3. 初始化连接 ---
# src_engine = create_engine(MARIADB_URI)
# dest_engine = create_engine(f"postgresql://{PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}")

# # 获取远程所有表名
# inspector = inspect(src_engine)
# all_tables = inspector.get_table_names()
# print(f"发现远程 ErgastF1 库中共有 {len(all_tables)} 张表。")

# # --- 4. 循环迁移并自动处理大小写 ---
# for remote_table in all_tables:
#     # 强制本地表名为小写
#     local_table = remote_table.lower()
#     print(f"\n>>> 正在迁移: {remote_table} -> {local_table}")
    
#     try:
#         # 分块读取数据
#         # ErgastF1 数据量远小于 Visual Genome，chunksize 可以设大一点
#         chunks = pd.read_sql(f"SELECT * FROM `{remote_table}`", src_engine, chunksize=10000)
        
#         is_first_chunk = True
#         for chunk in chunks:
#             # 强制所有列名为小写
#             chunk.columns = [c.lower() for c in chunk.columns]
            
#             if is_first_chunk:
#                 # if_exists='replace' 会根据 DataFrame 结构自动建表
#                 chunk.to_sql(local_table, dest_engine, if_exists='replace', index=False, method='multi')
#                 is_first_chunk = False
#             else:
#                 chunk.to_sql(local_table, dest_engine, if_exists='append', index=False, method='multi')
        
#         print(f"✅ {local_table} 迁移完成。")
        
#     except Exception as e:
#         print(f"❌ 迁移 {remote_table} 时出错: {e}")

# # --- 5. 刷新统计信息 (对 Cardinality Estimation 至关重要) ---
# print("\n>>> 正在更新统计信息 (ANALYZE)...")
# with dest_engine.connect() as conn:
#     conn.execute(text("ANALYZE"))
#     conn.commit()

# print(f"\n🎉 所有 ErgastF1 数据已就绪！你可以通过 psql -p 5433 -d {PG_DBNAME} 访问。")



import psycopg2
from psycopg2 import sql

# --- 配置参数 ---
# 你可以切换 dbname 为 'genome' 或 'ergastf1'
conn_params = {
    "host": "localhost",
    "port": 5433,
    "dbname": "ergastf1", 
    "user": "xuyining"
}

def add_primary_keys():
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()

        # 1. 获取所有没有主键的用户表
        # 这个查询会过滤掉已经有主键的表，防止重复添加报错
        cursor.execute("""
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public' 
            AND tablename NOT IN (
                SELECT conrelid::regclass::text 
                FROM pg_constraint 
                WHERE contype = 'p'
            );
        """)
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            print("所有表都已经拥有主键，无需操作。")
            return

        print(f"发现 {len(tables)} 张表缺失主键: {tables}")

        for table in tables:
            print(f"正在为表 {table} 添加自增主键 id...")
            try:
                # 使用 GENERATED ALWAYS AS IDENTITY (Postgres 10+ 标准语法)
                # 这会自动创建序列并从 1 开始
                query = sql.SQL("""
                    ALTER TABLE {} 
                    ADD COLUMN id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY
                """).format(sql.Identifier(table))
                
                cursor.execute(query)
                print(f"  ✅ 表 {table} 已成功添加 id 列。")
            except Exception as e:
                print(f"  ❌ 为表 {table} 添加主键时失败: {e}")

        # 2. 刷新统计信息
        print("\n正在更新统计信息 (ANALYZE)...")
        cursor.execute("ANALYZE;")
        
        print("\n🎉 所有操作已完成！")

    except Exception as e:
        print(f"无法连接数据库: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    add_primary_keys()