import os
import re
import sys
import psycopg2
from collections import defaultdict

# 获取当前脚本所在目录: .../Sampler/scripts/
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 定位到 Sampler 根目录 (../)
project_root = os.path.abspath(os.path.join(curr_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 定位到 mscn 目录 (.../Sampler/mscn)
# 这一步是为了让 query.py 里的 "from query_representation" 能够被解析
mscn_root = os.path.join(project_root, "mscn")
if mscn_root not in sys.path:
    sys.path.append(mscn_root)

from query_representation.utils import extract_join_graph

# ================= 配置区域 =================
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "ergastf1", 
    "user": "xuyining",
    "password": "123"
}

# 你的 SQL 文件列表（包含训练和测试）
SQL_FILES = [
    "/home/PRICE/datas/workloads/finetune/ergastf1/workloads.sql",
    "/home/PRICE/datas/workloads/test/ergastf1/workloads.sql"
]

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def remove_unnecessary_parentheses(sql_str):
    """
    预处理 SQL，专门剥离 WHERE 条件中每个谓词最外层的括号，并删去末尾分号
    例如：将 WHERE (A = B) AND ((C = D)) 转换为 WHERE A = B AND C = D
    注意：不会影响 SELECT COUNT(*) 中的括号，也不会影响 IN (1, 2) 的括号。
    """
    # 1. 找到 WHERE 的位置 (忽略大小写)
    match = re.search(r'\bWHERE\b', sql_str, re.IGNORECASE)
    if not match:
        return sql_str  # 如果没有 WHERE 子句，直接返回
    
    where_idx = match.end()
    select_from_part = sql_str[:where_idx] # "SELECT ... FROM ... WHERE"
    where_part = sql_str[where_idx:]       # " (a=b) AND (c=d);"
    
    # 2. 提取并暂时移除末尾的分号（如果有）
    where_part = where_part.strip()
    has_semicolon = where_part.endswith(";")
    if has_semicolon:
        where_part = where_part[:-1].strip()
        
    # 3. 按 AND 切分所有的谓词条件 (忽略大小写)
    conditions = re.split(r'\s+AND\s+', where_part, flags=re.IGNORECASE)
    
    # 4. 剥离每个条件最外层的括号
    clean_conditions = []
    for cond in conditions:
        cond = cond.strip()
        # 循环剥离，应对像 ((a.id = b.id)) 这种嵌套的多重括号
        while cond.startswith("(") and cond.endswith(")"):
            cond = cond[1:-1].strip()
        clean_conditions.append(cond)
        
    # 5. 重新使用 AND 拼装起来
    new_where = " AND ".join(clean_conditions)
    return select_from_part + " " + new_where


def collect_stats():
    max_join_size = 0
    # 存储所有唯一的物理连接关系: (table_a, col_a, table_b, col_b)
    # 注意：这里存物理表名，不存别名
    join_edges = set()

    print("--- Step 1: Parsing Workloads for Max Join Size and Edges ---")
    for sql_file in SQL_FILES:
        if not os.path.exists(sql_file):
            continue
        with open(sql_file, 'r') as f:
            for line in f:
                sql = line.split("||")[0].strip()
                if "SELECT" not in sql.upper(): continue

                sql = remove_unnecessary_parentheses(sql)
                
                try:
                    # 1. 统计 Join Size
                    # 这里简单的通过别名数量判断，如果是 Acyclic 则 Join Size = 节点数
                    # 如果有重复表别名，len(nodes) 依然代表参与 Join 的关系数量
                    jg = extract_join_graph(sql)
                    nodes = jg.nodes(data=True)
                    max_join_size = max(max_join_size, len(nodes))

                    # 2. 提取 Join Edges (用于后续查库算 Fanout)
                    for u, v, data in jg.edges(data=True):
                        cond = data.get('join_condition', "")
                        # 解析形如 "t.id = mi.movie_id"
                        match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', cond)
                        if match:
                            a_alias, a_col, b_alias, b_col = match.groups()
                            # 映射回真实物理表名
                            a_real = jg.nodes[a_alias]['real_name']
                            b_real = jg.nodes[b_alias]['real_name']
                            
                            # 存储为有序元组，防止 (A,B) 和 (B,A) 重复计算
                            # 我们需要分别看 A->B 和 B->A 的 fanout
                            join_edges.add((a_real, a_col, b_real, b_col))
                            join_edges.add((b_real, b_col, a_real, a_col))
                except:
                    continue

    print(f"Result: Maximum Join Size = {max_join_size}")
    print(f"Found {len(join_edges)} unique physical join directions.\n")

    print("--- Step 2: Querying Database for Max Join Fanout (X) ---")
    conn = get_connection()
    cursor = conn.cursor()
    
    global_max_fanout = 0
    detailed_fanouts = []

    for left_table, left_col, right_table, right_col in join_edges:

        check_sql = f"""
            SELECT MAX(cnt) 
            FROM (
                SELECT r.{right_col}, COUNT(*) as cnt 
                FROM {right_table} AS r
                WHERE r.{right_col} IN (SELECT l.{left_col} FROM {left_table} AS l WHERE l.{left_col} IS NOT NULL)
                GROUP BY r.{right_col}
            ) AS tmp;
        """
        
        try:
            cursor.execute(check_sql)
            res = cursor.fetchone()[0]
            current_fanout = res if res else 0
            global_max_fanout = max(global_max_fanout, current_fanout)
            
            detailed_fanouts.append({
                "edge": f"{left_table}.{left_col} -> {right_table}.{right_col}",
                "fanout": current_fanout
            })
            print(f"  Checked {left_table} -> {right_table}: Max Fanout = {current_fanout}")
        except Exception as e:
            print(f"  Error checking {left_table} -> {right_table}: {e}")
            conn.rollback()

    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    print(f"MAX JOIN SIZE:   {max_join_size}")
    print(f"MAX JOIN FANOUT: {global_max_fanout}")
    print("-" * 50)
    # 打印前 5 个最严重的倾斜边
    print("Top Skewed Edges (Potential Bottlenecks):")
    detailed_fanouts.sort(key=lambda x: x['fanout'], reverse=True)
    for item in detailed_fanouts[:5]:
        print(f"  {item['edge']}: {item['fanout']}")
    print("="*50)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    collect_stats()