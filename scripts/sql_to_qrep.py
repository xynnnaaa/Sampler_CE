import os
import pdb
import networkx as nx
from networkx.readwrite import json_graph
from query_representation.utils import *
from query_representation.query import *

# 输入文件：请按需修改为目标 workloads.sql 路径
INPUT_FN = "/data2/xuyining/PRICE/datas/workloads/test/imdb/workloads.sql"
# 输出目录
OUTPUT_DIR = "./queries/joblight/all_joblight/"

make_dir(OUTPUT_DIR)

def is_comment_line(s):
    s = s.lstrip()
    return s.startswith('/*') or s.startswith('--') or s == ''

with open(INPUT_FN, "r") as f:
    lines = f.readlines()

idx = 0
for line in lines:
    # 每行可能包含查询和后续用 || 分割的元数据，先取第一个部分
    stripped = line.split('||', 1)[0].strip()
    if is_comment_line(stripped):
        # 以注释开头的行代表子查询或注释，跳过
        continue
    if stripped == '':
        continue

    sql = stripped
    # 保证是一个 SELECT 查询（大小写不敏感）
    if 'select' not in sql.lower():
        continue

    try:
        qrep = parse_sql(sql, None, None, None, None, None,
                          compute_ground_truth=False)
    except Exception:
        # 如果解析失败，记录并跳过该查询以免中断批处理
        print(f"parse_sql failed for line {idx}: {sql[:200]}")
        continue

    # 转换图结构以便序列化
    if 'subset_graph' in qrep:
        qrep['subset_graph'] = nx.OrderedDiGraph(json_graph.adjacency_graph(qrep['subset_graph']))
    if 'join_graph' in qrep:
        qrep['join_graph'] = json_graph.adjacency_graph(qrep['join_graph'])

    output_fn = os.path.join(OUTPUT_DIR, f"{idx}.pkl")
    save_qrep(output_fn, qrep)
    idx += 1

