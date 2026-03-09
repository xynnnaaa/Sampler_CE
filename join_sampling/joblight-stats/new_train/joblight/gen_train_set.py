import os
import glob
import pickle
from collections import defaultdict
import re
import sqlglot
import random
import networkx as nx
from networkx.readwrite import json_graph

def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def collect_predicates(base_query_dir):
    """
    Collect all unique predicates for each table from pkl files.
    Similar to load_and_parse_workload but only collects predicates.
    """
    global_predicate_map = defaultdict(set)

    pkl_files = sorted(glob.glob(os.path.join(base_query_dir, "*.pkl")))
    for pkl_file in pkl_files:
        try:
            qrep = load_qrep(pkl_file)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue

        join_graph = qrep["join_graph"]
        for alias, node_data in join_graph.nodes(data=True):
            real_name = node_data["real_name"]
            preds_list = node_data.get("predicates", [])

            clean_pred_list = [] # 没有别名的谓词

            for pred in preds_list:
                # Remove alias from predicate
                pattern = fr"\b{alias}\."
                pred_clean = re.sub(pattern, "", pred).strip()
                clean_pred_list.append(pred_clean)

            if clean_pred_list:
                combined_pred = " AND ".join(clean_pred_list)
                global_predicate_map[real_name].add(combined_pred)

    # Convert sets to lists
    for table in global_predicate_map:
        global_predicate_map[table] = list(global_predicate_map[table])

    return global_predicate_map

def extract_tables(query):
    """
    Extract table names from a SQL query using sqlglot.
    """
    try:
        parsed = sqlglot.parse_one(query, read="postgres")
        tables = set()
        for table in parsed.find_all(sqlglot.exp.Table):
            tables.add(table.name)
        return list(tables)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return []

def modify_query(query, tables, predicates_map):
    """
    Rewrite the query so that the original WHERE clause is analysed and only
    non-join predicates are stripped away. Join conditions (expressions that
    refer to more than one alias) are retained. A new WHERE clause is then
    constructed consisting of the preserved join conditions plus randomly
    chosen predicates for each table alias (50% chance per alias).
    """
    def _split_and(expr):
        # return list of conjunctive subexpressions
        if isinstance(expr, sqlglot.exp.And):
            return _split_and(expr.this) + _split_and(expr.expression)
        return [expr]

    try:
        parsed = sqlglot.parse_one(query, read="postgres")

        # build alias -> real table name map
        alias_map = {}
        for table_expr in parsed.find_all(sqlglot.exp.Table):
            alias = table_expr.alias_or_name
            alias_map[alias] = table_expr.name

        join_terms = []
        # examine existing WHERE
        where = parsed.find(sqlglot.exp.Where)
        if where and where.this is not None:
            for term in _split_and(where.this):
                # gather distinct aliases referenced in this term
                refs = {col.table for col in term.find_all(sqlglot.exp.Column) if col.table}
                if len(refs) > 1:
                    # term involves multiple tables -> treat as join condition
                    join_terms.append(term)
                # else drop it (single-table filter)
        # prepare new random predicates for each alias in query
        new_terms = []
        for alias, real in alias_map.items():
            preds = predicates_map.get(real, [])
            if preds and random.random() > 0.5:
                pred_str = random.choice(preds)
                # parse the clean predicate and reattach alias to all columns
                try:
                    pred_expr = sqlglot.parse_one(pred_str, read="postgres")
                    for col in pred_expr.find_all(sqlglot.exp.Column):
                        if not col.table:
                            col.set("table", alias)
                    new_terms.append(pred_expr)
                except Exception:
                    # fallback: just prefix alias manually (may be incorrect)
                    new_terms.append(sqlglot.parse_one(f"{alias}.{pred_str}", read="postgres"))
        # if no predicate chosen due to randomness, force one table to have a predicate
        if not new_terms and alias_map:
            # pick random alias and add one predicate if available
            alias, real = random.choice(list(alias_map.items()))
            preds = predicates_map.get(real, [])
            if preds:
                pred_str = random.choice(preds)
                try:
                    pred_expr = sqlglot.parse_one(pred_str, read="postgres")
                    for col in pred_expr.find_all(sqlglot.exp.Column):
                        if not col.table:
                            col.set("table", alias)
                    new_terms.append(pred_expr)
                except Exception:
                    new_terms.append(sqlglot.parse_one(f"{alias}.{pred_str}", read="postgres"))

        all_terms = join_terms + new_terms
        if all_terms:
            # reassemble AND chain
            expr = all_terms[0]
            for t in all_terms[1:]:
                expr = sqlglot.exp.And(this=expr, expression=t)
            parsed.set("where", sqlglot.exp.Where(this=expr))
        else:
            parsed.set("where", None)

        return parsed.sql(dialect="postgres")
    except Exception as e:
        print(f"Error modifying query: {e}")
        return query

def generate_variants(sql_file, predicates_map, k=5, output_file='new_queries.sql'):
    """
    Generate k variants for each query in the sql_file by modifying predicates.
    """
    with open(sql_file, 'r') as f:
        content = f.read()

    queries = [q.strip() for q in content.split(';') if q.strip() and not q.strip().startswith('--')]

    new_queries = []
    for query in queries:
        tables = extract_tables(query)
        for _ in range(k):
            new_q = modify_query(query, tables, predicates_map)
            if new_q not in new_queries:
                new_queries.append(new_q)

    with open(output_file, 'w') as f:
        f.write(';\n'.join(new_queries) + ';\n')

if __name__ == "__main__":
    # Example usage, adjust paths as needed
    base_query_dir = "/data1/xuyining/CEB-default/queries/joblight_train/joblight-train-all"  # Path to directory with pkl files
    sql_file = "/home/xuyining/End-to-End-CardEst-Benchmark/workloads/job-light/job_light_queries.sql"     # Input SQL file with queries separated by ;
    output_file = "./joblight/train/train.sql"
    k = 150  # Number of variants per query

    predicates = collect_predicates(base_query_dir)
    print("Collected predicates:")
    for table, preds in predicates.items():
        print(f"  {table}: {len(preds)} predicates")

    generate_variants(sql_file, predicates, k=k, output_file=output_file)
    print(f"Generated variants saved to {output_file}")