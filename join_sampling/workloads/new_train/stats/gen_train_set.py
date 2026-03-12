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

def collect_predicates(sql_file):
    """
    Collect all unique predicates for each table from a SQL file.
    The SQL file format is consistent with workload.sql, where each line contains a query followed by || separators.
    """
    global_predicate_map = defaultdict(set)

    def _split_and(expr):
        # return list of conjunctive subexpressions
        if isinstance(expr, sqlglot.exp.And):
            return _split_and(expr.this) + _split_and(expr.expression)
        return [expr]

    with open(sql_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Assume format is query||...||
            query = line.split('||')[0].strip()
            if not query.endswith(';'):
                query += ';'
            try:
                parsed = sqlglot.parse_one(query, read="postgres")
                # build alias -> real table name map
                alias_map = {}
                for table_expr in parsed.find_all(sqlglot.exp.Table):
                    alias = table_expr.alias_or_name
                    alias_map[alias] = table_expr.name
                # examine existing WHERE
                where = parsed.find(sqlglot.exp.Where)
                if where and where.this is not None:
                    terms = _split_and(where.this)
                    table_preds = defaultdict(list)
                    for term in terms:
                        # gather distinct aliases referenced in this term
                        refs = {col.table for col in term.find_all(sqlglot.exp.Column) if col.table}
                        if len(refs) == 1:  # non-join predicate
                            alias = list(refs)[0]
                            real_name = alias_map.get(alias, alias)
                            term_str = term.sql(dialect="postgres")
                            # Remove alias from predicate
                            pattern = fr"\b{alias}\."
                            term_clean = re.sub(pattern, "", term_str).strip()
                            table_preds[real_name.lower()].append(term_clean)
                    for real_name, preds in table_preds.items():
                        if preds:
                            combined_pred = " AND ".join(preds)
                            global_predicate_map[real_name.lower()].add(combined_pred)
            except Exception as e:
                print(f"Error parsing query: {e}")
                continue

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
            # 把real转化成小写，因为在collect_predicates中是小写的
            preds = predicates_map.get(real.lower(), [])
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
            preds = predicates_map.get(real.lower(), [])
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
    queries = []
    with open(sql_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                query = line.split('||')[1].strip()
                if query and not query.startswith('--'):
                    queries.append(query)

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
    predicate_sql_file = "/data2/xuyining/PRICE/datas/workloads/finetune/stats/workloads.sql"  # Input SQL file with queries in workload.sql format
    output_file = "./train/train.sql"
    base_sql_file = "/home/xuyining/End-to-End-CardEst-Benchmark/workloads/stats_CEB/stats_CEB.sql"  # Base SQL file to modify (can be same as predicate_sql_file)
    k = 100  # Number of variants per query

    predicates = collect_predicates(predicate_sql_file)
    print("Collected predicates:")
    for table, preds in predicates.items():
        print(f"  {table}: {len(preds)} predicates")

    generate_variants(base_sql_file, predicates, k=k, output_file=output_file)
    print(f"Generated variants saved to {output_file}")