import psycopg2
import pickle
import os
import sys
from pathlib import Path
import glob
import sys
print(sys.path)

from query_representation.query import load_qrep, get_tables, get_predicates


DB_CONNECT_PARAMS = {
    "host": "localhost",
    "port": 5434,
    "dbname": "imdb",
    "user": "imdb"
}

QREP_BASE_DIR = "./queries/ceb-imdb"
OUTPUT_DIR = "./queries/sample_bitmaps/ceb-imdb"

SAMPLE_SIZE = 70
SAMPLE_TABLE_PREFIX = "sample_"

PRIMARY_KEY_COL = "id"  # assuming every table has a primary key column named 'id'

def execute_query(conn, query):
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return [item[0] for item in cur.fetchall()]
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    
def check_tables_for_pk(conn, tables):
    tables_missing_pk = []
    for table in tables:
        query = f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = '{table}' AND column_name = '{PRIMARY_KEY_COL}';
        """
        result = execute_query(conn, query)
        if result is None or result[0] == 0:
            tables_missing_pk.append(table)
            print(f"Table {table} is missing primary key column '{PRIMARY_KEY_COL}'.")

    if tables_missing_pk:
        print("Some tables are missing the primary key column.")
        sys.exit(1)
    else:
        print(f"All tables have the primary key column '{PRIMARY_KEY_COL}'.")
    return tables_missing_pk

def create_sample_tables(conn, tables):
    for table in tables:
        sample_table = SAMPLE_TABLE_PREFIX + table + "_" + str(SAMPLE_SIZE)
        check_exists_sql = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{sample_table}');"
        exists = execute_query(conn, check_exists_sql)
        if exists and exists[0]:
            print(f"Sample table {sample_table} already exists. Skipping creation.")
            continue
        create_sql = f"""
        CREATE TABLE {sample_table} AS
            SELECT * FROM {table}
            ORDER BY RANDOM()
            LIMIT {SAMPLE_SIZE};
        """
        execute_query(conn, create_sql)
        conn.commit()
        print(f"Created sample table {sample_table}.")
    print("All Sample tables creation completed.")


def main():
    import sys
    print(sys.path)

    total_zero_bitmaps = 0
    total_bitmaps = 0
    total_bitmaps_with_filter = 0
    success_queries = 0
    total_queries = 0

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        conn = psycopg2.connect(**DB_CONNECT_PARAMS)
        print("Connected to the database successfully.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qrep_files = []
    if not os.path.isdir(QREP_BASE_DIR):
        print(f"QREP base directory {QREP_BASE_DIR} does not exist.")
        sys.exit(1)

    template_names = [d for d in os.listdir(QREP_BASE_DIR) if os.path.isdir(os.path.join(QREP_BASE_DIR, d))]
    for template_name in sorted(template_names):
        template_dir = os.path.join(QREP_BASE_DIR, template_name)
        pkl_files_in_template = glob.glob(os.path.join(template_dir, "*.pkl"))
        qrep_files.extend(pkl_files_in_template)
    
    qrep_files = [Path(f) for f in qrep_files]
    print(f"Found {len(qrep_files)} QREP files to process.")

    all_unique_tables = [
        "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type",
        "company_name", "company_type", "complete_cast", "info_type", "keyword",
        "kind_type", "link_type", "movie_companies", "movie_info", "movie_info_idx",
        "movie_keyword", "movie_link", "name", "person_info", "role_type", "title"
    ]

    check_tables_for_pk(conn, all_unique_tables)

    create_sample_tables(conn, all_unique_tables)

    for qrep_file in qrep_files:
        total_queries += 1
        try:
            qrep = load_qrep(str(qrep_file))

            preds_by_alias = {}
            tables_by_alias = {} # alias to table name mapping

            for alias, node_info in qrep["join_graph"].nodes(data=True):
                table_name = node_info["real_name"]
                tables_by_alias[alias] = table_name
                predicate_list = node_info.get("predicates", [])
                preds_by_alias[alias] = [pred.strip() for pred in predicate_list]
            
            current_query_bitmap = {}
            for alias, table_name in tables_by_alias.items():
                sample_table = SAMPLE_TABLE_PREFIX + table_name + "_" + str(SAMPLE_SIZE)
                predicate_list = preds_by_alias.get(alias, [])
                if predicate_list:
                    where_clause = " AND ".join(f"({p})" for p in predicate_list)
                    query = f"""
                    SELECT {alias}.{PRIMARY_KEY_COL} FROM {sample_table} {alias}
                    WHERE {where_clause};
                    """
                    total_bitmaps_with_filter += 1
                else:
                    query = f"SELECT {PRIMARY_KEY_COL} FROM {sample_table} {alias};"
                result_ids = execute_query(conn, query)
                if result_ids == [] or result_ids is None:
                    total_zero_bitmaps += 1
                current_query_bitmap[(alias,)] = {'sb' + str(SAMPLE_SIZE): result_ids if result_ids else []}
                total_bitmaps += 1

            output_fn = Path(OUTPUT_DIR) / (qrep_file.stem + ".pkl")
            with open(output_fn, "wb") as f:
                pickle.dump(current_query_bitmap, f)

            success_queries += 1
        except Exception as e:
            print(f"Error processing {qrep_file}: {e}")
            continue
    
    conn.close()
    print("generation of sample bitmaps completed.")
    print(f"Total queries processed: {total_queries}")
    print(f"Successful queries: {success_queries}")
    print(f"Total bitmaps: {total_bitmaps}")
    print(f"Total bitmaps with filter: {total_bitmaps_with_filter}")
    print(f"Total zero bitmaps: {total_zero_bitmaps}")

if __name__ == "__main__":
    main()
