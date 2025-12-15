import json
import psycopg2
import os
import sys
from typing import List, Dict, Any
from sqlglot import parse_one, exp
from sqlglot.expressions import Table, Where, Column
from collections import defaultdict
import random
import glob
import pickle

class Sampler:
    def __init__(self, config_path: str):
        print("Initializing Sampler...")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        print("Connecting to the database...")
        self.conn = psycopg2.connect(
            dbname=self.config["db"]["name"],
            user=self.config["db"]["user"],
            password=self.config["db"]["password"],
            host=self.config["db"]["host"],
            port=self.config["db"]["port"],
        )
        self.cursor = self.conn.cursor()

        self.base_query_dir = self.config["sampling"]["base_query_dir"]
        self.k_bitmaps = self.config["sampling"]["k_bitmaps"]
        self.m_partitions = self.config["sampling"]["m_partitions"]
        self.primary_key = self.get_primary_key()
        self.samples: Dict[str, List[List[Any]]] = defaultdict(list)
        self.query_counter = 0

        try:
            template_names = [d for d in os.listdir(self.base_query_dir) if os.path.isdir(os.path.join(self.base_query_dir, d))]
        except FileNotFoundError:
            print(f"ERROR: Base query directory not found: {self.base_query_dir}")
            return
        
        if not template_names:
            print(f"No template subdirectories found in '{self.base_query_dir}'.")
            return
        
        self.workload = []
        for template_name in sorted(template_names):
            input_template_dir = os.path.join(self.base_query_dir, template_name)
            pkl_files = sorted(glob.glob(os.path.join(input_template_dir, "*.pkl")))
            if not pkl_files:
                print(f"No .pkl files found in '{input_template_dir}'. Skipping this template.")
                continue
            for pkl_file in pkl_files:
                with open(pkl_file, "rb") as f:
                    query_data = pickle.load(f)
                    sql_query = query_data.get("sql")
                    # query_name = os.path.basename(pkl_file)
                    if sql_query:
                        self.workload.append(sql_query)

        print("Initialization complete.")
        print(f"Loaded {len(self.workload)} queries from workload.")

    def get_primary_key(self) -> Dict[str, str]:
        """
        Fetches the primary key for each table, excluding samples.
        Also validates that all primary keys are of a numeric type.
        If a non-numeric PK is found, the program exits.
        """
        allowed_numeric_types = {
            'smallint', 'integer', 'bigint',
            'decimal', 'numeric',
            'real', 'double precision',
            'smallserial', 'serial', 'bigserial'
        }

        pk_map = {}
        
        query = """
            SELECT
                tc.table_name,
                kcu.column_name,
                c.data_type
            FROM
                information_schema.table_constraints AS tc
            JOIN 
                information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN
                information_schema.columns AS c
                ON c.table_schema = tc.table_schema 
                AND c.table_name = tc.table_name 
                AND c.column_name = kcu.column_name
            WHERE 
                tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name NOT LIKE %s
                AND tc.table_name NOT LIKE %s;
        """
        
        try:
            self.cursor.execute(query, ('%_s70', '%_s100'))
            results = self.cursor.fetchall()

            is_all_numeric = True

            for table_name, column_name, data_type in results:
                if data_type not in allowed_numeric_types:
                    print(f"Error: Table '{table_name}' has a primary key column '{column_name}' of non-numeric type '{data_type}'.")
                    is_all_numeric = False
                    continue

                print(f"Table '{table_name}' has primary key column '{column_name}' of type '{data_type}'.")
                pk_map[table_name] = column_name

            if not is_all_numeric:
                print("Primary key validation failed due to non-numeric types. Exiting.")
                self.close()
                sys.exit(1)
            
            print(f"Primary key validation successful. Found {len(pk_map)} tables with numeric primary keys (after filtering).")
            return pk_map

        except Exception as e:
            print(f"An error occurred while fetching or validating primary keys: {e}")
            self.close()
            sys.exit(1)

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Executes a SQL query and returns the results."""
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_table_alias_map(self, expression: exp.Expression) -> Dict[str, str]:
        """Extracts table aliases from the parsed SQL expression."""
        table_alias_map = {}
        for table in expression.find_all(exp.Table):
            table_name = table.name
            alias = table.alias_or_name
            table_alias_map[alias] = table_name
        return table_alias_map

    def parse_workload(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parses the SQL query to extract tables and their conditions.

        Returns:
        {
            table_name: [
                { "query_id": "...", "original_text": "...", "table_name": "...", "predicate_sql": "..." },
                ...
            ],
            ...
        }
        """

        print("Parsing workload queries...")

        all_parsed_queries : Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for query in self.workload:
            self.query_counter += 1
            # query_id = f"query_{self.query_counter}"
            try:
                expression = parse_one(query, read="postgres")
                table_alias_map = self.get_table_alias_map(expression)

                where_clause = expression.find(exp.Where)
                alias_specific_filters: Dict[str, List[exp.Expression]] = defaultdict(list)

                if where_clause:
                    conjuncts = list(where_clause.this.flatten()) if isinstance(where_clause.this, exp.And) else [where_clause.this]
                    for condition in conjuncts:
                        involved_table_aliases = set()
                        for col_name in condition.find_all(exp.Column):
                            alias = col_name.table
                            if alias and alias not in table_alias_map: # 处理表名直接用作别名的情况
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
                            combined_conditions = exp.and_(*conditions) if len(conditions) > 1 else conditions[0]
                            combined_conditions_no_alias = combined_conditions.transform(
                                lambda node: exp.Column(this=node.this) if isinstance(node, exp.Column) and node.table else node
                            )
                            predicate_sql = combined_conditions_no_alias.sql(dialect="postgres")
                            all_parsed_queries[table_name].append({
                                # "query_id": query_id,
                                # "original_text": query,
                                "table_name": table_name,
                                "predicate_sql": predicate_sql
                            })

            except Exception as e:
                print(f"Error parsing query: {query}\nException: {e}")

        print("\nDeduplicating predicates for each table...")
        unique_queries_by_table : Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_raw_predicates = 0
        total_unique_predicates = 0

        for table, queries in all_parsed_queries.items():
            seen_predicates = set()
            total_raw_predicates += len(queries)

            for q in queries:
                predicate = q["predicate_sql"]
                if predicate not in seen_predicates:
                    seen_predicates.add(predicate)
                    unique_queries_by_table[table].append(q)

            total_unique_predicates += len(unique_queries_by_table[table])
            print(f"Table '{table}': {len(queries)} raw predicates -> {len(unique_queries_by_table[table])} unique predicates.")

        print(f"Workload parsed. Found relevant queries for {len(all_parsed_queries)} tables.")
        return unique_queries_by_table


    def partition_table(self, table: str) -> tuple[bool, list]:
        """Partitions the table into m partitions based on the primary key."""
        primary_key_col = self.primary_key.get(table)
        if not primary_key_col:
            raise ValueError(f"No primary key found for table {table}")
        self.cursor.execute(f"SELECT {primary_key_col} FROM {table} ORDER BY {primary_key_col}")
        row_ids = [row[0] for row in self.cursor.fetchall()]
        n = len(row_ids)

        if n < self.m_partitions:
            print(f"Info: Table '{table}' size ({n}) is smaller than m_partitions ({self.m_partitions}). "
                  f"Skipping sampling and using all IDs.")
            return True, row_ids

        base_size, remainder = divmod(n, self.m_partitions)
        partitions = []
        start = 0
        for i in range(self.m_partitions):
            size = base_size + (1 if i < remainder else 0)
            partitions.append(row_ids[start:start + size])
            start += size
        return False, partitions

    def select_best_tuple_from_partition(
        self,
        table: str,
        partition_ids: List[int],
        relevant_queries: List[Dict[str, Any]],
        covered_queries: set
    ) -> Any:
        if not partition_ids:
            return None
            
        uncovered_queries_with_indices = [(i, q) for i, q in enumerate(relevant_queries) if i not in covered_queries]
        
        if not uncovered_queries_with_indices:
            return random.choice(partition_ids)

        from collections import Counter
        tuple_scores = Counter()
        BATCH_SIZE = 100
        
        primary_key_col = self.primary_key[table]
        partition_ids_str = ', '.join(map(str, partition_ids))

        for i in range(0, len(uncovered_queries_with_indices), BATCH_SIZE):
            batch = uncovered_queries_with_indices[i:i + BATCH_SIZE]
            
            sum_clauses = []
            for index, query in batch:
                # safe_predicate = query['predicate_sql'].replace("'", "''")
                sum_clauses.append(f"CASE WHEN ({query['predicate_sql']}) THEN 1 ELSE 0 END")

            if not sum_clauses:
                continue

            score_calculation = " + ".join(sum_clauses)
            
            batch_query = f"""
                SELECT {primary_key_col}, ({score_calculation}) as score
                FROM {table}
                WHERE {primary_key_col} IN ({partition_ids_str})
            """

            print(batch_query)
            
            try:
                self.cursor.execute(batch_query)
                for key_id, score in self.cursor.fetchall():
                    if score > 0:
                        tuple_scores[key_id] += score
            except Exception as e:
                print(f"Error executing batch scoring query on table {table}: {e}")
                self.conn.rollback()
                continue

        if tuple_scores:
            best_tuple = tuple_scores.most_common(1)[0][0]
            print(f"    selected tuple covers {tuple_scores[best_tuple]} new queries.")
            return best_tuple
        else:
            return random.choice(partition_ids)


    def update_covered_queries(
        self,
        table: str,
        selected_tuple: Any,
        relevant_queries: List[Dict[str, Any]],
        covered_queries: set
    ):
        """Updates the set of covered queries based on the selected tuple."""
        if not selected_tuple:
            return

        uncovered_queries_with_indices = [(i, q) for i, q in enumerate(relevant_queries) if i not in covered_queries]
        if not uncovered_queries_with_indices:
            return
        
        BATCH_SIZE = 100
        
        pk_col = self.primary_key[table]
        selected_tuple_str = str(selected_tuple)
        newly_covered_indices = set()

        for i in range(0, len(uncovered_queries_with_indices), BATCH_SIZE):
            batch = uncovered_queries_with_indices[i:i + BATCH_SIZE]
            
            union_all_clauses = []
            for index, query in batch:
                union_all_clauses.append(
                    f"""
                    SELECT {index} AS result_index
                    WHERE EXISTS (
                        SELECT 1 FROM {table}
                        WHERE {pk_col} = {selected_tuple_str} AND ({query['predicate_sql']})
                    )
                    """
                )
            if not union_all_clauses:
                continue

            batch_query = " UNION ALL ".join(union_all_clauses)

            try:
                self.cursor.execute(batch_query)
                results = self.cursor.fetchall()
                for (result_index,) in results:
                    newly_covered_indices.add(result_index)
            except Exception as e:
                print(f"Error executing coverage update query on table {table}: {e}")
                self.conn.rollback()
                continue
        
        if newly_covered_indices:
            covered_queries.update(newly_covered_indices)

    def sample(self):
        """Main sampling function that orchestrates the sampling process."""
        parsed_workload = self.parse_workload()
        for table, queries in parsed_workload.items():
            print(f"Sampling for table {table} with {len(queries)} relevant queries...")
            self.samples[table] = []
            is_small, partitions = self.partition_table(table)
            if is_small:
                print(f"Table {table} is small. Skipping sampling.")
                continue
            
            print(f"Table {table} partitioned into {len(partitions)} partitions.")
            covered_queries = set()

            num_total_queries = len(queries)

            for j in range(self.k_bitmaps):
                print(f"Constructing bitmap {j+1}/{self.k_bitmaps} for table {table}...")

                if len(covered_queries) == num_total_queries:
                    print(f"All queries already covered. Ending early at bitmap {j+1}.")
                    break

                current_bitmap = []
                for i, partition in enumerate(partitions):
                    num_covered_before = len(covered_queries)

                    print(f"  Processing partition {i+1}/{len(partitions)} with {len(partition)} tuples...")
                    print(f"    Covered queries before selection: {num_covered_before}/{num_total_queries} ({num_covered_before/num_total_queries if num_total_queries > 0 else 0:.1%})")

                    selected_tuple = self.select_best_tuple_from_partition(
                        table, partition, queries, covered_queries
                    )
                    if selected_tuple:
                        current_bitmap.append(selected_tuple)
                        print(f"    Selected tuple: {selected_tuple}")
                        # partition.remove(selected_tuple)
                        self.update_covered_queries(table, selected_tuple, queries, covered_queries)
                        num_covered_after = len(covered_queries)
                        print(f"    Covered queries after selection: {num_covered_after}/{num_total_queries} ({num_covered_after/num_total_queries if num_total_queries > 0 else 0:.1%})")
                    else:
                        print(f"    Warning: No tuple selected for this partition.")

                self.samples[table].append(current_bitmap)
                print(f"--- Bitmap {j+1} for '{table}' constructed. Final size: {len(current_bitmap)} ---")
    
    def save_samples(self, output_path: str):
        """Saves the sampled data to a JSON file."""
        print(f"Saving samples to {output_path}...")
        with open(output_path, 'w') as file:
            json.dump(self.samples, file, indent=4)
        print("Samples saved.")

    def close(self):
        """Closes the database connection."""
        self.cursor.close()
        self.conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    sampler = Sampler("sampler_config.json")
    try:
        sampler.sample()
        sampler.save_samples("samples_new.json")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sampler.close()