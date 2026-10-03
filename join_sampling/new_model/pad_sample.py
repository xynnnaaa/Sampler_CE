import glob
import sys
import os
import json
import random
import time
import hashlib


from join_sampler_linear import JoinSampler


class SamplePadder(JoinSampler):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _fetch_random_root_ids(self, real_name: str, limit: int = 5000):
        """批量获取 Root 表的随机 ID"""
        try:
            sql = f"SELECT id FROM {real_name} ORDER BY RANDOM() LIMIT {limit};"
            self.cursor.execute(sql)
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching random IDs for {real_name}: {e}")
            return []

    def perform_pure_wander_join(self, join_tree, root_table, root_pool):
        """
        执行一次纯粹的 Wander Join 随机游走，试图生成一条完整的 Tuple。
        join_tree 即线性执行计划列表，root_table 是根表的别名。
        """
        root_step = join_tree[0]
        root_real = root_step['real_name']
        root_sels = root_step['sels']

        max_retries = 50000

        for _ in range(max_retries):
            # 1. 维护 Root ID 资源池
            if not root_pool:
                new_ids = self._fetch_random_root_ids(root_real, limit=5000)
                if not new_ids:
                    print(f"        [Warning] Could not fetch root IDs for {root_table}")
                    return None
                root_pool.extend(new_ids)

            start_id = root_pool.pop()

            # 2. 获取 Root 元组
            neighbors, _ = self.engine._batch_fetch_neighbors(
                root_real, "id", [start_id], root_sels, root_table
            )
            if not neighbors or str(start_id) not in neighbors or not neighbors[str(start_id)]:
                continue

            current_tuple = neighbors[str(start_id)][0].copy()
            success = True

            # 3. 顺着 Join Tree 向下随机游走
            for step in join_tree[1:]:
                alias = step['alias']
                real_name = step['real_name']
                parent_alias = step['parent']
                raw_cond = step['join_condition']
                sel_cols = step.get('sels', [])

                my_col, parent_col = self.engine._parse_cond(raw_cond, alias, parent_alias)
                if not my_col:
                    success = False
                    break

                parent_key = f"{parent_alias}.{parent_col}"
                parent_val = current_tuple.get(parent_key)
                if parent_val is None or parent_val == 'None':
                    success = False
                    break

                next_neighbors, _ = self.engine._batch_fetch_neighbors(
                    real_name, my_col, [parent_val], sel_cols, alias
                )
                candidates = next_neighbors.get(str(parent_val), [])
                if not candidates:
                    success = False
                    break

                chosen = random.choice(candidates)
                current_tuple.update(chosen)

            if success:
                return current_tuple

        print(f"        [Warning] Failed to find a valid join path after {max_retries} attempts.")
        return None

    def _pad_single_file(self, input_json_path: str, output_json_path: str, target_size: int):
        """处理单个 JSON 文件的核心逻辑"""
        try:
            with open(input_json_path, 'r') as f:
                samples_data = json.load(f)
        except Exception as e:
            print(f"        [Error] reading {input_json_path}: {e}")
            return

        total_templates = len(samples_data)
        for idx, (template_id, k_bitmaps) in enumerate(samples_data.items()):
            if template_id not in self.template_by_id:
                print(f"        [Warning] Template {template_id} not found in workload. Skipping.")
                continue

            template_key, template_data = self.template_by_id[template_id]
            join_graph = template_data['graph']
            aliases = list(template_key[0])

            self.add_sel_info_to_graph(join_graph)

            try:
                join_execution_plan = self.build_join_tree_structure(join_graph, aliases)
                root_table = join_execution_plan[0]['alias']
            except Exception as e:
                print(f"        [Error] Building join tree failed: {e}")
                continue

            for b_idx, bitmap in enumerate(k_bitmaps):
                if not bitmap:
                    sorted_aliases = sorted(aliases)
                    header = [f"{alias}.id" for alias in sorted_aliases]
                    bitmap.append(header)
                else:
                    header = bitmap[0]

                current_size = len(bitmap) - 1
                if current_size >= target_size:
                    print(f"        Template {template_id}... Bitmap {b_idx+1}: Already has {current_size} tuples. Skipping padding.")
                    continue

                needed = target_size - current_size
                print(f"        Template {template_id}... Bitmap {b_idx+1}: Found {current_size}. Padding {needed}...")

                root_pool = []
                padded_count = 0
                start_time = time.time()

                try_cont = 0
                while len(bitmap) - 1 < target_size and try_cont < needed * 10:
                    tuple_dict = self.perform_pure_wander_join(join_execution_plan, root_table, root_pool)
                    try_cont += 1
                    if tuple_dict:
                        row = [tuple_dict.get(col, None) for col in header]
                        bitmap.append(row)
                        padded_count += 1
                    else:
                        continue

                if len(bitmap) - 1 < target_size:
                    print(f"            [Warning] Only padded {len(bitmap) - 1 - current_size} tuples after {try_cont} attempts.")
                print(f"            -> Padded {padded_count} tuples in {time.time() - start_time:.2f}s.")

        def default_serializer(obj):
            import decimal
            import datetime
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return str(obj)

        try:
            with open(output_json_path, 'w') as f:
                json.dump(samples_data, f, default=default_serializer, indent=2)
        except Exception as e:
            print(f"        [Error] saving to {output_json_path}: {e}")

    def process_worker_directory(self, results_dir: str, output_base_dir: str, worker_id: int, target_size: int = 100):
        """Worker 处理其专属子目录下的所有 JSON 文件，并输出到新的目录"""
        worker_in_dir = os.path.join(results_dir, str(worker_id))
        worker_out_dir = os.path.join(output_base_dir, str(worker_id))

        if not os.path.exists(worker_in_dir):
            print(f"Worker {worker_id}: Input directory {worker_in_dir} does not exist. Exiting.")
            return

        if not os.path.exists(worker_out_dir):
            os.makedirs(worker_out_dir)
            print(f"Worker {worker_id}: Created output directory {worker_out_dir}")

        json_files = glob.glob(os.path.join(worker_in_dir, "*.json"))
        target_files = [f for f in json_files if not f.endswith("_padded.json")]

        if not target_files:
            print(f"Worker {worker_id}: No valid JSON files found to process.")
            return

        print(f"Worker {worker_id}: Loading workload parsing logic...")
        self.load_and_parse_workload()
        if not self.temp_template_data:
            print("No join templates found. Check your SQL workload file.")
            return

        # 构建 template_id 字符串 -> (template_key, template_data) 的映射
        self.template_by_id = {}
        for tmpl_key, tmpl_data in self.temp_template_data.items():
            aliases_tuple = tmpl_key[0]
            sig_hash = hashlib.md5(tmpl_key[1].encode('utf-8')).hexdigest()[:6]
            tid = f"{'_'.join(aliases_tuple)}_{sig_hash}"
            self.template_by_id[tid] = (tmpl_key, tmpl_data)

        print(f"Worker {worker_id}: Found {len(target_files)} files to process in {worker_in_dir}")
        for input_json in target_files:
            base_name = os.path.basename(input_json)
            new_name = base_name.replace(".json", "_padded.json")
            output_json = os.path.join(worker_out_dir, new_name)

            print(f"\nWorker {worker_id}: Processing file {base_name} ...")
            self._pad_single_file(input_json, output_json, target_size)
            print(f"Worker {worker_id}: Finished saving {output_json}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python pad_samples.py <config_path> <input_results_dir> <output_base_dir> <worker_id> [target_size]")
        sys.exit(1)

    config_path = sys.argv[1]
    results_dir = sys.argv[2]
    output_base_dir = sys.argv[3]
    worker_id = int(sys.argv[4])

    target_size = 1000
    if len(sys.argv) >= 6:
        target_size = int(sys.argv[5])

    padder = SamplePadder(config_path)
    try:
        padder.process_worker_directory(results_dir, output_base_dir, worker_id, target_size=target_size)
    finally:
        padder.close()