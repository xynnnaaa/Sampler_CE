import json

def generate_candidate_info(samples_path: str, output_path: str):
    """
    Reads a samples.json file and generates a candidate_samples_info.json file.

    The output file maps each table name to a list of its available sample indices,
    which is required for the 'all_hit' embedding strategy.

    Args:
        samples_path (str): Path to the input samples.json file.
        output_path (str): Path for the output candidate_samples_info.json file.
    """
    print(f"Reading samples from: {samples_path}")
    
    try:
        with open(samples_path, 'r') as f:
            samples_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{samples_path}'. Please check the path.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: The file '{samples_path}' is not a valid JSON file.")
        return

    print("Generating candidate sample info...")
    candidate_info = {}

    # 遍历 samples.json 中的每一个表
    for table_name, samples_list in samples_data.items():
        # 获取为该表实际生成的样本数量
        # samples_list 是一个列表，其长度就是 k_actual
        k_actual = len(samples_list)
        
        # 只有当实际生成的样本数大于0时，才将其加入到候选信息中
        if k_actual > 0:
            # 生成一个从 0 到 k_actual-1 的索引列表
            candidate_info[table_name] = list(range(k_actual))
            print(f"  - Table '{table_name}': Found {k_actual} samples (indices 0 to {k_actual - 1}).")
        else:
            print(f"  - Table '{table_name}': Found 0 samples. It will be excluded from the candidate info.")

    print(f"\nSaving candidate sample info to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            # 使用 indent=4 参数使输出的 JSON 文件格式美观，易于阅读
            json.dump(candidate_info, f, indent=4)
        print("Successfully generated candidate_samples_info.json!")
    except IOError as e:
        print(f"ERROR: Could not write to output file '{output_path}'. Reason: {e}")


if __name__ == '__main__':
    # --- 配置 ---
    # 你只需要修改这两个文件名即可
    input_samples_file = "samples.json"
    output_candidate_file = "candidate_samples_info.json"
    
    # --- 运行 ---
    generate_candidate_info(input_samples_file, output_candidate_file)