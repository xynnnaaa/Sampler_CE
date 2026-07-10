#!/usr/bin/env python3
import re
import os
import glob

def extract_times(log_file_path):
    """
    从 log 文件中提取 t1 和 t2。
    t1: 行 'Parsed templates across all queries in XXX.XXs.' 中的时间
    t2: 文件倒数第二行 'Worker <编号> finished in YYY.YYs' 中的时间（编号任意）
    返回 (t1, t2) 浮点数，若提取失败则返回 (None, None)
    """
    t1 = None
    t2 = None

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取 t1（固定文本）
    t1_pattern = re.compile(r'Parsed templates across all queries in ([\d.]+)s\.')
    for line in lines:
        match = t1_pattern.search(line)
        if match:
            t1 = float(match.group(1))
            break

    # 提取 t2：匹配任意 Worker 编号的 finished 行，从末尾开始找第一个非空匹配行
    t2_pattern = re.compile(r'Worker \d+ finished in ([\d.]+)s')
    for line in reversed(lines):
        line = line.strip()
        if not line:          # 跳过空行
            continue
        match = t2_pattern.search(line)
        if match:
            t2 = float(match.group(1))
            break

    return t1, t2

def main():
    # 目标目录（可修改为实际路径）
    base_dir = '/home/Sampler_CE/join_sampling/new_model/linear/ergastf1/runfile'
    log_files = glob.glob(os.path.join(base_dir, '*.log'))

    if not log_files:
        print(f"未在 {base_dir} 下找到任何 .log 文件")
        return

    t1_list = []
    t2_list = []

    for fpath in log_files:
        t1, t2 = extract_times(fpath)
        if t1 is None or t2 is None:
            print(f"警告：文件 {fpath} 缺少 t1 或 t2，跳过")
            continue
        t1_list.append(t1)
        t2_list.append(t2)

    if not t1_list:
        print("没有成功提取任何数据")
        return

    sum_t1 = sum(t1_list)
    sum_t2 = sum(t2_list)
    mean_t1 = sum_t1 / len(t1_list)

    result = sum_t2 - sum_t1 + mean_t1

    print(f"文件数: {len(t1_list)}")
    print(f"sum(t1) = {sum_t1:.4f} s")
    print(f"sum(t2) = {sum_t2:.4f} s")
    print(f"mean(t1) = {mean_t1:.4f} s")
    print(f"结果 = sum(t2) - sum(t1) + mean(t1) = {result:.4f} s")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# import re
# import os
# import glob

# def extract_t2(log_file_path):
#     """
#     从 log 文件中提取倒数第二行（或附近）的 Worker 完成时间。
#     返回浮点数 t2，若未找到则返回 None。
#     """
#     with open(log_file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     # 匹配任意 Worker 编号的 finished 行
#     pattern = re.compile(r'Worker \d+ finished in ([\d.]+)s')
#     for line in reversed(lines):
#         line = line.strip()
#         if not line:
#             continue
#         match = pattern.search(line)
#         if match:
#             return float(match.group(1))
#     return None

# def main():
#     base_dir = '/home/Sampler_CE/join_sampling/new_model/linear/ergastf1/runfile'
#     log_files = glob.glob(os.path.join(base_dir, '*.log'))

#     if not log_files:
#         print(f"未在 {base_dir} 下找到任何 .log 文件")
#         return

#     t2_list = []
#     for fpath in log_files:
#         t2 = extract_t2(fpath)
#         if t2 is None:
#             print(f"警告：文件 {fpath} 缺少 t2，跳过")
#             continue
#         t2_list.append(t2)

#     if not t2_list:
#         print("没有成功提取任何 t2 数据")
#         return

#     total_t2 = sum(t2_list)
#     print(f"处理文件数: {len(t2_list)}")
#     print(f"所有 t2 的总和 = {total_t2:.4f} s")

# if __name__ == "__main__":
#     main()