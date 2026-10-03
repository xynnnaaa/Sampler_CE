#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
from pathlib import Path

# 需要排除的表名列表（全小写，用于不区分大小写匹配）
EXCLUDED_TABLES = [
    "laptimes"
]

def contains_excluded_table(sql: str, excluded_tables) -> bool:
    """
    检查SQL字符串中是否包含任意一个排除的表名（不区分大小写）。
    使用正则 \b 边界匹配，避免误匹配包含表名的字段或别名。
    """
    # 转为小写进行不区分大小写匹配
    sql_lower = sql.lower()
    for table in excluded_tables:
        # 使用正则边界匹配，确保表名作为独立单词出现
        # 注意：表名可能出现在引号、括号等后，\b能正确处理
        if re.search(r'\b' + re.escape(table) + r'\b', sql_lower):
            return True
    return False

def filter_sql_file(input_path: str, output_path: str, excluded_tables=None):
    """
    读取输入文件，过滤查询，写入输出文件。
    """
    if excluded_tables is None:
        excluded_tables = EXCLUDED_TABLES

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"错误：输入文件 {input_path} 不存在")
        sys.exit(1)

    output_file = Path(output_path)
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    kept_lines = 0
    total_lines = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line in fin:
                total_lines += 1
                line_stripped = line.strip()
                if not line_stripped:
                    # 空行直接写入？通常不保留空行，但保留也不影响
                    # 根据需求，可跳过空行。这里选择跳过空行，不输出。
                    continue

                # 解析SQL部分：取 '||' 之前的内容
                parts = line_stripped.split("||", 1)
                sql_str = parts[0].strip() if parts else ""

                # 去除开头的注释（如 /* comment */）
                if sql_str.startswith("/*"):
                    # 找到第一个 "*/" 并取后面的部分
                    if "*/" in sql_str:
                        sql_str = sql_str.split("*/", 1)[-1].strip()
                    else:
                        # 如果注释未闭合，保留整个字符串？但通常不会，这里简单处理
                        pass

                # 判断是否包含排除表
                if not contains_excluded_table(sql_str, excluded_tables):
                    # 不包含任何排除表，保留原始行
                    fout.write(line)  # 保留原始行（含换行）
                    kept_lines += 1

    except Exception as e:
        print(f"处理文件时出错：{e}")
        sys.exit(1)

    print(f"处理完成。总行数（非空）：{total_lines}，保留行数：{kept_lines}")
    print(f"结果已保存至：{output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="过滤SQL查询记录，剔除包含指定表名的查询"
    )
    parser.add_argument("input_file", help="输入的SQL记录文件路径")
    parser.add_argument("-o", "--output", default="filtered_queries.txt",
                        help="输出文件路径（默认：filtered_queries.txt）")
    parser.add_argument("-t", "--tables", nargs="+",
                        default=EXCLUDED_TABLES,
                        help="需要排除的表名列表（空格分隔），默认使用内置列表")
    args = parser.parse_args()

    filter_sql_file(args.input_file, args.output, args.tables)

if __name__ == "__main__":
    main()