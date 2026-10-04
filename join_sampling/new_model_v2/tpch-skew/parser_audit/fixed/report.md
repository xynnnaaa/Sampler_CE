# 谓词大小写修复后的回归验证

修复位置：`mscn/query_representation/utils.py::extract_join_graph()`。

不再对整个 SQL 执行 `.lower()`。先解析原 SQL，再仅将未加双引号的 `exp.Identifier` 节点归一化为小写；字符串字面量和带引号标识符保持原始大小写。没有改变谓词抽取范围、template 签名规则或采样参数与选择逻辑。

## 全量检查

当前 train.sql 共 44,118 条查询，文件指纹与修复前检查一致。执行 V2 实际 `load_and_parse_workload()`，逐条与保留字符串值的参考 AST 对照：

| 检查项 | 修复后 |
|---|---:|
| 已核对查询 | 44,118 |
| SQL 解析异常 | 0 |
| 本地谓词差异 | 0 |
| 字符串值变化 | 0 |
| join template | 164，与参考结果一致 |
| 错误合并、拆分 | 0 |
| 独立连接约束丢失 | 0 |

修复前有 32,057 条查询的字符串值被转成小写，修复后不再发生。

## 定向测试及数据库回归

`mscn/query_representation/tests/test_identifier_case.py` 的 5 项 unittest 全部通过，覆盖 TPCH 字符串值、混合大小写的未引用别名、带引号标识符、转义单引号/IN/LIKE 字符串、数值谓词和无过滤条件 alias。

通过用户已授权的 PostgreSQL 只读连接，使用修复后实际解析出的条件执行原 anno CASE 表达式：

- part 的 id=388310：`p_brand = 'Brand#35'`，位值为 1。
- customer 的 id=4：`c_mktsegment = 'MACHINERY'`，位值为 1。

这两条记录在修复前的小写条件下都产生 0。

本次没有修改数据库数据，没有删除或重建现存 anno 表。旧全局 PID anno 不会自动修复；原建表流程会跳过已有表，继续运行 V1 时需显式重新构建并保持 workload/PID 编号一致。V2 按 template 重新构建内存 QID 位图，直接使用修复后的谓词。

本次验证针对该 workload 及大小写回归，不代表 OR/NOT/BETWEEN/显式 JOIN 等其他既有解析范围问题已经修复。

运行测试：

```bash
cd /home/Sampler_CE
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ceb_env/bin/python -m unittest discover \
    -s mscn/query_representation/tests -p 'test_identifier_case.py' -v
```

完整统计见 `summary.json`，`issues.csv` 仅含表头（没有差异记录）。
