> 此文记录修复前的问题。共用解析器已修复：仅归一化未加引号的标识符，保留字符串值。修复后全量检查见 `fixed/report.md`。

# PostgreSQL 只读验证

已连接 `localhost:5432` 的 `tpch-skew-10`，用户 `xuyining`。使用只读事务，未修改数据或 anno 表。

| 表 | 原始谓词 | 匹配行数 | 当前解析后的谓词 | 匹配行数 |
|---|---|---:|---|---:|
| part | `p_brand='Brand#35'` | 68,988 | `p_brand='brand#35'` | 0 |
| customer | `c_mktsegment='MACHINERY'` | 300,441 | `c_mktsegment='machinery'` | 0 |

同一元组执行原建表逻辑的 CASE 表达式：

| 表与 ID | 原始条件产生的位 | 小写条件产生的位 |
|---|---|---|
| part，388310，`Brand#35` | 1 | 0 |
| customer，4，`MACHINERY` | 1 | 0 |
| lineitem，1，`TRUCK` | 1 | 0 |

这三列实际类型均为 `character(10)`。值存在尾部填充空格，但上述 SQL 比较使用列原类型，原始谓词能匹配；大小写变化确实造成匹配差异。

V2 使用的实际 extract_join_graph 对 train.sql 第一条查询返回：

```text
tpch_p: tpch_p.p_brand = 'brand#35'
tpch_c: tpch_c.c_mktsegment = 'machinery'
```

原建表流程为：extract_join_graph → node.predicates → 去别名/排序/AND 合并 → global_predicate_map → CASE WHEN pred_sql THEN B'1' ELSE B'0' END。当前共用解析器将整个 SQL 转小写，因此以当前代码重新构建 anno 确定会将以上正确匹配位写为 0。V2 直接构建 QID 位图也会继承相同错误。

本次验证了当前代码链路与真实数据的差异，没有逐 PID 核验现存全局 anno 表。历史 anno 是否受到相同影响，还取决于它创建时使用的解析器版本、workload 和 PID 编号，不能仅凭当前源码认定全部历史表的实际内容。
