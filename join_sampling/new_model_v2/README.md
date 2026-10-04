# JoinSampler V2：当前 template 的内存 QID 位图

本版从 `../new_model/join_sampler_linear.py` 及对应 WanderJoinEngine 的计时版本生成，只替换位图的准备、表示与读取方式。原版目录和 `workloads/join_sampler_sql.py` 未修改。

## 文件

| 文件 | 用途 |
|---|---|
| `join_sampler_linear.py` | 采样主程序，接入当前 template 位图的构建和释放 |
| `wander_join.py` | 保留 linear 流程实际使用的邻居查询和单步随机扩展，直接读取内存 QID 掩码 |
| `template_annotations.py` | 位图构建、紧凑内存存储、ID 定位、多个 worker 的内存预算 |
| `predicate_cache.py` | 按表批量构建谓词命中块、原子发布、文件锁和跨 worker 只读 mmap |
| `cal_time.py` | 解析日志，汇总原始、扣等待后的累计时间和缓存构建时间 |
| `sampling_timing.py` | 原版分阶段计时，增加准备、流式读取、内存查询、释放和进程峰值 RSS 统计 |

未复制数据集、配置、日志、`cal_time.py`、旧的 beam/lookahead 接口或其他辅助程序。主程序仍依赖项目现有的 `mscn.query_representation.utils`，SQL 解析器未修改。

## 保持一致的采样行为

- 原有 workload 解析、template key、PID 分配和 QID 编号顺序。
- 基于硬编码表基数的 root 选择、连接执行计划及 `sels`。
- `ORDER BY RANDOM()` 的 root 分区；分区先生成，然后构建内存位图。
- `m_partitions`、`k_bitmaps`、`w_samples` 的配置和默认值。
- 每批 2000 个 root ID、最多 2000 个 root 候选，原有排序及同分选择顺序。
- 每个 root 复制 `w_samples` 条路径；每一步使用原有 `random.choice()`。
- 邻居 SQL、结果组装、IS NOT NULL 检查和失效路径处理。
- 位图为零的路径仍按照原版继续扩展，本版未增加剪枝。
- `path['acc_bmp'] & qid_mask`、`bin(mask & uncovered_mask).count('1')`。
- 前半分区覆盖选择、后半分区纯随机游走、99% 覆盖切换以及已有 continue 行为。
- template 覆盖掩码跨 Bitmap 累积，随机阶段不更新覆盖率。
- worker 静态任务分配、每 5 个 template 保存一次及 JSON 样本格式。

生成的 SQL 和内存预算不消耗 Python 全局随机数。正常执行时不改变随机选择次数。

## 默认流程：共享谓词缓存 → template QID 位图

默认启用 `predicate_cache_enabled=true`。完整解析 workload 后，按真实表收集并排序不同的完整谓词组合。首次用到一张有过滤条件的表时，一个 worker 获取该表文件锁，以每块最多 64 个谓词批量扫描原表，保存每行一个小端 uint64 的命中块；其他 worker 等待后直接复用。

缓存按运行 ID、数据库配置（不含密码）、真实表、精确谓词 SQL 清单、块大小及格式版本隔离。构建先写临时目录，完成验证并写入 manifest 后原子发布整个目录；缓存文件只读映射。普通退出或强制终止后的半成品不会被读取，后续构建者可回收并重建。

当前 template 只读取需要的谓词块，使用 NumPy 分块组装 `uint64[N, ceil(Q/64)]` QID 矩阵。无谓词查询仍贡献恒定位，NULL 仍为不命中；不同 alias、不同 QID 顺序分别组装。结果仍写入原 `AliasBitmaps.buffer`，采样阶段的查询、交集和评分接口不变。没有逐行遍历全局 PID 的 Python 翻译。

缓存构建在申请 template 位图额度前完成，首次构建及等待均计入当前 template 时间。共享元数据中的 COUNT/MIN/MAX 同时用于后续 template 预分配，不再由各 worker 重复查询。连续 ID 表不排序，稀疏 ID 表排序，并跨谓词块核对 ID 序列。

运行期间基础表及谓词语义必须稳定，不适用于随时间或随机状态改变的谓词。默认每次脚本启动产生新的共同运行 ID，避免重用旧数据缓存。单 worker 自动产生自己的 ID；手动多 worker 启动必须设置相同的新 `JOIN_SAMPLING_CACHE_RUN_ID`，否则报错。详见 [共享缓存改动说明](SHARED_PREDICATE_CACHE.md)。

## 关闭共享缓存时的直接 SQL 构建

每个 template 只为当前 alias 构建位图。解析阶段的 PID 仍用于恢复谓词 SQL 和对重复谓词组合分组，但不会读取或创建全局 PID anno 表，也不会在采样时执行 PID→QID 翻译。

对同一 alias，同一个谓词组合在生成表达式中只出现一次：条件为真时贡献使用该组合的全部 QID 位，否则贡献零。各组合的贡献与无谓词查询的恒定位取 OR。

例如某 alias 的 QID 0、2 使用同一谓词，QID 1 没有谓词，则表达式等价于：

```sql
SELECT id,
       (B'010' |
        CASE WHEN (predicate) THEN B'101' ELSE B'000' END)::text
FROM base_table;
```

真实代码会显式转换到当前查询数宽度的 `bit varying(Q)`。SQL NULL 条件走 ELSE，与原 anno 表的 CASE 语义一致。字符串左侧是 QID Q-1、右侧是 QID 0，转换后严格满足 `bit q = 1 << q`。

完全无过滤条件的 alias 只保存一个常量掩码，不扫描原表、不分配全表位图。

### 扫描与内存峰值

- 使用 PostgreSQL 服务端命名游标和 `fetchmany(annotation_batch_size)` 流式读取。
- 每个批次直接转换为整数、编码到预分配缓冲区，随后释放批次。
- 不对位图全表使用 `fetchall()`，不保留全表字符串列表或 Python 整数字典。
- 为准确预分配，首次需要某张表的位图时用 `COUNT(*), MIN(id), MAX(id)` 获取元数据。结果缓存在当前 sampler/数据库连接内，同一 worker 的后续 template 和同表 alias 复用，不重复查询；不同 worker 各自维护缓存，重启后重新获取。
- 连续 ID 表的谓词扫描取消 `ORDER BY id`，按 `id - first_id` 写入数组，返回顺序不影响最终位图。稀疏 ID 表仍使用 `ORDER BY id` 和排序后的 ID 数组。
- 无序写入时用每行 1 bit 的临时标记检测重复 ID；每次只为正在构建的连续 ID alias 分配，构建结束释放。内存预算额外预留各 alias 中最大的标记大小，`annotations.validation_scratch_bytes` 记录该空间；`annotations.buffer_bytes` 仍只记录最终位图及稀疏 ID 数组。
- 假定采样期间原表数据稳定。重复 ID 或构建期间数量/连续 ID 范围变化会报错并释放资源；本版不改变原版的数据库隔离级别。

元数据缓存不会随 template 位图释放而清空，只保存每表三个数，不缓存位图。运行中若修改基础表，应重新启动 sampler 以刷新缓存；行数和 ID 检查不能代替数据库快照，也不能检测所有内容更新。采样参数、root 随机分区、路径选择、交集和评分逻辑均保持原样。已启动的 worker 需要重启才能使用此修改。

## 紧凑表示与 ID 定位

使用标准库 `bytearray` 连续保存最终位图，布局等价于小端 `uint64[N, ceil(Q/64)]`。共享缓存的批量打包和 QID 组装需要 NumPy（已在原项目运行环境中使用）。

```text
row_bytes = ceil(Q / 64) × 8
buffer_bytes = N × row_bytes
```

每行最后不足 64 位的部分补零。仅在读取选中元组时，通过 `int.from_bytes(memoryview(...), 'little')` 得到临时 Python 整数，继续使用原有交集和评分代码。

- 连续 ID：用 `id - first_id` 定位，直接支持起始 ID 不是 0 或 1。
- 稀疏 ID：保存排序的 `array('q')`，每行额外 8 字节，使用二分查找定位。支持负 ID 和超过 32 位的 ID。
- ID 采用整数语义，稀疏 ID 必须能放入有符号 64 位整数；这与当前数据集的数值主键一致。
- 真正的零掩码保留为零，不按缺失值处理。
- 缺失 ID 按原版 sidecar 缺行规则返回该 alias 的无谓词查询掩码。
- 同一真实表的多个 alias 分别构建，不混用不同 alias 的 QID 语义。

本版没有额外的全表 Python 整数缓存。一次批量查询返回的临时字典仍沿用原接口，之后由现有采样流程使用和释放。

## 生命周期与多个 worker

每个 worker 只拥有当前 template 的位图，所有分区与 Bitmap 复用。template 返回、异常退出时都在 finally 中释放；关闭 sampler 时再次幂等清理。

共享谓词文件在 template 之间保留，组装结束立即关闭 mmap；操作系统可以保留或回收共享文件缓存页。`predicate_cache_max_gib` 默认 96 GiB，限制本轮缓存的持久块文件、ID 文件及并发构建的这些文件，不是 RSS 上限，也不包含下面的 template 位图额度。没有自动 LRU 淘汰，超容量会报错。默认文件放在配置文件目录的 `.predicate_cache/` 下；旧运行文件不会自动删除，确认对应运行的所有 worker 结束后再清理旧目录。

多个 worker 构建前先申请当前 template 的**全部数组及临时 ID 检查标记的字节预算**，不足时等待；不会逐 alias 申请导致多个 worker 各持有部分内存而互相等待。超过整个预算的单个 template 会报错。进程退出后的过期额度会被后续申请回收，使用进程启动时间防止 PID 重用。

默认同一个配置文件的 worker 共享 `/tmp/join_sampling_v2_<配置路径哈希>.json` 及其 `.lock` 文件。该文件只记录额度，不存位图；不是 anno 数据表或位图文件。不同配置如需共用总预算，应显式设置相同的 `annotation_budget_path`，并使用相同预算值。

预算只约束**数组位图、稀疏 ID 数组和临时 ID 检查标记**，不约束整个进程 RSS、邻居对象、路径、root 分区或 PostgreSQL 内存。默认 96 GiB 根据当前约 248 GiB 内存机器设置，其他机器应调整。预算等待只改变执行时间，不改变采样策略或候选排序。

## 配置与运行

原有 JSON 配置结构可以继续使用。为保留原版采样结果，建议另存一份配置，并将 `sampling.output_path` 指向独立的 V2 结果目录。代码不会自动重写配置中的输出路径。

新增的可选 `sampling` 配置项：

```json
{
  "annotation_batch_size": 10000,
  "annotation_memory_budget_gib": 96,
  "annotation_budget_path": "/tmp/join_sampling_v2_shared_budget.json",
  "predicate_cache_enabled": true,
  "predicate_cache_block_size": 64,
  "predicate_cache_max_gib": 96,
  "annotation_compose_batch_size": 65536
}
```

除示例 `annotation_budget_path` 外，上述数值均为默认值。可选 `predicate_cache_dir` 控制缓存根目录；`predicate_cache_run_id` 可显式指定共同运行 ID，环境变量 `JOIN_SAMPLING_CACHE_RUN_ID` 优先。显式复用旧 ID 由调用方保证数据未变化，不能用 COUNT/MIN/MAX 判断数据版本。这些项不改变采样参数。

单个 worker：

```bash
cd /home/Sampler_CE/join_sampling/new_model_v2
python3 -u join_sampler_linear.py <config_path> 0 1
```

10 个 worker（配置文件需已准备）：

```bash
cd /home/Sampler_CE/join_sampling/new_model_v2
CONFIG=/path/to/tpch_v2_config.json
LOG_DIR=/path/to/v2_logs
mkdir -p "$LOG_DIR"
export JOIN_SAMPLING_CACHE_RUN_ID=$(cat /proc/sys/kernel/random/uuid)
for ((i=0; i<10; i++)); do
    nohup python3 -u join_sampler_linear.py "$CONFIG" "$i" 10 \
        > "$LOG_DIR/log_worker_$i.log" 2>&1 &
done
```

也可直接执行 `bash run_workers.sh <config_path> 10 <log_dir>`；脚本自动设置共同运行 ID，并通过绝对路径启动主程序。使用原项目能够运行 sampler 的 Python 环境，需要 NumPy、psycopg2、networkx、sqlglot 等现有依赖。

## 新计时项与比较方法

- `annotations.prepare`：完整位图准备，包含元数据查询、预算等待、分配与构建。
- `predicate_cache.build`：首次批量构建整张表的谓词缓存。
- `predicate_cache.table_lock_wait`：等待其他 worker 完成同表构建。
- `predicate_cache.stats.execute / fetchone`、`predicate_cache.scan.execute / fetchmany`：缓存构建中的数据库操作。
- `predicate_cache.pack_and_store_python`：批次打包、ID 验证和写入文件。
- `predicate_cache.open_table / open_mmap`：打开缓存描述和实际只读映射。
- `annotations.compose_qids_numpy`：当前 template 的分块 QID 组装。
- 缓存计数含 `table_hits`、`tables_built`、`scans`、`rows_scanned`、`predicates_built`、`bytes_built`、`blocks_mapped`、`mapped_bytes`；映射字节数是累计打开空间，不能作为物理内存峰值。
- `annotations.rows_composed / predicates_reused`：组装行数和使用的谓词组合数。
- `annotations.stats.execute / fetchone`：元数据聚合。
- `annotations.stats.cache_hits / cache_misses`：元数据缓存命中与首次查询次数（计数项）。
- `annotations.scan.execute / fetchmany`：命名游标建立和数据库批次读取；服务端谓词计算可能发生在 FETCH，因此不能只看 execute。
- `annotations.sql_build_python`：SQL 表达式生成。
- `annotations.decode_and_store_python`：批次解析和数组写入。
- `annotations.build_alias.<alias>`：各 alias 的构建总耗时。
- `annotations.allocate_python`：缓冲区分配。
- `annotations.budget_wait`：等待其他 worker 释放额度。
- `annotations.lookup.total / lookup_python`：采样期间的内存查询和整数解码。
- `annotations.release_python`：数组释放及预算归还。
- 计数中包含 `annotations.planned_bytes`、`annotations.buffer_bytes`、构建行数与查询 ID 数。
- `process.peak_rss_kib`：Linux `ru_maxrss`，单位 KiB，是 worker 自启动以来的 RSS 高水位，**不是当前模板专属峰值，也不是当前 RSS**。

每个 Bitmap 报告只包含采样阶段；最终 template 报告包含位图准备、采样和释放。DB 汇总包含 execute、fetchone、fetchmany、fetchall。比较性能以 template wall 为准，不能把预处理排除后宣称整体加速，也不能相加 inclusive 嵌套阶段。

统计日志：

```bash
python cal_time.py imdb/runfile_join_complexity
python cal_time.py tpch-skew/runfile --json /tmp/tpch_time.json --csv /tmp/tpch_templates.csv
```

不传路径默认统计 `imdb/runfile_join_complexity`。脚本只使用标准库，支持运行中日志，仅汇总完整 Template 报告；不累加 Bitmap 或重复的 inclusive/exclusive 数值。默认输出净累计时间（扣缓存锁等待、保留首次构建），并另外显示扣位图额度等待的累计时间和在线处理时间。缓存构建内的 capacity_lock_wait 会避免重复扣除；失败/空结果报告仍保留成本并显示状态。无统一开始/结束时间戳时不将最大 worker 时长宣称为精确的整轮实际时间，也不外推剩余时间。

## 共享缓存重构的验证

- `tests/` 中 16 项测试通过，覆盖直接 SQL 与缓存组装逐字节一致、跨 template/worker 复用、稀疏 ID、空表、NULL、字符串、QID 边界、容量限制、四进程竞争和强制终止构建者后的恢复。
- 缓存启用时，30 个随机种子下的三表采样结果、邻居 SQL 顺序及采样计数与 V1 一致。
- 真实 PostgreSQL 的事务临时表验证通过：连续/稀疏 ID、两个 alias、72 种谓词及 1/64/65/200/1042 个 QID，与直接 SQL 的最终缓冲区完全一致；结束后回滚，未修改数据集表。
- 未启动完整数据集采样；实际速度和并发内存需通过新日志测量。

```bash
python -m unittest discover -s tests -v
```

## 初版验证记录（历史）

- 所有四个 Python 文件通过语法检查。
- 11 项本地模拟测试通过：位序与填充、64/65/200/1042 位边界、连续/稀疏/负/大 ID、NULL 条件、零位图、重复谓词、多 alias、恒定位图、空表、异常释放、多进程额度及死进程额度回收。
- 30 个随机种子下，V1/V2 小型三表 workload 的最终样本、原表邻居 SQL 顺序，以及 root、扩展、随机游走、分区与样本数量统计一致。
- 本次没有执行真实 TPCH-skew 全量扫描。当前验证环境缺少原项目数据库依赖，SQL 用模拟数据库校验了生成结构和结果语义；实际 PostgreSQL 执行成本、真实内存峰值和性能提升应通过新日志确认。

本版保留原 SQL 解析和连接检查范围，不修复解析器、增加额外连接检查或改变采样搜索预算。前提是原版全局 anno 与当前 workload 一致；若旧 anno 过期，V2 重新计算得到的位图可能与过期位图不同。

## 后续修复：谓词字符串大小写

共用的 `mscn/query_representation/utils.py::extract_join_graph()` 原先对整条 SQL 执行 `.lower()`，导致 `'Brand#35'` 等字符串值被改变。现已改为解析原 SQL 后，仅将未加引号的 AST 标识符归一化为小写，保留字符串字面量及带引号标识符。

采样参数、连接计划和选择逻辑没有改变；V1、V2 通过共用解析器同时受益。V2 每个 template 都重新计算位图，直接使用修复后的条件。旧全局 PID anno 表不会自动更新：V1 若继续使用，需要按修复后的 workload/PID 映射显式重新构建；原建表代码会跳过已存在的表，单纯重新运行不能保证完成重建。本次没有删除或重建这些表。

回归测试位于 `mscn/query_representation/tests/test_identifier_case.py`；全量解析检查见 `tpch-skew/parser_audit/fixed/report.md`。
