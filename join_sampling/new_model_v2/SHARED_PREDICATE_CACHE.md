# 跨 template、跨 worker 的谓词命中缓存

这次重构只改变位图准备和读取来源，保持 workload 解析、QID 编号、root 随机分区、采样参数、候选排序、random.choice 调用、路径扩展、覆盖判断和输出格式。

## 数据流

1. 每个 worker 继续解析完整工作负载，得到原有的真实表 → PID → 完整谓词 SQL。
2. 缓存按精确 SQL 去重并排序，PID 编号不参与文件寻址；字符串常量及谓词表达式保持不变。
3. 第一次需要一张表时，在其文件锁内查询 COUNT/MIN/MAX，并按最多 64 个谓词为一块扫描整张表。块内第 i 个谓词为真时贡献 `1 << i`，NULL 走 ELSE。
4. 命名游标流式返回 ID 和固定 64 位字符串；NumPy 批量打包为小端 uint64。连续 ID 使用 ID 偏移定位并检测重复；非连续 ID 保留排序，第一块保存公共 int64 ID 数组，后续块验证相同的 ID 顺序。
5. 文件都完成后写 manifest，再原子发布整个目录。其他进程只使用完整目录，打开只读 mmap。
6. 当前 template 把需要的命中位映射到自己的 QID 集合。按默认 65,536 行分块初始化无谓词查询的 global_mask，再用 NumPy 条件 OR 写入原有 QID 缓冲区。
7. 组装结束关闭 mmap。当前 template 的 QID 缓冲区继续供原 WanderJoinEngine 使用，结束后照常释放；谓词文件保留供其他 template/worker 复用。

缓存无需为每个 QID 保存重复的相同谓词。组装阶段只处理当前 alias 需要的组合，不逐行扫描全部全局 PID。

## 文件布局与容量

默认根目录为配置文件所在目录的 `.predicate_cache/`。其下使用 query_file 文件名去掉扩展名的子目录，不再使用运行 ID 哈希。例如 `join_complexity_train.sql` 对应 `.predicate_cache/join_complexity_train/`。数据库配置（不含密码）和格式版本写入 namespace.json，身份不符时拒绝读取。表级子目录仍按真实表、排序后的谓词清单和块大小哈希划分：

```text
<query_file 文件名去掉扩展名>/
  namespace.lock / namespace.json
  capacity.lock / capacity.json
  <表哈希>.lock
  <表哈希>/
    manifest.json
    block_0.bin
    block_1.bin
    ...
    ids.bin        # 仅稀疏 ID 表
```

每个块是 N 个小端 uint64，最后一个块也保留 64 位宽度。若配置每块 B 个谓词，B 必须为 1～64，缓存数据大小为：

```text
8 × N × ceil(不同完整谓词数 / B)
另加稀疏 ID 数组 8 × N
```

例如 IMDB cast_info 的 1,745 种谓词，默认分为 28 个块，约 7.56 GiB；此前逐谓词紧密打包估计的 7.36 GiB 不包含最后块的补齐。本轮所有表的完整缓存会多一些补齐空间。TPCH-skew 按已审计的谓词数及近似行数估计约 80.3 GiB，仅供容量规划，不是实际分配结果。

`predicate_cache_max_gib` 默认 96，限制当前 workload 文件名目录的缓存块和 ID 文件的总字节数，包括并发构建中预留的完整文件；不会把缺页加载后的共享物理页按 worker 数量重复计算，也不是 RAM/RSS 上限。构建时还有批次临时数组和连续 ID 的每行 1 bit 检查标记，这些不在文件额度内。

原 `annotation_memory_budget_gib` 独立限制各 worker 私有的 template 位图数组。缓存页、临时数组、路径对象和 PostgreSQL 内存必须另外考虑。旧运行缓存占用真实磁盘空间；容量限制只针对当前 workload 文件名目录。默认不自动淘汰或删除旧运行文件，应在确认该运行的全部 worker 已结束后清理。

## 并发与中断

- 同表只允许一个构建者，不同表可以并行构建。
- 所有块先写 `.building.*` 临时目录；其他 worker 不读取临时文件。
- 构建者正常出错时清理临时文件和文件容量预留。
- 构建者被强制终止时 flock 自动释放。下一个同表构建者清理其临时目录；容量记录按 PID 加进程启动时间回收死进程预留，已发布目录的额度仍保留。
- 缓存文件尺寸、清单和格式不符时明确报错，不静默接受半成品。
- 缓存构建在申请 template 数组额度前执行，避免持有数组额度等待共享构建造成依赖死锁。

## 自动复用与数据正确性

基础表必须在整轮运行内保持稳定，谓词也必须是稳定的纯过滤表达式。缓存不是数据库快照；行数与 ID 检查不能检测列值更新。同一轮内各连接继续使用原隔离级别。

同一缓存根目录、query_file 文件名、数据库身份、谓词清单及块大小下，重复启动会自动复用已发布缓存。单 worker 和多 worker 都不再需要运行 ID；原 predicate_cache_run_id 和 JOIN_SAMPLING_CACHE_RUN_ID 不再使用。表级谓词清单变化会使用新的表级目录，不混用旧谓词。数据库列值变化却未修改谓词时无法自动发现；所有相关 worker 结束后应先删除对应 workload 目录再构建，仅 COUNT/MIN/MAX 相同不能证明数据没变。旧哈希目录不会自动迁移或删除，首次使用文件名目录需要重新构建。

启动示例：

```bash
cd /home/Sampler_CE/join_sampling/new_model_v2
bash run_workers.sh ./imdb/config.json 10 ./imdb/runfile_cache
```

此命令沿用 config 中的样本输出路径。需要区分结果时应另外设置 `sampling.output_path`。代码改动不会自动重写输出路径，也不会自动重启现有 worker。

可选配置：

```json
{
  "predicate_cache_enabled": true,
  "predicate_cache_block_size": 64,
  "predicate_cache_max_gib": 96,
  "annotation_compose_batch_size": 65536
}
```

关闭 `predicate_cache_enabled` 可回到取消 dense 排序、worker 内缓存 COUNT/MIN/MAX 的直接 SQL 构建方式。默认路径无需增加配置；`predicate_cache_dir` 可改用空间足够的本地磁盘。需要 NumPy 支持 `packbits(..., bitorder='little')`；本次验证环境为 NumPy 2.0.2。

## 计时及验证

首次构建、文件锁等待、SQL execute/FETCH、批量打包、mmap 打开和 QID 组装都纳入 template 总时间。分别查看 `predicate_cache.*` 和 `annotations.compose_qids_numpy`；inclusive 阶段嵌套，不能直接相加。`mapped_bytes` 是各次映射长度的累计值，不是当前物理内存。

16 项目录内回归测试通过，包括逐字节对照、30 个随机种子的真实采样方法对照、四进程同时申请时仅构建一次，以及强制终止后的重建。真实 PostgreSQL 会话中使用事务临时表另行验证了 72 种不同谓词、连续/稀疏 ID、NULL、大小写字符串、多个 alias、QID 1/64/65/200/1042；结果与原 SQL 完全一致，测试结束回滚。

尚未运行完整 IMDB/TPCH-skew 新采样，也未承诺特定加速倍数。首次构建的成本必须计入与旧日志的比较。
