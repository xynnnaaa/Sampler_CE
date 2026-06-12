import pickle
import glob
import os
import io
import time
import json
import psycopg2
from psycopg2.extras import execute_values
import sys
import re
import heapq
from collections import defaultdict
import hashlib

import sqlglot
from sqlglot import exp
import networkx as nx
from networkx.readwrite import json_graph

from wander_join import WanderJoinEngine

# --- 工具函数 ---
def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)
    query["subset_graph"] = nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    return query

def normalize_condition(cond_str):
    if not cond_str or cond_str == "None": return "None"
    parts = [p.strip() for p in cond_str.split('=')]
    parts.sort()
    return "=".join(parts)

TABLE_CARD = {
    "cast_info": 37610440,
    "movie_info": 13399115,
    "name": 5245495,
    "title": 4733511,
    "movie_keyword": 4523930,
    "char_name": 3140339,
    "person_info": 2963664,
    "movie_companies": 2609129,
    "movie_info_idx": 1380035,
    "aka_name": 901343,
    "aka_title": 361472,
    "company_name": 234997,
    "complete_cast": 135086,
    "keyword": 134170,
    "movie_link": 29997,
    "info_type": 113,
    "link_type": 18,
    "role_type": 12,
    "kind_type": 7,
    "comp_cast_type": 4,
    "company_type": 4,
}

# --- 数据结构 ---
class TrieNode:
    def __init__(self, token):
        # 当前节点的表统一用child标记

        self.token = token  # (parent_alias, child_alias, condition)
        self.children = {}  # Dict[StepToken, TrieNode]
        self.parent = None  

        # Template Info
        self.is_template_end = False
        self.end_template_keys = [] 

        # QID Info
        self.relevant_qids = set()   # 子树中所有查询的并集
        self.qid_map = {}            # {qid: {alias: pid}} (仅在该点结束的查询实例)
        self.subtree_total_queries = 0 

        # Graph Info，每一个节点都会在insert时设置
        self.child_alias = None
        self.real_name = None
        self.join_condition = None
        self.parent_alias = None
        self.sels = []

        # 采样状态 (Python 内存中的 T)
        self.beam = []  # List of {'data': dict, 'bmp': int, 'score': int}

class JoinPathTrie:
    def __init__(self):
        self.root = TrieNode("ROOT_SENTRY")

    def insert(self, template_key, path_signature, join_graph, instances):
        node = self.root
        for step_token in path_signature:
            if step_token not in node.children:
                new_node = TrieNode(step_token)

                # 填充 Graph Info 部分

                parent, child, cond = step_token
                new_node.parent = node
                new_node.parent_alias = parent
                new_node.child_alias = child
                new_node.join_condition = cond

                if child in join_graph.nodes:
                    new_node.real_name = join_graph.nodes[child]['real_name']
                    # new_node.sels = join_graph.nodes[child].get('sels', [])
                node.children[step_token] = new_node
            
            node = node.children[step_token]
            # 合并不同 template 对同一个节点的列需求
            if node.child_alias in join_graph.nodes:
                new_sels = join_graph.nodes[node.child_alias].get('sels', [])
                current_sels_set = set(node.sels)
                for s in new_sels:
                    if s not in current_sels_set:
                        node.sels.append(s)

        # 填充 Template Info 部分

        node.is_template_end = True
        node.pending_instances = instances # 暂存实例
        node.end_template_keys.append(template_key)

# --- 主类 ---
class JoinSampler:
    def __init__(self, config_path: str):
        print(f"Initializing JoinSampler with config: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        samp_conf = self.config.get("sampling", {})
        self.base_query_dir = samp_conf.get("base_query_dir", "./qrep")
        self.workload_name = samp_conf.get("workload_name", "")
        self.output_path = samp_conf.get("output_path", "./samples.json")
        self.skip_7a = samp_conf.get("skip_7a", True)

        self.m_partitions = samp_conf.get("m_partitions", 10)
        self.k_bitmaps = samp_conf.get("k_bitmaps", 5)
        self.limit_x = samp_conf.get("limit_x", 50)
        self.batch_size = samp_conf.get("batch_size", 1000)
        
        # [Added] Lookahead configuration
        self.lookahead_k = samp_conf.get("lookahead_k", 5) # 向后看几步
        self.lookahead_samples = samp_conf.get("lookahead_samples", 500) # 采样次数

        print("Configuration Loaded:")
        print(f"  Base Query Directory: {self.base_query_dir}")
        print(f"  Workload Name: {self.workload_name}")
        print(f"  Output Path: {self.output_path}")
        print(f"  Skip Template '7a': {self.skip_7a}")
        print(f"  M Partitions: {self.m_partitions}")
        print(f"  K Bitmaps: {self.k_bitmaps}")
        print(f"  Limit X: {self.limit_x}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Lookahead K: {self.lookahead_k}")

        db_conf = self.config.get("database")
        self.db_config = {
            "host": db_conf.get("host", "localhost"),
            "port": db_conf.get("port", 5432),
            "dbname": db_conf.get("dbname", "imdb"),
            "user": db_conf.get("user", "your_username")
        }
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("Database connection established.")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise

        self.engine = WanderJoinEngine(self.db_config)

        import itertools
        self.tie_breaker = itertools.count()
        
        self.trie = JoinPathTrie()
        self.global_predicate_map = defaultdict(dict) # {real_table: {pred_sql: pid}}
        self.global_pid_counters = defaultdict(int)   # {real_table: next_pid}
        self.all_involved_tables = set()

    def close(self):
        print("Closing database connection...")
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        # [Added] Close Engine
        if self.engine:
            self.engine.close()
        print("Resources released.")

    def load_and_parse_workload(self):
        # 和线性逻辑完全一致，信息存储到 self.temp_template_data

        self.all_involved_tables = set()

        self.temp_template_data = defaultdict(lambda: {'instances': [], 'graph': None})

        try:
            template_names = [d for d in os.listdir(self.base_query_dir) if os.path.isdir(os.path.join(self.base_query_dir, d))]
        except FileNotFoundError:
            print(f"ERROR: Base query directory not found: {self.base_query_dir}")
            return
        
        if not template_names:
            print(f"No template subdirectories found in '{self.base_query_dir}'.")
            return
        
        for template_name in sorted(template_names):
            if template_name == "7a" and self.skip_7a:
                print("Skipping template '7a' as per configuration.")
                continue  # Skip known problematic template
            input_template_dir = os.path.join(self.base_query_dir, template_name)
            pkl_files = sorted(glob.glob(os.path.join(input_template_dir, "*.pkl")))
            if not pkl_files:
                print(f"No .pkl files found in '{input_template_dir}'. Skipping this template.")
                continue
            for pkl_file in pkl_files:
                try:
                    qrep = load_qrep(pkl_file)
                except Exception as e:
                    print(f"Error loading {pkl_file}: {e}")
                    continue

                join_graph = qrep["join_graph"]
                subset_graph = qrep["subset_graph"]

                for subplan_tuple in sorted(subset_graph.nodes()):
                    # check 这里跳过了单表template
                    if len(subplan_tuple) < 2:
                        continue

                    sorted_aliases = sorted(list(subplan_tuple))
                    sub_graph = join_graph.subgraph(subplan_tuple)
                    edges_info = []

                    for u, v, data in sub_graph.edges(data=True):
                        if u > v:
                            u, v = v, u

                        cond = data.get("join_condition", "")
                        if not cond:
                            print(f"Warning: No join condition found between {u} and {v} in {pkl_file}")
                            continue

                        cond_clean = normalize_condition(cond)
                        edges_info.append(f"{u}|{v}|{cond_clean}")

                    edges_info.sort()
                    join_sig_str = "||".join(edges_info)

                    template_key = (tuple(sorted_aliases), join_sig_str)

                    # 存储当前查询的谓词-PID 映射
                    current_query_pids = {} # {alias: pid}

                    for alias in sorted_aliases:
                        node_data = join_graph.nodes[alias]
                        real_name = node_data["real_name"]

                        self.all_involved_tables.add(real_name)

                        preds_list = node_data.get("predicates", [])

                        clean_pred_list = [] # 没有别名的谓词
                        for pred in preds_list:
                            pattern = fr"\b{alias}\."
                            pred_clean = re.sub(pattern, "", pred).strip()
                            clean_pred_list.append(pred_clean)

                        if clean_pred_list:
                            combined_pred = " AND ".join(clean_pred_list)
                            if combined_pred not in self.global_predicate_map[real_name]:
                                pid = self.global_pid_counters[real_name]
                                # 从0开始编号
                                self.global_predicate_map[real_name][combined_pred] = pid
                                self.global_pid_counters[real_name] += 1

                            # 这里存 pid，不用存字符串了
                            current_pid = self.global_predicate_map[real_name][combined_pred]
                            current_query_pids[alias] = current_pid
                        else:
                            current_query_pids[alias] = -1  # 表示没有谓词


                    if template_key not in self.temp_template_data:
                        self.temp_template_data[template_key]['graph'] = sub_graph.copy()
                    self.temp_template_data[template_key]['instances'].append(current_query_pids)

        print(f"Parsing complete. Found {len(self.temp_template_data)} distinct join templates.")

        print("Global Predicate Map Summary:")
        for real_table, pred_map in self.global_predicate_map.items():
            print(f"  Table '{real_table}': {len(pred_map)} unique predicates.")


    def build_global_trie(self):
        print("\nBuilding Trie from Join Templates...")
        self.trie = JoinPathTrie()

        for template_key, data in self.temp_template_data.items():
            aliases_tuple = template_key[0]
            join_graph = data['graph']
            instances = data['instances'] # List[Dict{alias: pid}]

            for node, info in join_graph.nodes(data=True):
                sels = []
                edges = join_graph.edges(node)
                for edge in edges:
                    # edge_data = join_graph.get_edge_data(edge[0], edge[1])
                    edge_data = join_graph[edge[0]][edge[1]]
                    if "!" in edge_data["join_condition"]:
                        jconds = edge_data["join_condition"].split("!=")
                    else:
                        jconds = edge_data["join_condition"].split("=")
                    for jc in jconds:
                        jc = jc.strip()
                        if node == jc[0:len(node)]:
                            if jc not in sels:
                                sels.append(jc)
                        jc_node = jc.split(".")[0]
                        join_graph[edge[0]][edge[1]][jc_node] = jc
                
                # 如果没有主键就加上主键
                if f"{node}.id" not in sels and f"{node}.Id" not in sels:
                    sels.append(f"{node}.id")
                
                join_graph.nodes()[node]["sels"] = sels

            try:
                path_signature = self.get_deterministic_execution_plan(join_graph, list(aliases_tuple))
                self.trie.insert(template_key, path_signature, join_graph, instances)

            except Exception as e:
                print(f"Error building trie for {aliases_tuple}: {e}")
                continue

        print("Assigning QIDs per Root Subtree...", flush=True)

        for root_token, root_node in self.trie.root.children.items():
            current_qid_counter = 0

            def assign_and_aggregate(node):
                nonlocal current_qid_counter

                my_relevant = set()

                my_query = set()

                if node.is_template_end:
                    for inst in node.pending_instances:
                        qid = current_qid_counter
                        current_qid_counter += 1

                        node.qid_map[qid] = inst
                        my_relevant.add(qid)
                        my_query.add(qid)
                    node.pending_instances = [] 

                for child in node.children.values():
                    child_relevant = assign_and_aggregate(child)
                    my_relevant.update(child_relevant)

                node.relevant_qids = my_relevant
                node.current_qids = my_query
                return my_relevant
            
            assign_and_aggregate(root_node)

            root_node.subtree_total_queries = current_qid_counter

            print(f"  Root '{root_node.child_alias}' ({root_node.real_name}): {current_qid_counter} queries.", flush=True)

    def partition_root_table(self, real_name, m_partitions):
        """
        将root划分为 m 个分区，返回每个分区的 id 列表。
        """
        try:
            self.cursor.execute(f"SELECT id FROM {real_name} ORDER BY RANDOM();")
            all_ids = [row[0] for row in self.cursor.fetchall()]

            total_rows = len(all_ids)

            # 行数少于分区数时，允许重复选id
            if total_rows < m_partitions:
                partitions = []
                for i in range(m_partitions):
                    pid = all_ids[i % total_rows]
                    partitions.append([pid])

            if total_rows > m_partitions:
                base_size = total_rows // m_partitions
                remainder = total_rows % m_partitions

                partitions = []
                start_idx = 0
                for i in range(m_partitions):
                    part_size = base_size + (1 if i < remainder else 0)
                    end_idx = start_idx + part_size
                    partition_ids = all_ids[start_idx:end_idx]
                    partitions.append(partition_ids)
                    start_idx = end_idx

            return partitions

        except Exception as e:
            print(f"Error partitioning root table '{real_name}': {e}")
            raise e
        
    def prepare_subtree_pid_map(self, root_node):
        """
        为整个 Root 子树构建统一的 PID -> QID 映射。
        """
        # 收集所有 QID -> Instance
        subtree_instances = {} # { qid: {alias: pid} }
        
        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.is_template_end:
                subtree_instances.update(node.qid_map)
            for child in node.children.values():
                stack.append(child)
        
        # 构建 Map
        pid_map = defaultdict(dict)
        global_map = defaultdict(int)
        
        for qid, inst in subtree_instances.items():
            for alias, pid in inst.items():
                if pid == -1:
                    global_map[alias] |= (1 << qid)
                else:
                    if pid not in pid_map[alias]:
                        pid_map[alias][pid] = 0
                    pid_map[alias][pid] |= (1 << qid)
                    
        return pid_map, global_map

    def sample_trie_root(self, root_node):
        # 对root进行分区
        real_name = root_node.real_name
        t1 = time.time()
        partitions = self.partition_root_table(real_name, self.m_partitions)
        print(f"    Root partitioned into {len(partitions)} segments in {time.time() - t1:.2f}s")

        t2 = time.time()
        # 构建子树级 pid--qid 映射
        pid_map, global_map = self.prepare_subtree_pid_map(root_node)
        print(f"    build pid-qid map in {time.time() - t2:.2f}s")

        subtree_covered = set()
        total_qids = root_node.subtree_total_queries
        print(f"    total queries: {total_qids}")

        for k_idx in range(self.k_bitmaps):
            print(f"    --> Bitmap {k_idx+1}/{self.k_bitmaps}...", flush=True)

            if len(subtree_covered) == total_qids:
                print("        All queries covered. Stop sampling.")
                break

            if len(subtree_covered)/total_qids >= 0.95:
                print("        Coverage >= 95%. Stop sampling.")
                break

            bitmap_start_time = time.time()

            current_bitmap_samples = defaultdict(list)
            
            for p_idx, partition_ids in enumerate(partitions):
                print(f"            Sampling for partition {p_idx+1}...")

                # 收集当前分区的样本
                # 结构: { template_key: sample_dict }
                partition_results = {}
                partition_newly_covered = set()

                t_partition_start = time.time()

                # A. Root 处理
                t_root_start = time.time()
                success = self.execute_root_step(root_node, partition_ids, pid_map, global_map, subtree_covered, total_qids)

                if success:
                    if root_node.is_template_end:
                        sample, covered = self.collect_samples_for_node(root_node, subtree_covered)
                        if sample:
                            partition_newly_covered.update(covered)
                            for key in root_node.end_template_keys:
                                partition_results[key] = sample
                    
                    print(f"                Root sampled in {time.time() - t_root_start:.2f}s")

                    # B. DFS 遍历子树
                    child_nodes = list(root_node.children.values())
                    for i, child in enumerate(child_nodes):
                        # i > 0 表示进入了兄弟分支，触发回溯
                        child_results, child_covered = self.dfs_join_recursive(
                            child, 
                            is_branch_switch=(i > 0),
                            global_covered=subtree_covered,
                            pid_map=pid_map,
                            global_map=global_map,
                            total_queries=total_qids
                        )
                        partition_results.update(child_results)
                        partition_newly_covered.update(child_covered)
                    
                    subtree_covered.update(partition_newly_covered)
                    for key, sample in partition_results.items():
                        current_bitmap_samples[key].append(sample)

                self.conn.commit()

                coverage_pct = (len(subtree_covered) / total_qids * 100) if total_qids > 0 else 0
                print(f"            Partition {p_idx+1} finished, total time:{time.time() - t_partition_start:.2f}s. Covers {len(partition_newly_covered)} new queries. Current Global Coverage: {coverage_pct:.2f}%.")

            print(f"        Bitmap {k_idx + 1} finished, total time:{time.time() - bitmap_start_time:.2f}s.")

            if current_bitmap_samples:
                self.save_single_bitmap(k_idx + 1, root_node.child_alias, current_bitmap_samples)


    def dfs_join_recursive(self, current_node, is_branch_switch, global_covered, pid_map, global_map, total_queries):
        """
        DFS 遍历核心逻辑
        """
        collected_results = defaultdict(list)
        all_newly_covered = set()

        if is_branch_switch:
            # 切换分支时，回溯 k+1 层重算
            success = self.backtrack_and_recalculate(current_node, global_covered, pid_map, global_map, total_queries)
        else:
            # 正常线性向下 Join
            success = self.execute_join_step(current_node, global_covered, pid_map, global_map, total_queries)

        if not success: 
            return collected_results, all_newly_covered

        # 收集样本
        if current_node.is_template_end:
            sample_dict, newly_covered = self.collect_samples_for_node(current_node, global_covered)
            if sample_dict:
                all_newly_covered.update(newly_covered)
                for tmpl_key in current_node.end_template_keys:
                    collected_results[tmpl_key] = sample_dict

        # 递归处理子节点
        children = list(current_node.children.values())
        for i, child in enumerate(children):
            child_results, child_covered = self.dfs_join_recursive(
                child, 
                is_branch_switch=(i > 0), # 从第二个孩子开始触发回溯
                global_covered=global_covered,
                pid_map=pid_map,
                global_map=global_map,
                total_queries=total_queries
            )
            all_newly_covered.update(child_covered)
            collected_results.update(child_results)

        return collected_results, all_newly_covered

    def backtrack_and_recalculate(self, branch_entry_node, global_covered, pid_map, global_map, total_queries):
        """
        从当前分叉节点向上回溯最多 lookahead_k + 1 步，重新打分。
        """
        success = True

        path = []
        curr = branch_entry_node
        # 向上找受影响的节点链
        for _ in range(self.lookahead_k + 1):
            if curr.parent and curr.parent.token != "ROOT_SENTRY":
                path.append(curr)
                curr = curr.parent
            else:
                path.append(curr)
                break
        
        # 逆序（从祖先向下）重新执行 execute_join_step
        # 关键：execute_join_step 内部会根据当前的 branch_entry_node 提取新的前瞻路径
        for node in reversed(path):
            success = self.execute_join_step(node, global_covered, pid_map, global_map, total_queries, target_branch=branch_entry_node)
            if not success: break

        return success

    def execute_join_step(self, node, global_covered, pid_map, global_map, total_queries, target_branch=None):
        """
        执行单步采样：Parent Beam -> Wander Join -> Pruning -> Node Beam
        """
        parent_node = node.parent
        if not parent_node.beam: return False

        # 1. 提取前瞻计划
        # 如果提供了 target_branch，说明我们在回溯中，lookahead 必须指向新分支方向
        # 否则默认沿着子树第一个分支看
        lookahead_plan = self._extract_lookahead_plan(node, target_branch)

        # 2. 计算当前目标掩码 (Relevant & Uncovered)
        target_mask = self._get_uncovered_mask(node.relevant_qids, global_covered)

        # 3. 调用引擎扩展 Beam
        candidates, _ = self.engine.sample_beam_extensions(
            parent_node.beam,
            lookahead_plan,
            pid_map,
            global_map,
            k_samples=self.lookahead_samples,
            workload_name=self.workload_name,
            uncovered_mask_int=target_mask
        )

        if not candidates:
            node.beam = []
            return False

        # 4. 剪枝保留 Top-B
        # candidates 是 [{t_idx, rj_data, rj_bmp, score}]
        node.beam = self._prune_to_beam(parent_node.beam, candidates, self.limit_x)
        return True

    def _extract_lookahead_plan(self, node, target_branch=None):
        """
        从 node 开始向下提取 k 步作为 Wander Join 的探测路径。
        如果指定了 target_branch，确保路径经过它。
        """
        plan = []
        curr = node
        
        for _ in range(self.lookahead_k + 1):
            plan.append({
                'alias': curr.child_alias,
                'real_name': curr.real_name,
                'parent': curr.parent_alias,
                'join_condition': curr.join_condition,
                'sels': curr.sels
            })
            
            if not curr.children: break
            
            # 路径决策
            if target_branch and self._is_ancestor_of(curr, target_branch):
                # 寻找通往目标分支的孩子
                found = False
                for child in curr.children.values():
                    if child == target_branch or self._is_ancestor_of(child, target_branch):
                        curr = child
                        found = True
                        break
                if not found: curr = list(curr.children.values())[0]
            else:
                # 默认逻辑：取第一个孩子（通常是最深或者最重要的）
                curr = list(curr.children.values())[0]
        return plan

    def collect_samples_for_node(self, node, covered_mask):
        """从当前 node.beam 的 Top-1 提取结果并保存"""
        if not node.beam: return None, set()

        try:
            sorted_aliases = node.end_template_keys[0][0]
            
            best = node.beam[0]
            sample_dict = {}
            for k, v in best['data'].items():
                if k.endswith(".id") or k.endswith(".Id"):
                    alias = k.split('.')[0]
                    sample_dict[alias] = v
            
            for alias in sorted_aliases:
                if alias not in sample_dict:
                    print(f"            Warning: ID for alias '{alias}' not found in selected row.")

            # 3. 更新覆盖率，只更新当前node对应template对应的所有查询
            # 解析 anno_bits (QID)
            newly_covered = set()
            anno_bits = best['bmp']
            if anno_bits:
                total_len = len(anno_bits)
                for idx, bit in enumerate(anno_bits):
                    if bit == '1':
                        qid = total_len - 1 - idx
                        if qid in node.current_qids and qid not in covered_mask:
                            newly_covered.add(qid)
            
            return sample_dict, newly_covered
        
        except Exception as e:
            print(f"Error collecting samples: {e}")
            return None, set()
        
    def save_single_bitmap(self, k, root_alias, current_bitmap_samples):
        """
        保存当前 Root 子树在第 k 轮生成的样本。
        文件名: {output_path}/{root_alias}_bitmap_{k}.json
        内容结构: { template_id: [ [sample_dict...] ] } (保持结构一致性，虽然只有一层)
        或者简化为: { template_id: [sample_dict...] } (更直观)
        """
        # 确保目录存在
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        filename = f"{root_alias}_bitmap_{k}.json"
        full_path = os.path.join(self.output_path, filename)
        
        formatted_output = {}
        
        for tmpl_key, samples_list in current_bitmap_samples.items():
            # 生成 Template ID
            aliases_tuple = tmpl_key[0]
            sig = tmpl_key[1]
            import hashlib
            sig_hash = hashlib.md5(sig.encode('utf-8')).hexdigest()[:6]
            template_id = f"{'_'.join(aliases_tuple)}_{sig_hash}"
            
            # 格式化: 转换为 [Header, Rows...]
            if not samples_list:
                formatted_output[template_id] = []
                continue
                
            aliases = sorted(samples_list[0].keys())
            header = [f"{alias}.id" for alias in aliases]
            formatted_rows = [header]
            
            for row_dict in samples_list:
                row = [row_dict[alias] for alias in aliases]
                formatted_rows.append(row)
            
            formatted_output[template_id] = formatted_rows

        # 序列化
        def default_serializer(obj):
            import decimal
            import datetime
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            return str(obj)
        
        try:
            with open(full_path, 'w') as f:
                json.dump(formatted_output, f, default=default_serializer, indent=2)
            print(f"    Saved bitmap {k} to {filename}", flush=True)
        except Exception as e:
            print(f"Error saving bitmap: {e}", flush=True)

    # --- 辅助方法 ---
    def _is_ancestor_of(self, potential_ancestor, node):
        curr = node
        while curr:
            if curr == potential_ancestor: return True
            curr = curr.parent
        return False

    def _get_uncovered_mask(self, relevant_qids, covered_mask_int):
        """将 Set 形式的 relevant_qids 过滤掉已覆盖的，转为 Int Mask"""
        mask = 0
        for qid in relevant_qids:
            # 检查 covered_mask_int 的第 qid 位是否为 0
            if not (covered_mask_int & (1 << qid)):
                mask |= (1 << qid)
        return mask

    def _prune_to_beam(self, parent_beam, candidates, limit):
        """将 WanderJoinEngine 返回的扩展候选转化为标准的 beam 结构"""
        next_candidates_heap = []
        for cand in candidates:
            t_idx = cand['t_idx']
            rj_data = cand['rj_data'] 
            rj_bmp = cand['rj_bmp']   
            best_score = cand['score'] # 带未来潜力的打分

            t_tuple = parent_beam[t_idx]
            current_base_bmp = t_tuple['bmp']
            new_real_bmp = current_base_bmp & rj_bmp

            new_data = t_tuple['data'].copy()
            new_data.update(rj_data)

            item = (best_score, next(self.tie_breaker), new_data, new_real_bmp)
            if len(next_candidates_heap) < self.limit_x:
                heapq.heappush(next_candidates_heap, item)
            else:
                if best_score > next_candidates_heap[0][0]:
                    heapq.heappushpop(next_candidates_heap, item)

        sorted_cands = sorted(next_candidates_heap, key=lambda x: x[0], reverse=True)
        new_beam = []
        for score, _, data, bmp in sorted_cands:
            new_beam.append({
                'data': data,
                'bmp': bmp,
                'score': score
            })
        return new_beam
    
    def sample(self):
        """
        [主入口]
        执行完整的 Join 采样流程，并保存结果。
        """
        print("Starting Join Sampling Process...")
        start_time = time.time()

        self.load_and_parse_workload()
        # self.create_annotation_tables()
        self.build_global_trie()

        print(f"Total Workload Analysis Time: {time.time() - start_time:.2f}s", flush=True)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")

        for root_token, root_node in self.trie.root.children.items():
            print(f"\nProcessing Root: {root_node.child_alias}")
            self.sample_trie_root(root_node)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python join_sampler.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]

    sampler = JoinSampler(config_path)
    try:
        sampler.sample()
    finally:
        sampler.close()
