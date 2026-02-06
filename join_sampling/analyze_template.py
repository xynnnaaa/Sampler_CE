import os
import glob
import pickle
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
import time

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



def load_qrep(fn):
    assert ".pkl" in fn
    with open(fn, "rb") as f:
        query = pickle.load(f)

    query["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])

    return query

def normalize_condition(cond_str):
    """
    标准化连接条件字符串，确保 'a=b' 和 'b=a' 被视为相同。
    去掉空格并按字典序排列等号两边。
    """
    if not cond_str:
        return "None"
    parts = [p.strip() for p in cond_str.split('=')]
    parts.sort()
    return "=".join(parts)


def get_deterministic_execution_plan(join_graph, aliases):
    """
    为给定的 Join Graph 生成一个绝对确定（Deterministic）的遍历路径。
    返回一个由 (parent, child, condition) 组成的签名列表。
    """

    # def root_score(alias):
    #     real_name = join_graph.nodes[alias]['real_name']
    #     return (-TABLE_CARD.get(real_name, float("inf")), alias)

    # root_table = max(aliases, key=root_score)

    aliases = sorted(aliases)

    scored = []
    for a in aliases:
        real_name = join_graph.nodes[a]['real_name']
        card = TABLE_CARD.get(real_name, float("inf"))
        scored.append((card, a))
    scored.sort()

    root_table = None

    for card, alias in scored:
        if card > 10000:
            root_table = alias
            break

    if root_table is None or root_table == 'ci' or root_table == 'mi1' or root_table == 'mi2':
        root_table = scored[0][1]

    visited = {root_table}

    path_signature = [("ROOT", root_table, "None")]

    while len(visited) < len(aliases):
        candidates = []
        for u in sorted(visited):
            for v in sorted(join_graph.neighbors(u)):
                if v not in visited:
                    real_name = join_graph.nodes[v]['real_name']
                    card = TABLE_CARD.get(real_name, float("inf"))
                    candidates.append((card, u, v))

        if not candidates:
            raise RuntimeError("Join graph is not connected")
        
        _, parent, child = min(candidates)
        visited.add(child)
        edge_data = join_graph.get_edge_data(parent, child)
        raw_condition = edge_data.get("join_condition")
        norm_cond = normalize_condition(raw_condition)

        path_signature.append((parent, child, norm_cond))
    
    return path_signature


class TrieNode:
    def __init__(self, token):
        self.token = token
        self.children = {} # Dict[StepToken, TrieNode]
        self.is_template_end = False # 标记是否有 Template 在此结束
        self.template_ids = []

class JoinPathTrie:
    def __init__(self):
        self.root = TrieNode("ROOT_SENTRY")
        self.template_freq = {}  # template_id -> query_count
    
    def insert(self, template_id, path_signature):
        node = self.root
        for step_token in path_signature:
            if step_token not in node.children:
                node.children[step_token] = TrieNode(step_token)
            node = node.children[step_token]
        
        node.is_template_end = True
        node.template_ids.append(template_id)

    def count_nodes(self):
        """
        计算叶子节点数量。
        叶子节点意味着一条游走路径的终点，对应最小链覆盖数。
        """
        leaves_count = 0
        total_count = 0
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            total_count += 1
            
            if not node.children:
                leaves_count += 1
            else:
                for child in node.children.values():
                    stack.append(child)

        return leaves_count, total_count
    
    def _dfs_subtree_stats(self, node):
        """
        DFS 统计以 node 为根的子树：
        - 节点数
        - template 数
        - 叶子节点数
        """
        node_count = 1
        template_count = len(node.template_ids)
        query_count = sum(
            self.template_freq.get(tid, 0)
            for tid in node.template_ids
        )
        leaf_count = 1 if not node.children else 0

        for child in node.children.values():
            c_node, c_tpl, c_leaf, c_query = self._dfs_subtree_stats(child)
            node_count += c_node
            template_count += c_tpl
            leaf_count += c_leaf
            query_count += c_query

        return node_count, template_count, leaf_count, query_count

    def count_root_subtrees(self):
        """
        对 root 的每一个子节点，分别统计其子树信息
        返回 dict:
        {
            step_token: {
                'nodes': int,
                'templates': int,
                'leaves': int,
                'queries': int
            }
        }
        """
        results = {}

        for step_token, child in self.root.children.items():
            node_cnt, tpl_cnt, leaf_cnt, query_cnt = self._dfs_subtree_stats(child)

            results[step_token] = {
                "nodes": node_cnt,
                "templates": tpl_cnt,
                "leaves": leaf_cnt,
                "queries": query_cnt,
            }

        return results

    def analyze_wanderjoin_efficiency(self):
        k = 1
        efficient_count = 0
        total_count = 0

        stack = [self.root]  # 从根节点开始遍历所有节点
        
        while stack:
            node = stack.pop()
            total_count += 1
            
            # 检查从当前节点开始的连续k层是否都是单分支
            current = node
            is_efficient = True
            
            for i in range(k):
                if not current.children or len(current.children) != 1:
                    is_efficient = False
                    break
                current = next(iter(current.children.values()))  # 获取唯一的子节点
            
            if is_efficient:
                efficient_count += 1
            
            # 继续遍历所有子节点
            for child in node.children.values():
                stack.append(child)

        print(f"Strategy Efficiency Analysis:")
        print(f"Total nodes: {total_count}")
        print(f"Nodes with single-child paths of length {k}: {efficient_count}")
        print(f"Efficiency Ratio: {100 * efficient_count / total_count:.2f}%")

class WorkloadAnalyzer:
    def __init__(self, base_query_dir, skip_7a=False):
        self.base_query_dir = base_query_dir
        self.skip_7a = skip_7a
        self.unique_templates = {} # {signature_string: join_graph}
        self.trie = JoinPathTrie()
        self.template_query_count = defaultdict(int)

    def load_and_group_workload_ceb(self):
        """
        遍历文件，提取唯一的 Join Template 结构。
        这里去掉了谓词处理，只保留图结构的分组。
        """
        print(f"Loading workload from: {self.base_query_dir}")
        temp_groups = {} # Key: (aliases_tuple, join_sig_str), Value: subgraph

        try:
            template_dirs = sorted([d for d in os.listdir(self.base_query_dir) 
                                    if os.path.isdir(os.path.join(self.base_query_dir, d))])
        except FileNotFoundError:
            print("Base dir not found.")
            return

        total_files = 0

        total_subquery_counts = 0
        
        for template_name in template_dirs:
            if template_name == "7a" and self.skip_7a:
                print("Skipping template '7a' as per configuration.")
                continue

            input_template_dir = os.path.join(self.base_query_dir, template_name)
            pkl_files = sorted(glob.glob(os.path.join(input_template_dir, "*.pkl")))
            
            for pkl_file in pkl_files:
                try:
                    qrep = load_qrep(pkl_file)
                    total_files += 1
                except Exception as e:
                    print(f"Error loading {pkl_file}: {e}")
                    continue

                join_graph = qrep["join_graph"]
                subset_graph = qrep["subset_graph"]

                for subplan_tuple in sorted(subset_graph.nodes()):
                    if len(subplan_tuple) < 2:
                        continue

                    total_subquery_counts += 1

                    sorted_aliases = sorted(list(subplan_tuple))
                    sub_graph = join_graph.subgraph(subplan_tuple)
                    
                    edges_info = []
                    for u, v, data in sub_graph.edges(data=True):
                        if u > v: u, v = v, u
                        cond = normalize_condition(data.get("join_condition", ""))
                        edges_info.append(f"{u}|{v}|{cond}")
                    
                    edges_info.sort()
                    join_sig_str = "||".join(edges_info)
                    template_key = (tuple(sorted_aliases), join_sig_str)

                    if template_key not in temp_groups:
                        temp_groups[template_key] = sub_graph.copy()

                    self.template_query_count[template_key] += 1

        print(f"Processed {total_files} query files.")
        print(f"Have {total_subquery_counts} subqueries in total.")
        print(f"Found {len(temp_groups)} distinct unique join templates (ignoring predicates).")
        
        self.unique_templates = temp_groups

    
    def load_and_group_workload_job(self):
        """
        遍历文件，提取唯一的 Join Template 结构。
        这里去掉了谓词处理，只保留图结构的分组。
        """
        print(f"Loading workload from: {self.base_query_dir}")
        temp_groups = {} # Key: (aliases_tuple, join_sig_str), Value: subgraph

        total_files = 0

        total_subquery_counts = 0

        pkl_files = sorted(glob.glob(os.path.join(self.base_query_dir, "*.pkl")))
        
        for pkl_file in pkl_files:
            try:
                qrep = load_qrep(pkl_file)
                total_files += 1
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue

            join_graph = qrep["join_graph"]
            subset_graph = qrep["subset_graph"]

            for subplan_tuple in sorted(subset_graph.nodes()):
                if len(subplan_tuple) < 2:
                    continue

                total_subquery_counts += 1

                sorted_aliases = sorted(list(subplan_tuple))
                sub_graph = join_graph.subgraph(subplan_tuple)
                
                edges_info = []
                for u, v, data in sub_graph.edges(data=True):
                    if u > v: u, v = v, u
                    cond = normalize_condition(data.get("join_condition", ""))
                    edges_info.append(f"{u}|{v}|{cond}")
                
                edges_info.sort()
                join_sig_str = "||".join(edges_info)
                template_key = (tuple(sorted_aliases), join_sig_str)

                if template_key not in temp_groups:
                    temp_groups[template_key] = sub_graph.copy()

                self.template_query_count[template_key] += 1

        print(f"Processed {total_files} query files.")
        print(f"Have {total_subquery_counts} subqueries in total.")
        print(f"Found {len(temp_groups)} distinct unique join templates (ignoring predicates).")
        
        self.unique_templates = temp_groups

    def build_trie_and_count(self):
        """
        对提取出的唯一 Template 构建 Trie 树并计数
        """
        print("\nBuilding Trie from Join Templates...")

        cnt = 0
        
        for idx, (key, sub_graph) in enumerate(self.unique_templates.items()):
            aliases = list(key[0])

            # # 只要所有表基数都小于1000000的模版
            # all_small = True
            # for alias in aliases:
            #     real_name = sub_graph.nodes[alias]['real_name']
            #     card = TABLE_CARD.get(real_name, float("inf"))
            #     if card >= 3000000:
            #         all_small = False
            #         cnt += 1
            #         break
            # if not all_small:
            #     continue

            try:
                path_signature = get_deterministic_execution_plan(sub_graph, aliases)
            except Exception as e:
                continue

            template_id = f"T_{idx}" 
            self.trie.template_freq[template_id] = self.template_query_count[key]
            self.trie.insert(template_id, path_signature)

        # 3. 计算结果
        leaf_count, total_count = self.trie.count_nodes()
        
        print("=" * 40)
        print(f"Final Result Analysis")
        print("=" * 40)
        print(f"Total Unique Templates : {len(self.unique_templates)}")
        print(f"Trie Leaf Nodes        : {leaf_count}")
        print(f"Trie Total Nodes        : {total_count}")
        print(f"Minimum Chains Needed  : {leaf_count}")
        print(f"Optimization Ratio     : {100 * (1 - leaf_count / len(self.unique_templates)):.2f}% reduction")
        print("=" * 40)

        print(cnt)

        subtree_stats = self.trie.count_root_subtrees()

        print("\nPer-root-subtree statistics:")
        for step, stats in subtree_stats.items():
            print(f"{step}: "
                f"nodes={stats['nodes']}, "
                f"templates={stats['templates']}, "
                f"leaves={stats['leaves']}, "
                f"queries={stats['queries']}")
            
        self.trie.analyze_wanderjoin_efficiency()
            

    def load_one_template(self, template_name):
        """
        解析某一个完整查询 template（如 1a / 2b）
        返回：
            unique_templates
            template_query_count
        """
        temp_groups = {}
        template_query_count = defaultdict(int)

        input_template_dir = os.path.join(self.base_query_dir, template_name)
        pkl_files = sorted(glob.glob(os.path.join(input_template_dir, "*.pkl")))

        for pkl_file in pkl_files:
            try:
                qrep = load_qrep(pkl_file)
            except Exception:
                continue

            join_graph = qrep["join_graph"]
            subset_graph = qrep["subset_graph"]

            for subplan_tuple in sorted(subset_graph.nodes()):
                if len(subplan_tuple) < 2:
                    continue

                sorted_aliases = tuple(sorted(subplan_tuple))
                sub_graph = join_graph.subgraph(subplan_tuple)

                edges_info = []
                for u, v, data in sub_graph.edges(data=True):
                    if u > v:
                        u, v = v, u
                    cond = normalize_condition(data.get("join_condition", ""))
                    edges_info.append(f"{u}|{v}|{cond}")

                edges_info.sort()
                join_sig_str = "||".join(edges_info)
                template_key = (sorted_aliases, join_sig_str)

                if template_key not in temp_groups:
                    temp_groups[template_key] = sub_graph.copy()

                template_query_count[template_key] += 1

        return temp_groups, template_query_count
    
    def analyze_one_template(self, template_name):
        print(f"\nAnalyzing template {template_name}")

        unique_templates, template_query_count = \
            self.load_one_template(template_name)

        trie = JoinPathTrie()

        for idx, (key, sub_graph) in enumerate(unique_templates.items()):
            aliases = list(key[0])

            try:
                path_signature = get_deterministic_execution_plan(sub_graph, aliases)
            except Exception:
                continue

            template_id = f"T_{idx}"
            trie.template_freq[template_id] = template_query_count[key]
            trie.insert(template_id, path_signature)

        leaf_cnt, total_cnt = trie.count_nodes()

        print(f"Templates: {len(unique_templates)}")
        print(f"Leaf nodes: {leaf_cnt}")
        print(f"Total nodes: {total_cnt}")

        subtree_stats = trie.count_root_subtrees()
        return subtree_stats

    def analyze_all_templates(self):
        template_dirs = sorted([
            d for d in os.listdir(self.base_query_dir)
            if os.path.isdir(os.path.join(self.base_query_dir, d))
        ])

        for template_name in template_dirs:
            if template_name == "7a" and self.skip_7a:
                continue

            stats = self.analyze_one_template(template_name)

            print(f"Per-root stats for {template_name}:")
            for step, s in stats.items():
                print(
                    f"{step}: nodes={s['nodes']}, "
                    f"templates={s['templates']}, "
                    f"leaves={s['leaves']}, "
                    f"queries={s['queries']}"
                )



if __name__ == "__main__":
    # BASE_DIR = "/data1/xuyining/CEB/my_queries_all/half_ceb_full"
    BASE_DIR = "/data2/xuyining/Sampler/mscn/queries/ceb-imdb"
    # BASE_DIR = "/data1/xuyining/Sampler/mscn/queries/joblight_train/joblight-train-all"
    
    analyzer = WorkloadAnalyzer(BASE_DIR, skip_7a=True)

    t1 = time.time()
    
    analyzer.load_and_group_workload_ceb()

    print(f"Load and parse workload in {time.time() - t1:.2f}s.")

    t2 = time.time()

    if analyzer.unique_templates:
        analyzer.build_trie_and_count()
    else:
        print("No templates found to analyze.")

    print(f"Build trie and count leaves in {time.time() - t2:.2f}s.")

    # analyzer.analyze_all_templates()