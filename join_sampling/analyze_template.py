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

    def root_score(alias):
        real_name = join_graph.nodes[alias]['real_name']
        return (-TABLE_CARD.get(real_name, float("inf")), alias)

    root_table = max(aliases, key=root_score)


    visited = {root_table}

    path_signature = [("ROOT", root_table, "None")]

    while len(visited) < len(aliases):
        candidates = []
        for u in visited:
            for v in join_graph.neighbors(u):
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

        # current_node = queue.pop(0)
        
        # neighbors = list(join_graph.neighbors(current_node))
        # neighbors.sort()

        # for neighbor in neighbors:
        #     if neighbor not in visited:
        #         visited.add(neighbor)
        #         queue.append(neighbor)

        #         edge_data = join_graph.get_edge_data(current_node, neighbor)
        #         raw_condition = edge_data.get("join_condition")

        #         if not raw_condition:
        #             raise ValueError(f"Missing join condition between {current_node} and {neighbor}")

        #         norm_cond = normalize_condition(raw_condition)

        #         # 存的是 (parent, child, cond)，构成了 Trie 的一条边
        #         path_signature.append((current_node, neighbor, norm_cond))
    
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


class WorkloadAnalyzer:
    def __init__(self, base_query_dir, skip_7a=False):
        self.base_query_dir = base_query_dir
        self.skip_7a = skip_7a
        self.unique_templates = {} # {signature_string: join_graph}
        self.trie = JoinPathTrie()

    def load_and_group_workload(self):
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
                    # 单表直接跳过
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

        print(f"Processed {total_files} query files.")
        print(f"Have {total_subquery_counts} subqueries in total.")
        print(f"Found {len(temp_groups)} distinct unique join templates (ignoring predicates).")
        
        self.unique_templates = temp_groups

    def build_trie_and_count(self):
        """
        对提取出的唯一 Template 构建 Trie 树并计数
        """
        print("\nBuilding Trie from Join Templates...")
        
        for idx, (key, sub_graph) in enumerate(self.unique_templates.items()):
            aliases = list(key[0])
            
            try:
                path_signature = get_deterministic_execution_plan(sub_graph, aliases)
            except Exception as e:
                continue

            template_id = f"T_{idx}" 
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


if __name__ == "__main__":
    # BASE_DIR = "/data1/xuyining/CEB/my_queries_all/half_ceb_full"
    BASE_DIR = "/data1/xuyining/CEB-default/queries/ceb-imdb"
    
    analyzer = WorkloadAnalyzer(BASE_DIR, skip_7a=True)

    t1 = time.time()
    
    analyzer.load_and_group_workload()

    print(f"Load and parse workload in {time.time() - t1:.2f}s.")

    t2 = time.time()

    if analyzer.unique_templates:
        analyzer.build_trie_and_count()
    else:
        print("No templates found to analyze.")

    print(f"Build trie and count leaves in {time.time() - t2:.2f}s.")