import time
import random
from collections import defaultdict
import psycopg2

class JoinSamplingEngine:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # global_cache 结构: 
        # { alias: { 
        #     'rows': { pk_id: {'_bmp': int, 'col1': val...} }, 
        #     'indexes': { col: { val: [pk_ids] } } 
        #   } 
        # }
        self.global_cache = {}

        # 上界缓存，结构: {node_name: {tuple_id: weight}}
        self.memo = defaultdict(dict)

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()

    def _translate_pid_bitmap(self, pid_bitmap_str, pid_map, global_mask_int):
        """将局部 Bitmap 转为全局 QID Bitmap (int)"""
        result_mask = global_mask_int
        if not pid_bitmap_str: return result_mask
        pos = pid_bitmap_str.find('1')
        while pos != -1:
            result_mask |= pid_map.get(pos, 0)
            pos = pid_bitmap_str.find('1', pos + 1)
        return result_mask

    def preload_data(self, tables_info, template_data, workload_name=""):
        """
        加载数据 (逻辑保持不变，确保所有可能用到的连接列都建立了索引)
        """
        self.connect()
        self.global_cache.clear() # 根据需要决定是否清空

        pid_map_full = template_data.get('pid_map', {})
        global_map_full = template_data.get('global_map', {})
        
        # print(f"  [Engine] Preloading {len(tables_info)} tables...")

        for t_info in tables_info:
            alias = t_info['alias']
            if alias in self.global_cache: continue 

            real_name = t_info['real_name']
            join_keys = t_info['join_keys']
            
            my_pid_map = pid_map_full.get(alias, {})
            my_global_mask = global_map_full.get(alias, 0)
            sidecar_name = f"{real_name}_anno_idx_{workload_name}" if workload_name else f"{real_name}_anno_idx"

            cols_str = ", ".join([f"t.{c}" for c in join_keys])

            sql = f"""
                SELECT {cols_str}, s.query_anno::text
                FROM {real_name} t
                JOIN {sidecar_name} s ON t.id = s.query_anno_id
            """
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            
            row_map = {}
            indexes = defaultdict(lambda: defaultdict(list))
            col_name_to_idx = {name: i for i, name in enumerate(join_keys)}
            id_idx = col_name_to_idx['id']

            for r in rows:
                pk_id = str(r[id_idx])
                raw_bitmap = r[-1]
                qid_mask = self._translate_pid_bitmap(raw_bitmap, my_pid_map, my_global_mask)
                
                row_data = {'_bmp': qid_mask}
                for col, idx in col_name_to_idx.items():
                    val = str(r[idx])
                    row_data[col] = val
                    indexes[col][val].append(pk_id)
                
                row_map[pk_id] = row_data
            
            self.global_cache[alias] = {
                'rows': row_map,
                'indexes': indexes,
                'real_name': real_name
            }

    def _get_candidates(self, alias, conds, context_data):
        """
        获取候选行 ID。
        """
        candidates = None
        
        # conds结构: [(my_col, target_alias, target_col), ...]
        for my_col, target_alias, target_col in conds:
            if target_alias not in context_data or target_col not in context_data[target_alias]:
                print(f"  [Warning] Missing context for {target_alias}.{target_col}")
                return []
            
            target_val = context_data[target_alias][target_col]

            matched = self.global_cache[alias]['indexes'][my_col].get(target_val, [])
            
            if candidates is None:
                candidates = set(matched)
            else:
                candidates &= set(matched) # 多个join条件取交集，现在的workload应该只有一个条件
            
            if not candidates: break
            
        return list(candidates) if candidates else []

    def _compute_subtree_weight(self, tree_node, context_data):
        """
        递归计算子树权重 (Tree-based Weight Computation)
        """
        alias = tree_node['alias']
        conds = tree_node['conds']
        children = tree_node.get('children', [])
        
        # 1. 获取当前节点的所有候选
        candidates = self._get_candidates(alias, conds, context_data)
        if not candidates: return 0.0
        
        total_weight = 0.0
        
        # 2. 遍历候选，累加权重
        for cid in candidates:
            if cid in self.memo[alias]:
                total_weight += self.memo[alias][cid]
                continue

            # 将当前选中行加入上下文，供子节点查询
            context_data[alias] = self.global_cache[alias]['rows'][cid]
            
            # 当前分支的权重 = 所有子分支权重的乘积 (Product)
            node_weight = 1.0
            is_valid = True
            
            for child_node in children:
                child_w = self._compute_subtree_weight(child_node, context_data)
                if child_w == 0:
                    is_valid = False
                    break
                node_weight *= child_w
            
            del context_data[alias] # 回溯
            
            if is_valid:
                self.memo[alias][cid] = node_weight
                total_weight += node_weight
                
        return total_weight

    def _sample_subtree_recursive(self, tree_node, context_data):
        """
        递归采样 (Tree-based Sampling)
        返回: 当前子树采样到的 Bitmap (OR 聚合)
        """
        alias = tree_node['alias']
        conds = tree_node['conds']
        children = tree_node.get('children', [])
        
        # 1. 获取候选
        candidates = self._get_candidates(alias, conds, context_data)
        if not candidates: return None
        
        # 2. 计算候选权重
        weighted_cands = []
        sum_w = 0.0
        
        for cid in candidates:
            context_data[alias] = self.global_cache[alias]['rows'][cid]
            
            w = 1.0
            is_valid = True
            for child_node in children:
                # TODO: 加缓存
                cw = self._compute_subtree_weight(child_node, context_data)
                if cw == 0:
                    is_valid = False
                    break
                w *= cw
            
            del context_data[alias]
            
            if is_valid:
                sum_w += w
                weighted_cands.append((cid, w))
        
        if sum_w == 0: return None
        
        # 3. 轮盘赌选择
        r = random.uniform(0, sum_w)
        curr = 0.0
        sel_cid = None
        for cid, w in weighted_cands:
            curr += w
            if r <= curr:
                sel_cid = cid
                break
        if not sel_cid: sel_cid = weighted_cands[-1][0]
        
        # 4. 递归采样所有子分支
        row_data = self.global_cache[alias]['rows'][sel_cid]
        context_data[alias] = row_data
        
        final_bitmap = row_data['_bmp']
        
        # 对于选中的行，我们需要合并所有子节点的采样结果 (Union/OR)
        for child_node in children:
            child_bmp = self._sample_subtree_recursive(child_node, context_data)
            if child_bmp is None:
                del context_data[alias]
                return None # 只要有一个子分支断了，整个样本作废
            
            final_bitmap &= child_bmp
            
        del context_data[alias] # 回溯
        return final_bitmap

    def sample_extensions(self, current_tuple_ids, join_tree, k_samples=5):
        """
        [Algorithm 4 核心入口]
        
        Args:
            current_tuple_ids: dict {alias: pk_id}, 即 T 元组
            join_tree: dict, 描述 Lookahead 的树结构。
                       Root 必须是 Rj。
                       {
                           'alias': 'Rj',
                           'conds': [('col', 'T_table', 'col')...],
                           'children': [ ... ]
                       }
            k_samples: 采样次数
            
        Returns:
            dict: { rj_candidate_id: aggregated_score_bitmap }
        """
        # 1. 初始化上下文 (把 T 放入 context)
        context_data = {}
        for alias, pk_id in current_tuple_ids.items():
            pk_str = str(pk_id)
            if alias in self.global_cache:
                if pk_str in self.global_cache[alias]['rows']:
                    context_data[alias] = self.global_cache[alias]['rows'][pk_str]
                else:
                    # T 中的某些表可能未预加载 (如果 Lookahead 不依赖它们，则无所谓)
                    pass
        
        # 2. 我们需要对 Rj 的每个候选项进行评估
        # 为了高效，我们手动展开第一层 (Rj)，然后对选中的 Rj 递归 Lookahead
        
        rj_alias = join_tree['alias']
        rj_conds = join_tree['conds']
        rj_children = join_tree.get('children', [])
        
        # 获取 Rj 候选 (依赖于 T)
        rj_candidates = self._get_candidates(rj_alias, rj_conds, context_data)
        if not rj_candidates: return {}
        
        # 3. 计算 Rj 候选的权重
        weighted_rj = []
        total_weight = 0.0
        
        for cid in rj_candidates:
            context_data[rj_alias] = self.global_cache[rj_alias]['rows'][cid]
            
            w = 1.0
            is_valid = True
            for child in rj_children:
                # 递归计算子树权重
                cw = self._compute_subtree_weight(child, context_data)
                if cw == 0:
                    is_valid = False
                    break
                w *= cw
                
            del context_data[rj_alias]
            
            if is_valid:
                total_weight += w
                weighted_rj.append((cid, w))
                
        if total_weight == 0: return {}
        
        # 4. 执行 k 次采样
        # 每次采样我们都从加权后的 Rj 候选集中选一个，然后跑完剩下的树
        
        rj_extensions = defaultdict(int)
        
        for _ in range(k_samples):
            # 4.1 选 Rj
            r = random.uniform(0, total_weight)
            curr = 0.0
            sel_rj = None
            for cid, w in weighted_rj:
                curr += w
                if r <= curr:
                    sel_rj = cid
                    break
            if not sel_rj: sel_rj = weighted_rj[-1][0]
            
            # 4.2 递归采子树
            context_data[rj_alias] = self.global_cache[rj_alias]['rows'][sel_rj]
            rj_base_bmp = context_data[rj_alias]['_bmp']
            
            success = True
            sample_bmp = rj_base_bmp
            
            for child in rj_children:
                cb = self._sample_subtree_recursive(child, context_data)
                if cb is None:
                    success = False
                    break
                sample_bmp &= cb
            
            del context_data[rj_alias]
            
            if success:
                # 聚合分数
                rj_extensions[sel_rj] |= sample_bmp
                
        return rj_extensions