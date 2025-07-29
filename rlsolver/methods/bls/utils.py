# utils.py
import random
from collections import defaultdict

class BucketSort:
    """
    桶排序结构：维护 gain -> set(nodes)，并快速选取当前 maxgain
    """
    def __init__(self, gains=None):
        # gains: dict node->gain
        self.node_gain = {}
        self.buckets = defaultdict(set)
        self.maxgain = None
        if gains:
            self.reset(gains)

    def reset(self, gains):
        """一次性初始化所有顶点的增益"""
        self.node_gain = gains.copy()
        self.buckets.clear()
        for v, g in gains.items():
            self.buckets[g].add(v)
        self.maxgain = max(self.buckets) if self.buckets else 0

    def update(self, v, new_gain):
        """更新单个节点 v 的增益"""
        old = self.node_gain[v]
        self.buckets[old].remove(v)
        if not self.buckets[old]:
            del self.buckets[old]
            if old == self.maxgain:
                self.maxgain = max(self.buckets) if self.buckets else 0

        self.node_gain[v] = new_gain
        self.buckets[new_gain].add(v)
        if self.maxgain is None or new_gain > self.maxgain:
            self.maxgain = new_gain

    def get_max_nodes(self):
        """返回当前所有 maxgain 对应的节点集合"""
        if not self.buckets:
            return set()
        return self.buckets[self.maxgain]

class TabuList:
    """
    禁忌表：记录 node 被禁忌到哪一迭代释放
    """
    def __init__(self):
        self.expire = {}  # node -> iteration index when tabu 解除

    def forbid(self, v, current_iter, tenure):
        self.expire[v] = current_iter + tenure

    def is_allowed(self, v, current_iter, aspiration=False, best_val=None, move_gain=None, curr_val=None):
        """
        如果当前迭代 >= expire[v]，则允许；
        否则只有在“渴望”情况下（aspiration 且 curr_val+move_gain > best_val）才允许。
        """
        if current_iter >= self.expire.get(v, 0):
            return True
        if aspiration and best_val is not None and move_gain is not None and curr_val is not None:
            return (curr_val + move_gain) > best_val
        return False

def compute_cut_value(G, cut):
    """计算割 cut（dict node->bool）的目标值"""
    val = 0
    for u, v, data in G.edges(data=True):
        if cut[u] != cut[v]:
            val += data.get("weight", 1)
    return val

def compute_gain(G, cut, v):
    delta = 0
    side = cut[v]
    for u in G[v]:
        w = G[v][u].get("weight", 1)
        if cut[u] == side:
            delta += w
        else:
            delta -= w
    return delta