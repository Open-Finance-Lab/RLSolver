# data_util.py
import torch
import networkx as nx
import numpy as np
import os
import glob
from tqdm import tqdm


def convert_graph_to_data(graph):
    """优化的图转换函数"""
    edge_list = list(graph.edges(data=True))
    if not edge_list:
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_weight': torch.empty(0, dtype=torch.float32),
            'num_nodes': graph.number_of_nodes()
        }
    
    # 创建双向边
    edges = []
    weights = []
    for u, v, data in edge_list:
        weight = data.get('weight', 1.0)
        edges.extend([[u, v], [v, u]])
        weights.extend([weight, weight])
    
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': graph.number_of_nodes()
    }


def generate_graph(graph_type, n_min=50, n_max=200):
    """生成单个图
    Args:
        graph_type: 'BA', 'ER', 'PL' 之一
        n_min, n_max: 节点数范围
    """
    n = np.random.randint(n_min, n_max + 1)
    
    if graph_type == 'BA':
        # Barabási–Albert图
        m = np.random.randint(3, max(4, min(5, n//10) + 1))  # 每个新节点的边数
        G = nx.barabasi_albert_graph(n, m)
    
    elif graph_type == 'ER':
        # Erdős–Rényi图
        p = np.random.uniform(0.02, 0.1)  # 边概率
        G = nx.erdos_renyi_graph(n, p)
    
    elif graph_type == 'PL':
        # Power-Law图 (使用配置模型)
        gamma = np.random.uniform(2.0, 3.0)  # 幂律指数
        sequence = nx.utils.powerlaw_sequence(n, gamma)
        sequence = [int(d) for d in sequence]
        # 确保度序列和为偶数
        if sum(sequence) % 2 != 0:
            sequence[0] += 1
        G = nx.configuration_model(sequence)
        G = nx.Graph(G)  # 移除多重边和自环
    
    # 添加随机权重
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 10.0)
    
    return convert_graph_to_data(G)


def generate_batch_graphs(batch_size, graph_types=['BA', 'ER', 'PL']):
    """批量生成图数据 - 保留带图类型随机化的版本"""
    graphs = []
    for _ in range(batch_size):
        graph_type = np.random.choice(graph_types)
        graph_data = generate_graph(graph_type)
        graphs.append(graph_data)
    return graphs


def load_graphs_from_directory(directory, config):
    """优化的图加载函数，带缓存功能"""
    # 检查预处理文件
    preprocessed_path = os.path.join(directory, "preprocessed_graphs.pt")
    
    if os.path.exists(preprocessed_path) and not config.force_reload:
        print(f"加载预处理的图数据：{preprocessed_path}")
        return torch.load(preprocessed_path)
    
    print("预处理文件不存在，从原始文件加载...")
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith('dataset_info.txt')]
    
    graph_datas = []
    
    for filepath in tqdm(files[:100], desc="加载并预处理图"):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            n, m = map(int, lines[0].strip().split())
            
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            
            for i in range(1, m + 1):
                parts = lines[i].strip().split()
                u, v = int(parts[0]), int(parts[1])
                weight = float(parts[2]) if len(parts) > 2 else 1.0
                graph.add_edge(u, v, weight=weight)
            
            # 转换为张量格式
            graph_data = convert_graph_to_data(graph)
            graph_datas.append(graph_data)
        except Exception as e:
            print(f"无法加载 {filepath}: {e}")
            continue
    
    # 保存预处理数据
    print(f"保存 {len(graph_datas)} 个预处理图到 {preprocessed_path}")
    torch.save(graph_datas, preprocessed_path)
    
    return graph_datas

def graph_prefetcher(queue, stop_event, batch_size, graph_types=['BA', 'ER', 'PL']):
    """图数据预取进程"""
    while not stop_event.is_set():
        try:
            # 只在队列未满时生成
            if not queue.full():
                batch_graph_datas = generate_batch_graphs(batch_size, graph_types)
                queue.put(batch_graph_datas, timeout=1.0)
        except:
            pass