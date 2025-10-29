import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from rlsolver.methods.util_read_data import load_mygraph2
from rlsolver.methods.util_read_data import MyGraph
from rlsolver.methods.util import build_adjacency_bool
from rlsolver.methods.util_read_data import build_adjacency_indies
from rlsolver.methods.util_read_data import calc_num_nodes_in_mygraph
from rlsolver.methods.util_read_data import update_xs_by_vs
from rlsolver.methods.util import gpu_info_str, evolutionary_replacement
from rlsolver.methods.L2A.TNCO_simulator import *
from rlsolver.methods.L2A.TNCO_local_search import *
from rlsolver.methods.util import show_gpu_memory, reset_parameters_of_model
from rlsolver.methods.L2A.config import ConfigPolicyL2A
from torch.nn.utils import clip_grad_norm_

TEN = th.Tensor


class EnvMaxcut:
    def __init__(self, sim_name: str = 'max_cut', mygraph: MyGraph = (),
                 device=th.device('cpu'), if_bidirectional: bool = False):
        self.device = device
        self.sim_name = sim_name
        self.int_type = int_type = th.long
        self.if_maximize = True
        self.if_bidirectional = if_bidirectional

        '''load graph'''
        mygraph: MyGraph = mygraph if mygraph else load_mygraph2(graph_name=sim_name)

        '''建立邻接矩阵'''
        self.adjacency_bool = build_adjacency_bool(mygraph=mygraph, if_bidirectional=True).to(device)

        '''建立邻接索引'''
        n0_to_n1s, n0_to_dts = build_adjacency_indies(mygraph=mygraph, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_nodes = calc_num_nodes_in_mygraph(mygraph)
        self.num_edges = len(mygraph)
        self.adjacency_indies = n0_to_n1s

        '''基于邻接索引，建立基于边edge的索引张量：(n0_ids, n1_ids)是所有边(第0个, 第1个)端点的索引'''
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]
        self.n0_num_n1 = th.tensor([n1s.shape[0] for n1s in n0_to_n1s], device=device)[None, :]

    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
        num_sims = xs.shape[0]  # 并行维度，环境数量。xs, vs第一个维度， dim0 , 就是环境数量
        if num_sims != self.sim_ids.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_sims, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_sims, 1)
            self.sim_ids = self.sim_ids[0:1] + th.arange(num_sims, dtype=self.int_type, device=self.device)[:, None]

        values = xs[self.sim_ids, self.n0_ids] ^ xs[self.sim_ids, self.n1_ids]
        if if_sum:
            values = values.sum(1)
        if self.if_bidirectional:
            values = values // 2
        return values

    def calculate_obj_values_for_loop(self, xs: TEN, if_sum: bool = True) -> TEN:  # 代码简洁，但是计算效率低
        num_sims, num_nodes = xs.shape
        values = th.zeros((num_sims, num_nodes), dtype=self.int_type, device=self.device)
        for node0 in range(num_nodes):
            node1s = self.adjacency_indies[node0]
            if node1s.shape[0] > 0:
                values[:, node0] = (xs[:, node0, None] ^ xs[:, node1s]).sum(dim=1)

        if if_sum:
            values = values.sum(dim=1)
        if self.if_bidirectional:
            values = values.float() / 2
        return values

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs

    def local_search_inplace(self, good_xs: TEN, good_vs: TEN,
                             num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):

        vs_raw = self.calculate_obj_values_for_loop(good_xs, if_sum=False)
        good_vs = vs_raw.sum(dim=1).long() if good_vs.shape == () else good_vs.long()
        ws = self.n0_num_n1 - (2 if self.if_bidirectional else 1) * vs_raw
        ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]
        rd_std = ws_std.float() * noise_std
        spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
        thresh = th.kthvalue(spin_rand, k=self.num_nodes - num_spin, dim=1)[0][:, None]

        for _ in range(num_iters):
            '''flip randomly with ws(weights)'''
            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
            spin_mask = spin_rand.gt(thresh)

            xs = good_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])
            vs = self.calculate_obj_values(xs)

            update_xs_by_vs(good_xs, good_vs, xs, vs, if_maximize=self.if_maximize)

        '''addition'''
        for i in range(self.num_nodes):
            xs1 = good_xs.clone()
            xs1[:, i] = th.logical_not(xs1[:, i])
            vs1 = self.calculate_obj_values(xs1)

            update_xs_by_vs(good_xs, good_vs, xs1, vs1, if_maximize=self.if_maximize)
        return good_xs, good_vs


'''check'''


def find_best_num_sims_maxcut():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    calculate_obj_func = 'calculate_obj_values'
    graph_name = 'gset_14'
    num_sims = 2 ** 16
    num_iter = 2 ** 6
    # calculate_obj_func = 'calculate_obj_values_for_loop'
    # graph_name = 'gset_14'
    # num_sims = 2 ** 13
    # num_iter = 2 ** 9

    if os.name == 'nt':
        graph_name = 'powerlaw_64'
        num_sims = 2 ** 4
        num_iter = 2 ** 3

    graph = load_mygraph2(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = EnvMaxcut(sim_name=graph_name, mygraph=graph, device=device, if_bidirectional=False)

    print('find the best num_sims')
    from math import ceil
    for j in (1, 1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        _num_sims = int(num_sims * j)
        _num_iter = ceil(num_iter * num_sims / _num_sims)

        timer = time.time()
        for i in range(_num_iter):
            xs = simulator.generate_xs_randomly(num_sims=_num_sims)
            vs = getattr(simulator, calculate_obj_func)(xs=xs)
            assert isinstance(vs, TEN)
            # print(f"| {i}  max_obj_value {vs.max().item()}")
        print(f"_num_iter {_num_iter:8}  "
              f"_num_sims {_num_sims:8}  "
              f"UsedTime {time.time() - timer:9.3f}  "
              f"GPU {gpu_info_str(device)}")


def check_env_maxcut():
    gpu_id = -1
    num_sims = 16
    num_nodes = 100
    graph_name = f'PL_{num_nodes}'

    graph = load_mygraph2(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = EnvMaxcut(sim_name=graph_name, mygraph=graph, device=device)

    for i in range(8):
        xs = simulator.generate_xs_randomly(num_sims=num_sims)
        obj = simulator.calculate_obj_values(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass


def check_local_search_maxcut():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    graph_type = 'gset_14'
    mygraph = load_mygraph2(graph_name=graph_type)
    num_nodes = calc_num_nodes_in_mygraph(mygraph)

    show_gap = 4

    num_sims = 2 ** 8
    num_iters = 2 ** 8
    reset_gap = 2 ** 6
    save_dir = f"./{graph_type}_{num_nodes}"

    if os.name == 'nt':
        num_sims = 2 ** 2
        num_iters = 2 ** 5

    '''simulator'''
    sim = EnvMaxcut(mygraph=mygraph, device=device, if_bidirectional=True)
    if_maximize = sim.if_maximize

    '''evaluator'''
    good_xs = sim.generate_xs_randomly(num_sims=num_sims)
    good_vs = sim.calculate_obj_values(xs=good_xs)
    from rlsolver.methods.util_evaluator import Evaluator
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_maximize,
                          x=good_xs[0], v=good_vs[0].item(), )

    for i in range(num_iters):
        evolutionary_replacement(good_xs, good_vs, low_k=2, if_maximize=if_maximize)

        for _ in range(4):
            sim.local_search_inplace(good_xs, good_vs)

        if_show_x = evaluator.record2(i=i, vs=good_vs, xs=good_xs)
        if (i + 1) % show_gap == 0 or if_show_x:
            show_str = f"| cut_value {good_vs.float().mean():8.2f} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {gpu_info_str(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            good_xs = sim.generate_xs_randomly(num_sims=num_sims)
            good_vs = sim.calculate_obj_values(xs=good_xs)

    print(f"\nbest_x.shape {evaluator.best_x.shape}"
          f"\nbest_v {evaluator.best_v}"
          f"\nbest_x_str {evaluator.best_x_str}")


def metropolis_hastings_sampling_TNCO(probs: TEN, start_xs: TEN, num_repeats: int, num_iters: int = -1,
                                      accept_rate: float = 0.25) -> TEN:
    """随机平稳采样 是 metropolis-hastings sampling is:
    - 在状态转移链上 using transition kernel in Markov Chain
    - 使用随机采样估计 Monte Carlo sampling
    - 依赖接受概率去达成细致平稳条件 with accept ratio to satisfy detailed balance condition
    的采样方法。

    工程实现上:
    - 让它从多个随机的起始位置 start_xs 开始
    - 每个起始位置建立多个副本 repeat
    - 循环多次 for _ in range
    - 直到接受的样本数量超过阈值 count >= 再停止

    这种采样方法允许不同的区域能够以平稳的概率采集到符合采样概率的样本，确保样本数量比例符合期望。
    具体而言，每个区域内采集到的样本数与该区域所占的平稳分布概率成正比。
    """
    # metropolis-hastings sampling: Monte Carlo sampling using transition kernel in Markov Chain with accept ratio
    xs = start_xs.repeat(num_repeats, 1)
    ps = probs.repeat(num_repeats, 1)

    num, dim = xs.shape
    device = xs.device
    num_iters = int(dim * accept_rate) if num_iters == -1 else num_iters  # 希望至少有accept_rate的节点的采样结果被接受

    count = 0
    for _ in range(4):  # 迭代4轮后，即便被拒绝的节点很多，也不再迭代了。
        ids = th.randperm(dim, device=device)  # 按随机的顺序，均匀地选择节点进行采样。避免不同节点被采样的次数不同。
        for i in range(dim):
            idx = ids[i]
            chosen_p0 = ps[:, idx]
            chosen_xs = xs[:, idx]
            chosen_ps = th.where(chosen_xs, chosen_p0, 1 - chosen_p0)

            accept_rates = (1 - chosen_ps) / chosen_ps
            accept_masks = th.rand(num, device=device).lt(accept_rates)
            xs[:, idx] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

            count += accept_masks.sum()
            if count >= num * num_iters:
                break
        if count >= num * num_iters:
            break
    return xs


class McmcIterator_TNCO:
    def __init__(self, num_sims: int, num_repeats: int, num_searches: int,
                 graph_type: str = 'graph', nodes_list: list = (), device=th.device('cpu')):
        self.num_sims = num_sims
        self.num_repeats = num_repeats
        self.num_searches = num_searches
        self.device = device
        self.sim_ids = th.arange(num_sims, device=device)

        # build in reset
        self.simulator = EnvTNCO(nodes_list=nodes_list, ban_edges=0, device=self.device)
        self.num_bits = self.simulator.num_bits
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_bits=self.num_bits)
        self.if_maximize = self.searcher.if_maximize

    def reset(self):
        xs = self.simulator.generate_xs_randomly(num_sims=self.num_sims)
        self.searcher.reset(xs)
        for _ in range(self.num_searches * 2):
            self.searcher.random_search(num_iters=8)

        good_xs = self.searcher.good_xs
        good_vs = self.searcher.good_vs
        return good_xs, good_vs

    def step(self, start_xs: TEN, probs: TEN) -> (TEN, TEN):
        xs = metropolis_hastings_sampling_TNCO(probs=probs, start_xs=start_xs, num_repeats=self.num_repeats, num_iters=-1)
        vs = self.searcher.reset(xs)
        for _ in range(self.num_searches):
            xs, vs, num_update = self.searcher.random_search(num_iters=2 ** 3, num_spin=8, noise_std=0.5)
        return xs, vs

    def good(self, full_xs, full_vs) -> (TEN, TEN):
        # update good_xs: use .view() instead of .reshape() for saving GPU memory
        xs_view = full_xs.view(self.num_repeats, self.num_sims, self.num_bits)
        vs_view = full_vs.view(self.num_repeats, self.num_sims)
        ids = vs_view.argmax(dim=0) if self.if_maximize else vs_view.argmin(dim=0)

        good_xs = xs_view[ids, self.sim_ids]
        good_vs = vs_view[ids, self.sim_ids]
        return good_xs, good_vs


def valid_in_single_graph_TNCO(
        args0: ConfigPolicyL2A = None,
        nodes_list: list = None,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    # args0 = args0 if args0 else ConfigPolicy(graph_type='SycamoreN12M14', num_nodes=1)  # todo plan remove num_nodes
    # nodes_list = NodesSycamoreN12M14 if nodes_list is None else nodes_list
    args0 = args0 if args0 else ConfigPolicyL2A(graph_type='SycamoreN53M20', num_nodes=1)  # todo plan remove num_nodes
    nodes_list = NodesSycamoreN53M20 if nodes_list is None else nodes_list

    '''custom'''
    args0.num_iters = 2 ** 6 * 7
    args0.reset_gap = 2 ** 6
    args0.num_sims = 2 ** 4  # LocalSearch 的初始解数量
    args0.num_repeats = 2 ** 4  # LocalSearch 对于每以个初始解进行复制的数量

    if os.name == 'nt':
        args0.num_sims = 2 ** 2
        args0.num_repeats = 2 ** 3

    '''config: graph'''
    graph_type = args0.graph_type
    num_nodes = args0.num_nodes

    '''config: train'''
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''iterator'''
    iterator = McmcIterator_TNCO(num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                                 graph_type=graph_type, nodes_list=nodes_list, device=device)
    num_bits = iterator.num_bits  # todo add num_bits
    if_maximize = iterator.if_maximize

    '''model'''
    # from network import PolicyORG
    # policy_net = PolicyORG(num_bits=num_bits).to(device)
    from rlsolver.methods.L2A.network import PolicyMLP
    policy_net = PolicyMLP(num_bits=num_bits).to(device)
    policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''evaluator'''
    save_dir = f"./ORG_{graph_type}_{num_bits}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = iterator.reset()
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_maximize,
                          x=good_xs[0], v=good_vs[0].item())

    '''loop'''
    th.set_grad_enabled(False)
    lamb_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_i, None, :].float())
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = iterator.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = iterator.good(full_xs=full_xs, full_vs=full_vs)

        advantages = full_vs.float()
        if if_maximize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages.mean() - advantages) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)  # ↓↓↓↓↓↓↓↓↓↓ gradient
        for j in range(num_sgd_steps):
            good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_i, None, :].float())
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lamb_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)  # ↑↑↑↑↑↑↑↑↑ gradient

        '''update good_xs'''
        good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} obj_value {good_vs.min():9.6f} < {good_vs.mean():9.6f}"
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            reset_parameters_of_model(model=policy_net)
            good_xs, good_vs = iterator.reset()

    evaluator.save_record_draw_plot(fig_dpi=300)

if __name__ == '__main__':
    check = True
    if check:
        check_env_maxcut()

    valid = False
    if valid:
        valid_in_single_graph_TNCO()
    # check_local_search()
