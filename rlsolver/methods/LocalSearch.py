import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import torch as th


TEN = th.Tensor

def update_xs_by_vs(xs0, vs0, xs1, vs1, if_maximize: bool = True):
    """
    并行的子模拟器数量为 num_sims, 解x 的节点数量为 num_nodes
    xs: 并行数量个解x,xs.shape == (num_sims, num_nodes)
    vs: 并行数量个解x对应的 objective value. vs.shape == (num_sims, )

    更新后，将xs1，vs1 中 objective value数值更高的解x 替换到xs0，vs0中
    如果被更新的解的数量大于0，将返回True
    """
    good_is = vs1.ge(vs0) if if_maximize else vs1.le(vs0)
    xs0[good_is] = xs1[good_is]
    vs0[good_is] = vs1[good_is]
    return good_is.shape[0]

class LocalSearch:
    def __init__(self, simulator, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        vs = self.simulator.calculate_obj_values(xs=xs)

        self.good_xs = xs
        self.good_vs = vs
        self.num_sims = xs.shape[0]
        return vs

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

    # self.simulator是并行
    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs_raw = sim.calculate_obj_values_for_loop(prev_xs, if_sum=False)
        prev_vs = prev_vs_raw.sum(dim=1)

        thresh = None
        for _ in range(num_iters):
            '''flip randomly with ws(weights)'''
            ws = sim.n0_num_n1 - (4 if sim.if_bidirectional else 2) * prev_vs_raw
            ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]

            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * (ws_std.float() * noise_std)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)

            xs = prev_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])
            vs = sim.calculate_obj_values(xs)

            update_xs_by_vs(prev_xs, prev_vs, xs, vs)

        '''addition'''
        for i in range(sim.num_nodes):
            xs1 = prev_xs.clone()
            xs1[:, i] = th.logical_not(xs1[:, i])
            vs1 = sim.calculate_obj_values(xs1)

            update_xs_by_vs(prev_xs, prev_vs, xs1, vs1)

        num_update = update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs)
        return self.good_xs, self.good_vs, num_update
