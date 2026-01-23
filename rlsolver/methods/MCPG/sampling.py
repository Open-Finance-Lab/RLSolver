import torch
from torch_scatter import scatter
from torch.distributions.bernoulli import Bernoulli
from config import DEVICE

def sample_initializer(problem_type, probs, config,
                       device=DEVICE, data=None):
    if problem_type in ["r_cheegercut", "n_cheegercut"]:
        samples = torch.zeros(config['total_mcmc_num'], data.num_nodes)
        index = data.sorted_degree_nodes[- config['total_mcmc_num']:]
        for i in range(config['total_mcmc_num']):
            samples[i][index[i]] = 1
        samples = samples.repeat(config['repeat_times'], 1)
        return samples.t()
    m = Bernoulli(probs)
    samples = m.sample([config['total_mcmc_num'] * config['repeat_times']])
    samples = samples.detach().to(device)
    return samples.t()


def sampler_select(problem_type):
    if problem_type == "maxcut":
        return mcpg_sampling_maxcut
    elif problem_type == "maxcut_edge":
        return mcpg_sampling_maxcut_edge
    elif problem_type == "maxsat":
        return mcpg_sampling_maxsat
    elif problem_type == "mimo":
        return mcpg_sampling_mimo
    elif problem_type == "qubo":
        return mcpg_sampling_qubo
    elif problem_type == "qubo_bin":
        return mcpg_sampling_qubo_bin
    elif problem_type == "r_cheegercut":
        return mcpg_sampling_rcheegercut
    elif problem_type == "n_cheegercut":
        return mcpg_sampling_ncheegercut
    else:
        raise (Exception("Unrecognized problem type {}".format(problem_type)))


def metro_sampling(probs, start_status, max_transfer_time,
                   device=DEVICE):
    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain))).to(device)

    probs = probs.detach().to(device)
    samples = start_status.bool().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain*max_transfer_time:
            break
        index_row = torch.randint(low=0, high=num_node, size=[
                                  num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(
            chosen_value, chosen_probs_base, 1-chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs
        r = torch.rand(num_chain, device=device)
        is_accept = (r < accept_rate)
        samples[index_row, index_col] = torch.where(
            is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()

    return samples.float().to(device)


def mcpg_sampling_maxcut(data,
                         start_result, probs,
                         num_ls, change_times, total_mcmc_num,
                         device=DEVICE):
    probs = probs.to(torch.device("cpu"))
    num_nodes = data.num_nodes
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs-0.5)*2+0.5

    # local search
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for node_index in range(0, num_nodes):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]
            neighbor_edge_weight = data.neighbor_edges[node]
            node_temp_v = torch.mm(
                neighbor_edge_weight, graph_probs[neighbor_index])
            node_temp_v = torch.squeeze(node_temp_v)
            node_temp_v += torch.rand(node_temp_v.shape[0],
                                      device=torch.device(device))/4
            graph_probs[node] = (node_temp_v <
                                 data.weighted_degree[node]/2+0.125).int()
        if cnt >= num_ls:
            break

    # maxcut

    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1)*(
        2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    max_cut = expected_cut[index]
    return ((edge_weight_sum-max_cut) / 2), graph_probs[:, index], start, (expected_cut-torch.mean(expected_cut)).to(device)

def mcpg_sampling_maxcut_edge(data,
                         start_result, probs,
                         num_ls, change_times, total_mcmc_num,
                         device=DEVICE):
    probs = probs.to(torch.device("cpu"))
    num_edges = data.num_edges
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times, device)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs%2

    graph_probs = (graph_probs-0.5)*2+0.5

    # local search
    temp = torch.zeros(4,graph_probs.size(dim=1)).to(device)
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for i in range(num_edges):

            index = data.sorted_degree_edges[i]
            node_r = nlr_graph[index]
            node_c = nlc_graph[index]
            edges_r = data.n0_edges[index]
            edges_c = data.n1_edges[index]
            add_0 = data.add[0][index]
            add_1 = data.add[1][index]
            add_2 = data.add[2][index]

            temp_r_v = torch.mm(edges_r,graph_probs[data.n0[index]])
            temp_c_v = torch.mm(edges_c,graph_probs[data.n1[index]])

            temp[1] = temp_r_v +torch.rand(graph_probs.size(dim=1),device=torch.device('cuda:0'))*0.1 + add_0
            temp[2] = temp_c_v +torch.rand(graph_probs.size(dim=1),device=torch.device('cuda:0'))*0.1 + add_1
            temp[0] = temp[1] + temp[2] +torch.rand(graph_probs.size(dim=1),device=torch.device('cuda:0'))*0.1 - add_2

            max_index = torch.argmax(temp, dim=0)
            graph_probs[node_r] = torch.floor(max_index/2)
            graph_probs[node_c] = max_index%2


        if cnt >= num_ls:
            break

    # maxcut
    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1)*(
        2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    max_cut = expected_cut[index]

    return ((edge_weight_sum-max_cut) / 2), graph_probs[:, index], start, (expected_cut-torch.mean(expected_cut)).to(device)



def mcpg_sampling_rcheegercut(data,
                              start_result, probs,
                              num_ls, change_times, total_mcmc_num,
                              device=DEVICE):
    probs = probs.to(torch.device("cpu"))
    nvar = data.num_nodes
    edges = data.edge_index
    nlr_graph, nlc_graph = edges
    graph_probs = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, graph_probs, change_times)
    samples = raw_samples.clone()

    res_cut = ((2 * samples[nlr_graph.type(torch.long)][:] - 1)*(
        2 * samples[nlc_graph.type(torch.long)][:] - 1)).sum(dim=0)
    res_cut[:] = (data.edge_weight_sum - res_cut) / 2
    res_node = samples.sum(dim=0)
    cheeger_cut = res_cut / torch.min(res_node, nvar - res_node)

    for cnt in range(num_ls):
        for node_index in range(nvar):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]

            change_cut_size = torch.sum(samples[neighbor_index], dim=0)
            new_res_cut = res_cut - \
                (2 * samples[node] - 1) * \
                (data.weighted_degree[node] - 2 * change_cut_size)
            new_res_node = res_node - (2 * samples[node] - 1)
            new_cheeger_cut = new_res_cut / \
                torch.min(new_res_node, nvar - new_res_node)
            new_min_node = torch.min(new_res_node, nvar - new_res_node)
            cond = torch.logical_or(
                (cheeger_cut < new_cheeger_cut), (new_min_node < 0.0000001))
            samples[node] = torch.where(cond, samples[node], 1 - samples[node])
            res_cut = torch.where(cond, res_cut, new_res_cut)
            res_node = torch.where(cond, res_node, new_res_node)
            cheeger_cut = torch.where(cond, cheeger_cut, new_cheeger_cut)
    # maxcut
    cheeger_cut_reshape = torch.reshape(cheeger_cut, (-1, total_mcmc_num))
    index = torch.argmin(cheeger_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    min_cheeger_cut = cheeger_cut[index]

    return -min_cheeger_cut, samples[:, index], raw_samples, (cheeger_cut-torch.mean(cheeger_cut)).to(device)


def mcpg_sampling_ncheegercut(data,
                              start_result, probs,
                              num_ls, change_times, total_mcmc_num,
                              device=DEVICE):
    probs = probs.to(torch.device("cpu"))
    nvar = data.num_nodes
    edges = data.edge_index
    nlr_graph, nlc_graph = edges
    graph_probs = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, graph_probs, change_times)
    samples = raw_samples.clone()

    res_cut = ((2 * samples[nlr_graph.type(torch.long)][:] - 1)*(
        2 * samples[nlc_graph.type(torch.long)][:] - 1)).sum(dim=0)
    res_cut[:] = (data.edge_weight_sum - res_cut) / 2
    res_node = samples.sum(dim=0)
    cheeger_cut = res_cut * (1 / res_node + 1 / (nvar - res_node))

    for cnt in range(num_ls):
        for node_index in range(nvar):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]

            change_cut_size = torch.sum(samples[neighbor_index], dim=0)
            new_res_cut = res_cut - \
                (2 * samples[node] - 1) * \
                (data.weighted_degree[node] - 2 * change_cut_size)
            new_res_node = res_node - (2 * samples[node] - 1)
            new_cheeger_cut = new_res_cut * \
                (1 / new_res_node + 1 / (nvar - new_res_node))
            new_min_node = torch.min(new_res_node, nvar - new_res_node)
            cond = torch.logical_or(
                (cheeger_cut < new_cheeger_cut), (new_min_node < 0.0000001))
            samples[node] = torch.where(cond, samples[node], 1 - samples[node])
            res_cut = torch.where(cond, res_cut, new_res_cut)
            res_node = torch.where(cond, res_node, new_res_node)
            cheeger_cut = torch.where(cond, cheeger_cut, new_cheeger_cut)
    # maxcut
    cheeger_cut_reshape = torch.reshape(cheeger_cut, (-1, total_mcmc_num))
    index = torch.argmin(cheeger_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    min_cheeger_cut = cheeger_cut[index]

    return -min_cheeger_cut, samples[:, index], raw_samples, (cheeger_cut-torch.mean(cheeger_cut)).to(device)


def mcpg_sampling_maxsat(
        data,
        start_result, probs,
        num_ls, change_times, total_mcmc_num,
        device=DEVICE):

    nvar, nclause, vi, ci, neg = data.pdata[0:5]
    raw_samples = start_result.clone()
    raw_samples = metro_sampling(
        probs, raw_samples, change_times, device)
    samples = raw_samples.t().clone()
    samples = samples * 2 - 1
    for cnt in range(num_ls):
        for index in range(nvar):
            i = data.ndata[3][index].item()
            cal_sample = samples[:, data.ndata[0][i]] * data.ndata[2][i]
            res_clause = scatter(
                cal_sample, data.ndata[1][i], reduce="max", dim=1)
            res_sample_old = torch.sum(res_clause, dim=1)
            samples[:, i] = - samples[:, i]
            cal_sample = samples[:, data.ndata[0][i]] * data.ndata[2][i]
            res_clause = scatter(
                cal_sample, data.ndata[1][i], reduce="max", dim=1)
            res_sample_new = torch.sum(res_clause, dim=1)
            # ind = (res_sample_new > res_sample_old)
            ind = (res_sample_new > res_sample_old +
                   torch.rand(res_sample_old.shape[0], device=device)-0.5)
            samples[:, i] = torch.where(ind, samples[:, i], -samples[:, i])

    cal_sample = samples[:, vi] * neg
    res_clause = scatter(cal_sample, ci, reduce="max", dim=1)
    res_sample = torch.sum(res_clause, dim=1)
    if len(data.pdata) == 7:
        res_sample += nclause-data.pdata[6]+data.pdata[5]*data.pdata[6]
    else:
        res_sample += nclause
    res_sample = res_sample/2

    res_sample_reshape = torch.reshape(res_sample, (-1, total_mcmc_num))
    index = torch.argmax(res_sample_reshape, dim=0)
    index = torch.tensor(list(range(total_mcmc_num)),
                         device=device) + index*total_mcmc_num
    max_res = res_sample[index]
    samples = (samples + 1) / 2
    return max_res, samples.t()[:, index], raw_samples, -(res_sample - torch.mean(res_sample.float())).to(device)


def mcpg_sampling_mimo(data,
              start_result, probs,
              num_ls, change_times, total_mcmc_num,
              device=DEVICE):
    Sigma = data[0]
    Diag = data[1]
    num_n = data[0].shape[0]
    info = start_result.clone()
    # get probs
    info = metro_sampling(
        probs, info, change_times, device)
    start = info.clone()

    info = (info-0.5)*4  # convert to 2, -2

    # local search
    cnt = 0
    while True:
        for node in range(0, num_n):
            if cnt >= num_ls*num_n:
                break
            cnt += 1
            neighbor_weight = Sigma[node].unsqueeze(0)
            node_temp_v = torch.matmul(
                neighbor_weight, info)
            node_temp_v = torch.squeeze(node_temp_v)
            temp = (node_temp_v < - Diag[node]/2).int()
            info[node] = temp*2-1
        if cnt >= num_ls*num_n:
            break

    # compute value
    expected = (info * torch.matmul(Sigma, info)).sum(dim=0)
    expected += torch.matmul(Diag.unsqueeze(0), info).sum(dim=0)
    expected_cut_reshape = torch.reshape(expected, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    min_res = expected[index]
    info = (info+1)/2
    return -(min_res+data[3]), info[:, index], start, (expected-torch.mean(expected)).to(device)


def mcpg_sampling_qubo(data, start_result, probs, num_ls, change_times, total_mcmc_num,
                       device=DEVICE):
    Q = data['Q']
    nvar = data['nvar']
    raw_samples = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, raw_samples, change_times, device)
    samples = raw_samples.clone()
    samples = samples * 2 - 1
    # local search
    for cnt in range(num_ls):
        for index in range(nvar):
            samples[index] = 0
            res = torch.matmul(Q[index], samples)
            ind = (res > 0)
            samples[index] = 2 * ind - 1
    # compute value
    res_sample = torch.matmul(Q, samples)
    res_sample = torch.sum(torch.mul(samples, res_sample), dim=0)
    res_sample_reshape = torch.reshape(res_sample, (-1, total_mcmc_num))
    index = torch.argmax(res_sample_reshape, dim=0)
    index = torch.tensor(list(range(total_mcmc_num)),
                         device=device) + index*total_mcmc_num
    max_res = res_sample[index]
    samples = (samples + 1) / 2
    return max_res, samples[:, index], raw_samples, -(res_sample - torch.mean(res_sample.float())).to(device)
def mcpg_sampling_qubo_bin(data, start_result, probs, num_ls, change_times, total_mcmc_num,
                       device=DEVICE):
    Q = data['Q']
    nvar = data['nvar']
    raw_samples = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, raw_samples, change_times, device)
    samples = raw_samples.clone()
    # local search
    for cnt in range(num_ls):
        for index in range(nvar):
            samples[index] = 0
            res = torch.matmul(Q[index], samples)
            ind = (res > (- Q[index][index]/2))
            samples[index] = ind
    # compute value
    res_sample = torch.matmul(Q, samples)
    res_sample = torch.sum(torch.mul(samples, res_sample), dim=0)
    res_sample_reshape = torch.reshape(res_sample, (-1, total_mcmc_num))
    index = torch.argmax(res_sample_reshape, dim=0)
    index = torch.tensor(list(range(total_mcmc_num)),
                         device=device) + index*total_mcmc_num
    max_res = res_sample[index]
    return max_res, samples[:, index], raw_samples, -(res_sample - torch.mean(res_sample.float())).to(device)
