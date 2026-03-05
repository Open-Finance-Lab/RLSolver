import torch

from model import simple
from sampling import sampler_select, sample_initializer
from config import update_config_for_maxsat_partial_maxsat
from config import Problem, DEVICE


def base_solver(problem: Problem, num_vars, config, data, verbose=False):
    device = DEVICE

    update_config_for_maxsat_partial_maxsat(problem, config, num_vars)

    sampler = sampler_select(problem)

    change_times = int(num_vars / 10)  # transition times for metropolis sampling

    net = simple(num_vars)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr_init)

    start_samples = None
    for epoch in range(config.max_epoch_num):

        if epoch % config.reset_epoch_num == 0:
            net.to(device).reset_parameters()
            regular = config.regular_init

        net.train()
        if epoch <= 0:
            return_dict = net(regular, None, None)
        else:
            return_dict = net(regular, start_samples, value)

        return_dict["loss"][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # get start samples
        if epoch == 0:
            probs = (torch.zeros(num_vars) + 0.5).to(device)
            tensor_probs = sample_initializer(problem, probs, config, data=data)
            temp_max, temp_max_info, temp_start_samples, value = sampler(data, tensor_probs, probs, config.num_ls, 0, config.total_mcmc_num)
            now_max_res = temp_max
            now_max_info = temp_max_info
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config.repeat_times)
            start_samples = temp_start_samples.t().to(device)

        # get samples
        if epoch % config.sample_epoch_num == 0 and epoch > 0:
            probs = return_dict["output"][0]
            probs = probs.detach()
            temp_max, temp_max_info, start_samples_temp, value = sampler(data, tensor_probs, probs, config.num_ls, change_times, config.total_mcmc_num)
            # update now_max
            for i0 in range(config.total_mcmc_num):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min = min(now_max_res).item()
            now_min_index = torch.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config.repeat_times)
            # construct the start point for next iteration
            start_samples = start_samples_temp.t()
            if verbose:
                if problem in [Problem.maxsat, Problem.partial_maxsat] and len(data.pdata) == 7:
                    res = max(now_max_res).item()
                    if res > data.pdata[5] * data.pdata[6]:
                        res -= data.pdata[5] * data.pdata[6]
                        print("obj {:.3f}".format(res))
                else:
                    print("obj {:.3f}".format((max(now_max_res).item())))
        del (return_dict)

    total_max = now_max_res
    best_sort = torch.argsort(now_max_res, descending=True)
    solution = torch.squeeze(now_max_info[:, best_sort[0]])

    obj = max(total_max).item()
    return obj, solution, now_max_res, now_max_info
