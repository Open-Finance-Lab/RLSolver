import enum
import torch as th

def calc_device(gpu_id: int):
    return th.device(f"cuda:{gpu_id}" if th.cuda.is_available() and gpu_id >= 0 else "cpu")


class Problem(enum.Enum):
    maxcut = "maxcut"
    maxcut_edge = "maxcut_edge"
    maxsat = "maxsat"
    mimo = "mimo"
    n_cheegercut = "n_cheegercut"
    pmaxsat = "pmaxsat"
    qubo_bin = "qubo_bin"
    qubo = "qubo"
    r_cheegercut = "r_cheegercut"

GPU_ID: int = 0
PROBLEM = Problem.maxcut
DEVICE: th.device = calc_device(GPU_ID)

class config_maxcut:
    problem_type = "maxcut"
    lr_init = 0.25
    regular_init = 0
    sample_epoch_num = 8
    max_epoch_num = 800
    reset_epoch_num = 80
    total_mcmc_num = 130
    repeat_times = 120
    num_ls = 3


def update_config_for_maxsat_pmaxsat(config, nvar):
    if config["problem_type"] == "maxsat":
        num_epochs = 10
        if nvar >= 3000:
            num_epochs = 16
            config["num_ls"] = 10
        if nvar >= 4000:
            num_epochs = 20
            config["num_ls"] = 10

        config["max_epoch_num"] = (num_epochs - 1) * config["sample_epoch_num"] + 1
    elif config["problem_type"] == "pmaxsat":
        num_epochs = 35
        config["num_ls"] = 2
        if nvar >= 700:
            config["total_mcmc_num"] = 900
        elif nvar >= 1000:
            config["total_mcmc_num"] = 1200
        else:
            config["total_mcmc_num"] = 600
            num_epochs = 50
        config["repeat_times"] = 80

        config["max_epoch_num"] = (num_epochs - 1) * config["sample_epoch_num"] + 1


