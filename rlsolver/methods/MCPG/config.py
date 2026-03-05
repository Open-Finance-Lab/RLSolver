import enum
import torch as th

from rlsolver.methods.config import calc_device


class Problem(enum.Enum):
    maxcut = "maxcut"
    maxcut_edge = "maxcut_edge"
    maxsat = "maxsat"
    MIMO = "MIMO"
    ncheegercut = "ncheegercut"
    partial_maxsat = "partial_maxsat"
    qubo = "qubo"
    qubo_bin = "qubo_bin"
    rcheegercut = "rcheegercut"


GPU_ID: int = 0
PROBLEM = Problem.maxcut
DEVICE: th.device = calc_device(GPU_ID)


class ConfigMaxcut:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 800
        self.reset_epoch_num = 80
        self.total_mcmc_num = 130
        self.repeat_times = 120
        self.num_ls = 3


class ConfigMaxcutEdge:
    def __init__(self):
        self.lr_init = 0.15
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 200
        self.reset_epoch_num = 80
        self.total_mcmc_num = 130
        self.repeat_times = 120
        self.num_ls = 3


class ConfigMaxcutLarge:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 1600
        self.reset_epoch_num = 160
        self.total_mcmc_num = 130
        self.repeat_times = 120
        self.num_ls = 2


class ConfigMaxsat:
    def __init__(self):
        self.lr_init = 0.1
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 80
        self.reset_epoch_num = 80
        self.total_mcmc_num = 400
        self.repeat_times = 120
        self.num_ls = 5


class ConfigMIMO:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 30
        self.reset_epoch_num = 80
        self.total_mcmc_num = 150
        self.repeat_times = 100
        self.num_ls = 4

# only for MIMO
class ConfigLocalMimo:
    snr: int = 2
    size: int = 180

class ConfigNcheegercut:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 800
        self.reset_epoch_num = 80
        self.total_mcmc_num = 140
        self.repeat_times = 120
        self.num_ls = 2


class ConfigPartialMaxsat:
    def __init__(self):
        self.lr_init = 0.1
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 400
        self.reset_epoch_num = 80
        self.total_mcmc_num = 600
        self.repeat_times = 80
        self.num_ls = 2


class ConfigQuboBin:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 1600
        self.reset_epoch_num = 800
        self.total_mcmc_num = 120
        self.repeat_times = 60
        self.num_ls = 2


class ConfigQubo:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 1600
        self.reset_epoch_num = 130
        self.total_mcmc_num = 120
        self.repeat_times = 60
        self.num_ls = 3


class ConfigRcheegercut:
    def __init__(self):
        self.lr_init = 0.25
        self.regular_init = 0
        self.sample_epoch_num = 8
        self.max_epoch_num = 800
        self.reset_epoch_num = 80
        self.total_mcmc_num = 140
        self.repeat_times = 120
        self.num_ls = 2


def class_to_dict_recursive(obj):
    """递归转换方法，支持嵌套对象转换"""
    if not hasattr(obj, '__dict__'):
        return obj

    result = {}
    for key, value in obj.__dict__.items():
        if hasattr(value, '__dict__'):
            result[key] = class_to_dict_recursive(value)
        elif isinstance(value, (list, tuple, set)):
            result[key] = [class_to_dict_recursive(item) for item in value]
        else:
            result[key] = value
    return result


def update_config_for_maxsat_partial_maxsat(problem: Problem, config, nvar):
    if problem == Problem.maxsat:
        num_epochs = 10
        if nvar >= 3000:
            num_epochs = 16
            config.num_ls = 10
        if nvar >= 4000:
            num_epochs = 20
            config.num_ls = 10

        config.max_epoch_num = (num_epochs - 1) * config.sample_epoch_num + 1
    elif problem == Problem.partial_maxsat:
        num_epochs = 35
        config.num_ls = 2
        if nvar >= 700:
            config.total_mcmc_num = 900
        elif nvar >= 1000:
            config.total_mcmc_num = 1200
        else:
            config.total_mcmc_num = 600
            num_epochs = 50
        config.repeat_times = 80

        config.max_epoch_num = (num_epochs - 1) * config.sample_epoch_num + 1
