

import logging
import os
import time
import datetime
import pytz
import re

import numpy as np

from config import *
from TORCH_OBJECTS import *


########################################
# Get_Logger
########################################
tz = pytz.timezone("Asia/Seoul")


def timetz(*args):
    return datetime.datetime.now(tz).timetuple()


def Get_Logger(SAVE_FOLDER_NAME):
    # make_dir
    #######################################################
    prefix = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M__")
    result_folder_no_postfix = "./result/{}".format(prefix + SAVE_FOLDER_NAME)

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(result_folder_path)  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('{}/log.txt'.format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)

    return logger, result_folder_path


def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = '{}/log.txt'.format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + '[^\n]+', line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command


########################################
# Average_Meter
########################################

class Average_Meter:
 
    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = torch.tensor(0.).to(device)
        self.count = 0

    def push(self, some_tensor, n_for_rank_0_tensor=None):
        assert not some_tensor.requires_grad # You get Memory error, if you keep tensors with grad history
        
        rank = len(some_tensor.shape)

        if rank == 0: # assuming "already averaged" Tensor was pushed
            self.sum += some_tensor * n_for_rank_0_tensor
            self.count += n_for_rank_0_tensor
            
        else:
            self.sum += some_tensor.sum()
            self.count += some_tensor.numel()

    def peek(self):
        average = (self.sum / self.count).tolist()
        return average

    def result(self):
        average = (self.sum / self.count).tolist()
        self.reset()
        return average





########################################
# View NN Parameters
########################################

def get_n_params1(model):
    pp = 0
    for p in list(model.parameters()):
        nn_count = 1
        for s in list(p.size()):
            nn_count = nn_count * s
        pp += nn_count
        print(nn_count)
        print(p.shape)
    print("Total: {:d}".format(pp))


def get_n_params2(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def get_n_params3(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_structure(model):
    print(model)


