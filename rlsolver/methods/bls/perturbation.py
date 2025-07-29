# perturbation.py
import random
import math
def choose_perturbation(omega, T, P0, Q):
    """
    根据 ω、T、P0、Q 计算三种扰动的选择概率
    返回 'random'、'direct1' 或 'direct2'
    """
    # 强制随机扰动条件
    if omega >= T:
        return 'random'

    # 计算定向扰动概率 P
    P = max(P0, math.exp(-omega / T))
    r = random.random()
    if r < P * Q:
        return 'direct1'
    if r < P:
        return 'direct2'
    return 'random'
