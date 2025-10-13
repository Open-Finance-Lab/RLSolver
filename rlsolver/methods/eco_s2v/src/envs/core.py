from rlsolver.methods.eco_s2v.config import *

if ALG == Alg.eco or ALG == Alg.s2v:
    from rlsolver.methods.eco_s2v.src.envs.spinsystem import SpinSystemFactory
elif ALG == Alg.eco_torch:
    from rlsolver.methods.eco_s2v.src.envs.spinsystem_torch import SpinSystemFactory
elif ALG == Alg.peco:
    from rlsolver.methods.eco_s2v.src.envs.spinsystem_peco import SpinSystemFactory


def make(id2, *args, **kwargs):
    if id2 == "SpinSystem":
        env = SpinSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env
