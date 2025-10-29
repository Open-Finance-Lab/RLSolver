from rlsolver.methods.ECO_S2V.config import *

if ALG == Alg.ECO or ALG == Alg.S2V:
    from rlsolver.methods.ECO_S2V.src.envs.spinsystem import SpinSystemFactory
elif ALG == Alg.ECO_torch:
    from rlsolver.methods.ECO_S2V.src.envs.spinsystem_torch import SpinSystemFactory
elif ALG == Alg.PECO:
    from rlsolver.methods.ECO_S2V.src.envs.spinsystem_PECO import SpinSystemFactory


def make(id2, *args, **kwargs):
    if id2 == "SpinSystem":
        env = SpinSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env
