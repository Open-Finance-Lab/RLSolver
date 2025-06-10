from rlsolver.methods.eco_s2v.config import *

if ALG == Alg.eco or ALG == Alg.s2v:
    from rlsolver.methods.eco_s2v.src.envs.spinsystem import SpinSystemFactory
elif ALG == Alg.eco_torch:
    from rlsolver.methods.eco_s2v.src.envs.torch_spinsystem import SpinSystemFactory
elif ALG == Alg.eeco:
    from rlsolver.methods.eco_s2v.src.envs.eeco_spinsystem import SpinSystemFactory


def make(id2, *args, **kwargs):
    if id2 == "SpinSystem":
        env = SpinSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env
