import sys, time, statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx


project_root = Path(__file__).resolve().parent  # .../RLSolver/rlsolver

sys.path.insert(0, str(project_root))


from methods.bls_maxcut.bls import BLSMaxCut



def run_trial(run_id):
    inst = project_root / "data" / "gset" / "gset_14.txt"
    G = nx.read_edgelist(str(inst), nodetype=int, data=(('weight', int),))
    n = G.number_of_nodes()
    params = {
        "L0_ratio": 0.01,
        "T": 1000,
        "phi_min": 3,
        "phi_max_ratio": 0.1,
        "P0": 0.8,
        "Q": 0.5,

        "max_iters": 10_000 * n,
    }
    solver = BLSMaxCut(G, params)
    t0 = time.time()
    _, best_val = solver.run(target=3064, time_limit=120)
    elapsed = time.time() - t0
    return run_id, best_val, elapsed


if __name__ == "__main__":
    runs = 20
    workers = min(4, runs)
    print(f"Running {runs} trials in parallel with {workers} workers...")


    with ProcessPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(run_trial, range(1, runs + 1)))


    bests = []
    for run_id, best_val, elapsed in results:
        print(f"Run {run_id:2d}: best = {best_val}, time = {elapsed:.2f}s")
        bests.append(best_val)

    print("\n=== Summary for gset_14 (parallel runs) ===")
    print(" Best =", max(bests))
    print(" Avg  =", f"{statistics.mean(bests):.2f}")
    print(" Std  =", f"{statistics.pstdev(bests):.2f}")
