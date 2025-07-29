import sys
import os
import time
import statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
from rlsolver.methods.bls.bls import BLSMaxCut
import csv
from pathlib import Path

data_dir  = "gset"
file_name = "gset_14.txt"
runs      = 20
workers   = 4


script_dir    = Path(__file__).resolve().parent    # .../rlsolver/methods/bls
rlsolver_root = script_dir.parents[1]               # .../rlsolver
sys.path.insert(0, str(rlsolver_root))


instance_path = (rlsolver_root / "data" / data_dir / file_name).resolve()
if not instance_path.exists():
    sys.exit(f"ERROR: instance file not found: {instance_path}")
print(f"Loading instance from {instance_path}")

def run_trial(run_id):
    G = nx.read_edgelist(
        str(instance_path),
        nodetype=int,
        data=(("weight", int),)
    )
    n = G.number_of_nodes()
    params = {
        "L0_ratio":      0.01,
        "T":              1000,
        "phi_min":         3,
        "phi_max_ratio":  0.1,
        "P0":             0.8,
        "Q":              0.5,
        "max_iters": 10_000 * n,
    }
    solver = BLSMaxCut(G, params)
    t0 = time.time()
    _, best_val = solver.run(target=3064, time_limit=120)
    return run_id, best_val, time.time() - t0

if __name__ == "__main__":
    workers = min(workers, runs)
    print(f"Running {runs} trials in parallel with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(run_trial, range(1, runs+1)))

    bests = []
    for run_id, best_val, elapsed in results:
        print(f"Run {run_id:2d}: best = {best_val}, time = {elapsed:.2f}s")
        bests.append(best_val)

    best_overall = max(bests)
    avg = statistics.mean(bests)
    std = statistics.pstdev(bests)
    print("\n=== Summary ===")
    print(" Best =", best_overall)
    print(" Avg  =", f"{avg:.2f}")
    print(" Std  =", f"{std:.2f}")


    out_dir = Path(rlsolver_root) / "result"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"bls_{file_name.replace('.txt', '')}.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["run_id", "best_val", "elapsed_s"])

        for run_id, best_val, elapsed in results:
            writer.writerow([run_id, best_val, f"{elapsed:.2f}"])

        writer.writerow([])
        writer.writerow(["Best", best_overall])
        writer.writerow(["Avg", f"{avg:.2f}"])
        writer.writerow(["Std", f"{std:.2f}"])

    print(f"All results saved to {out_file}")