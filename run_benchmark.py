import os
import glob
import time
import pandas as pd
from tsp_challenge_solver import parse_tsplib, compute_dist_matrix, solve_tsp_challenge

DATA_DIR = r"D:\IMI\TERM 5\DDDM VS\TSP_Data"
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tsp")))

results = []
print(f"Found {len(files)} TSP instances in {DATA_DIR}...\n")

for filepath in files:
    filename = os.path.basename(filepath)
    name, coords, edge_type = parse_tsplib(filepath)
    n = len(coords)
    
    t0 = time.time()
    dist_matrix = compute_dist_matrix(coords)
    # Solve using pure Python heuristics without external solvers
    sol = solve_tsp_challenge(dist_matrix, time_limit_sec=3.0)
    elapsed = time.time() - t0
    
    results.append({
        "Instance": filename,
        "Nodes (Stops)": n,
        "NN Starting (km)": round(sol["nn_dist"], 2),
        "Optimized (km)": round(sol["best_dist"], 2),
        "Improvement (%)": f"{sol['improvement_pct']:.1f}%",
        "Time (s)": round(elapsed, 2)
    })
    print(f"[{filename}] Nodes: {n:3d} | NN: {sol['nn_dist']:9.2f} km -> Optimized: {sol['best_dist']:9.2f} km | Saved: {sol['improvement_pct']:4.1f}% | Time: {elapsed:.2f}s")

df = pd.DataFrame(results)
print("\n" + "="*70)
print("            TRAVELING SALESMAN CHALLENGE - RESULTS TABLE")
print("="*70)
print(df.to_string(index=False))
df.to_csv(r"D:\IMI\TERM 5\DDDM VS\TSP_Results.csv", index=False)
print(f"\nSaved results to D:\IMI\TERM 5\DDDM VS\TSP_Results.csv")

