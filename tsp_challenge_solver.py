import os
import math
import time
import numpy as np

# ==============================================================================
# TSPLIB PARSER
# ==============================================================================
def parse_tsplib(filepath):
    """
    Parses a TSPLIB format file (.tsp) and extracts node coordinates.
    Supports EUC_2D and general 2D coordinates.
    """
    coords = []
    node_coord_section = False
    name = os.path.basename(filepath)
    edge_weight_type = "EUC_2D"

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                parts = line.split(":")
                if len(parts) > 1:
                    name = parts[1].strip()
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                parts = line.split(":")
                if len(parts) > 1:
                    edge_weight_type = parts[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                node_coord_section = True
                continue
            elif line.startswith("EOF"):
                break
            elif node_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        pass
    coords = np.array(coords)
    return name, coords, edge_weight_type

def compute_dist_matrix(coords):
    """
    Computes standard TSPLIB EUC_2D rounded or float Euclidean distances.
    For standard benchmarks, nint(sqrt(dx^2 + dy^2)) or float can be used.
    We compute standard Euclidean distances.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Standard Euclidean distance in km/units:
    return np.hypot(diff[:, :, 0], diff[:, :, 1])

def tour_length(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

# ==============================================================================
# ALGORITHM 1: NEAREST NEIGHBOR (With Multi-Start for Best Initial Tour)
# ==============================================================================
def nearest_neighbor(dist_matrix, start_node=0):
    n = len(dist_matrix)
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    current = start_node

    for _ in range(n - 1):
        best_d = float('inf')
        best_next = -1
        for nxt in range(n):
            if not visited[nxt] and dist_matrix[current][nxt] < best_d:
                best_d = dist_matrix[current][nxt]
                best_next = nxt
        visited[best_next] = True
        tour.append(best_next)
        current = best_next
    return tour, tour_length(tour, dist_matrix)

def multi_start_nearest_neighbor(dist_matrix, num_starts=10):
    n = len(dist_matrix)
    best_tour = None
    best_dist = float('inf')
    
    # Try start_node=0 (mandated home) and a few random nodes
    starts = [0]
    if n > 1:
        extra_starts = np.linspace(1, n - 1, min(num_starts - 1, n - 1), dtype=int)
        starts.extend(extra_starts)
        
    for s in starts:
        tour, d = nearest_neighbor(dist_matrix, start_node=s)
        # Rotate tour so it starts at 0 (Home)
        idx0 = tour.index(0)
        tour = tour[idx0:] + tour[:idx0]
        if d < best_dist:
            best_dist = d
            best_tour = tour
    return best_tour, best_dist

# ==============================================================================
# ALGORITHM 2: FAST 2-OPT (EDGE INVERSION)
# ==============================================================================
def fast_2opt(tour, dist_matrix, max_passes=50):
    n = len(tour)
    best_tour = list(tour)
    best_dist = tour_length(best_tour, dist_matrix)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                
                # Check if reversing [i:j+1] reduces distance
                delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])
                if delta < -1e-6:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_dist += delta
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_dist

# ==============================================================================
# ALGORITHM 3: RELOCATE / 1-SHIFT (ESCAPE 1 FROM SLIDES)
# ==============================================================================
def fast_relocate(tour, dist_matrix, max_passes=20):
    n = len(tour)
    best_tour = list(tour)
    best_dist = tour_length(best_tour, dist_matrix)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n):
            node = best_tour[i]
            prev_node = best_tour[i - 1]
            next_node = best_tour[(i + 1) % n]
            
            # Removal delta
            cost_remove = dist_matrix[prev_node][next_node] - (dist_matrix[prev_node][node] + dist_matrix[node][next_node])
            
            for j in range(1, n):
                if j == i or j == i - 1:
                    continue
                u = best_tour[j]
                v = best_tour[(j + 1) % n]
                cost_insert = (dist_matrix[u][node] + dist_matrix[node][v]) - dist_matrix[u][v]
                delta = cost_remove + cost_insert
                
                if delta < -1e-6:
                    candidate = [x for k, x in enumerate(best_tour) if k != i]
                    insert_pos = j if j < i else j
                    candidate.insert(insert_pos, node)
                    best_tour = candidate
                    best_dist += delta
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_dist

# ==============================================================================
# ALGORITHM 4: VARIABLE NEIGHBORHOOD SEARCH / ESCAPE (HYBRID SOLVER)
# ==============================================================================
def solve_tsp_challenge(dist_matrix, time_limit_sec=5.0):
    """
    Applies the full lecture pipeline:
    1. Nearest Neighbor construction (Slide 2 & 3)
    2. 2-Opt untangling (Slide 3)
    3. Relocate Neighborhood (Escape 1 - Slide 7)
    4. Perturbation / Tabu kicks to escape local minima (Slide 8)
    """
    start_time = time.time()
    n = len(dist_matrix)
    
    # 1. Construction
    tour, d_init = nearest_neighbor(dist_matrix, start_node=0)
    
    # 2. Local search: 2-Opt
    tour, d_opt = fast_2opt(tour, dist_matrix)
    
    # 3. Escape 1: Relocate
    tour, d_rel = fast_relocate(tour, dist_matrix)
    
    best_tour = list(tour)
    best_dist = tour_length(best_tour, dist_matrix)
    
    # 4. Multi-restart / Double-bridge kick to escape local minima (Iterated Local Search / Tabu)
    iterations = 0
    while time.time() - start_time < time_limit_sec and iterations < 15:
        iterations += 1
        # Perturbation: double bridge (4-edge swap) or small segment shuffle
        perturbed = list(best_tour)
        if n > 12:
            cut = np.sort(np.random.choice(range(1, n), 3, replace=False))
            p1, p2, p3 = cut[0], cut[1], cut[2]
            perturbed = perturbed[:p1] + perturbed[p2:p3] + perturbed[p1:p2] + perturbed[p3:]
        
        # Local search on perturbed tour
        cand_tour, _ = fast_2opt(perturbed, dist_matrix, max_passes=20)
        cand_tour, _ = fast_relocate(cand_tour, dist_matrix, max_passes=10)
        cand_dist = tour_length(cand_tour, dist_matrix)
        
        if cand_dist < best_dist - 1e-6:
            best_dist = cand_dist
            best_tour = cand_tour

    # Ensure tour starts at 0
    idx0 = best_tour.index(0)
    best_tour = best_tour[idx0:] + best_tour[:idx0]
    
    return {
        "nn_dist": d_init,
        "best_tour": best_tour,
        "best_dist": best_dist,
        "improvement_pct": ((d_init - best_dist) / d_init) * 100
    }

