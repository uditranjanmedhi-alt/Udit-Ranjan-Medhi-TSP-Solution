import numpy as np

def compute_total_distance(tour, dist_matrix):
    """
    Computes total round-trip distance of a tour.
    tour: list of node indices [0, 1, 2, ..., n-1]
    dist_matrix: 2D numpy array or list of lists
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total

# ==============================================================================
# 1. CONSTRUCTION: Nearest Neighbor Algorithm
# ==============================================================================
def solve_nearest_neighbor(dist_matrix, start_node=0):
    """
    Constructs an initial TSP tour using greedy Nearest Neighbor heuristic.
    Starts at start_node, repeatedly visits the nearest unvisited node,
    and returns to the start node.
    """
    n = len(dist_matrix)
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    
    current = start_node
    for _ in range(n - 1):
        best_dist = float('inf')
        nearest_neighbor = -1
        for nxt in range(n):
            if not visited[nxt] and dist_matrix[current][nxt] < best_dist:
                best_dist = dist_matrix[current][nxt]
                nearest_neighbor = nxt
        visited[nearest_neighbor] = True
        tour.append(nearest_neighbor)
        current = nearest_neighbor
        
    return tour, compute_total_distance(tour, dist_matrix)

# ==============================================================================
# 2. IMPROVEMENT 1: 2-Opt (Edge Swapping / Segment Inversion)
# ==============================================================================
def solve_2opt(tour, dist_matrix, max_iterations=500):
    """
    2-Opt heuristic: iteratively reverses sub-segments [i:j+1] if it shortens
    the total tour length, eliminating crossing paths.
    """
    best_tour = list(tour)
    best_dist = compute_total_distance(best_tour, dist_matrix)
    n = len(tour)
    
    improved = True
    iteration = 0
    history = [(list(best_tour), best_dist)]
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Change in distance if we invert subsegment best_tour[i:j+1]
                # Old edges: (i-1 -> i) and (j -> (j+1)%n)
                # New edges: (i-1 -> j) and (i -> (j+1)%n)
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                
                delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])
                if delta < -1e-6:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_dist += delta
                    improved = True
                    history.append((list(best_tour), best_dist))
                    break
            if improved:
                break
                
    return best_tour, best_dist, history

# ==============================================================================
# 3. IMPROVEMENT 2: Relocate / 1-Shift (Node Reinsertion - Slide Escape 1)
# ==============================================================================
def solve_relocate(tour, dist_matrix, max_iterations=500):
    """
    Relocate move: Takes one customer out of the route and inserts it
    at a different position if it improves the tour distance.
    """
    best_tour = list(tour)
    best_dist = compute_total_distance(best_tour, dist_matrix)
    n = len(tour)
    
    improved = True
    iteration = 0
    history = [(list(best_tour), best_dist)]
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, n):  # Keep home/start node fixed at position 0
            node = best_tour[i]
            # Try inserting node at every other position j
            for j in range(1, n):
                if i == j:
                    continue
                candidate_tour = [x for k, x in enumerate(best_tour) if k != i]
                candidate_tour.insert(j, node)
                candidate_dist = compute_total_distance(candidate_tour, dist_matrix)
                
                if candidate_dist < best_dist - 1e-6:
                    best_tour = candidate_tour
                    best_dist = candidate_dist
                    improved = True
                    history.append((list(best_tour), best_dist))
                    break
            if improved:
                break
                
    return best_tour, best_dist, history

# ==============================================================================
# 4. IMPROVEMENT 3: Tabu Search (Slide Escape 2)
# ==============================================================================
def solve_tabu_search(tour, dist_matrix, tabu_tenure=5, max_iterations=100):
    """
    Tabu Search with Pairwise Swap moves.
    Accepts the best move in the neighborhood even if it worsens the tour,
    using a tabu tenure to prevent cycling and climb out of local minima.
    """
    n = len(tour)
    current_tour = list(tour)
    current_dist = compute_total_distance(current_tour, dist_matrix)
    
    best_tour = list(current_tour)
    best_dist = current_dist
    
    # tabu_dict[(min(nodeA, nodeB), max(nodeA, nodeB))] = iteration_when_expired
    tabu_dict = {}
    history = [(list(best_tour), best_dist)]
    
    for iteration in range(1, max_iterations + 1):
        best_candidate = None
        best_candidate_dist = float('inf')
        best_move = None
        
        # Evaluate pairwise swap moves (excluding home at index 0)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                node_a, node_b = current_tour[i], current_tour[j]
                move = (min(node_a, node_b), max(node_a, node_b))
                
                cand_tour = list(current_tour)
                cand_tour[i], cand_tour[j] = cand_tour[j], cand_tour[i]
                cand_dist = compute_total_distance(cand_tour, dist_matrix)
                
                # Aspiration criterion: allow tabu move if it beats the global best
                is_tabu = (move in tabu_dict and tabu_dict[move] > iteration)
                if (not is_tabu) or (cand_dist < best_dist - 1e-6):
                    if cand_dist < best_candidate_dist:
                        best_candidate_dist = cand_dist
                        best_candidate = cand_tour
                        best_move = move
                        
        if best_candidate is None:
            break
            
        current_tour = best_candidate
        current_dist = best_candidate_dist
        tabu_dict[best_move] = iteration + tabu_tenure
        
        if current_dist < best_dist - 1e-6:
            best_dist = current_dist
            best_tour = list(current_tour)
            history.append((list(best_tour), best_dist))
            
    return best_tour, best_dist, history

