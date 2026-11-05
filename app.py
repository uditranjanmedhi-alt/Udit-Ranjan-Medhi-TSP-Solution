import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsp_engine import (
    solve_nearest_neighbor,
    solve_2opt,
    solve_relocate,
    solve_tabu_search,
    compute_total_distance
)

st.set_page_config(page_title="Traveling Salesman Problem App", page_icon="📍", layout="wide")

# ==============================================================================
# BANGALORE RAKHI LECTURE DATASET (Slides 1 to 8)
# Nodes: Home (0), A (1), B (2), C (3), D (4), E (5), F (6), G (7)
# ==============================================================================
BANGALORE_NODES = {
    "Home": (10, 80),
    "A": (20, 68),
    "B": (35, 78),
    "C": (85, 75),
    "D": (70, 40),
    "E": (90, 30),
    "F": (55, 25),
    "G": (20, 35)
}
BANGALORE_NODE_NAMES = list(BANGALORE_NODES.keys())

# Exact edge weights from Slide 2 / 5 graph (fill large number 999 for non-edges, or shortest path)
def create_bangalore_dist_matrix():
    # Direct graph weights from slide 2:
    # Home-A: 1, Home-B: 2, Home-G: 5
    # A-B: 1, A-G: 2
    # B-C: 5, B-D: 3
    # C-D: 1, C-E: 3
    # D-E: 2, D-F: 2
    # E-F: 6
    # F-G: 2
    n = 8
    M = 999.0
    adj = np.full((n, n), M)
    np.fill_diagonal(adj, 0.0)
    
    edges = [
        (0, 1, 1), (0, 2, 2), (0, 7, 5),
        (1, 2, 1), (1, 7, 2),
        (2, 3, 5), (2, 4, 3),
        (3, 4, 1), (3, 5, 3),
        (4, 5, 2), (4, 6, 2),
        (5, 6, 6),
        (6, 7, 2)
    ]
    for u, v, w in edges:
        adj[u][v] = w
        adj[v][u] = w
        
    # Floyd-Warshall to get full all-pairs shortest paths
    dist = adj.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# ==============================================================================
# UI HEADER
# ==============================================================================
st.title("📍 Traveling Salesperson Problem (TSP) Solver & Visualizer")
st.markdown("""
*Built for the **Traveling Salesman Challenge** (DDDM).*  
Implements **pure algorithmic heuristics from scratch** without external optimization solvers:
**1. Nearest Neighbor (Construction)** $\\rightarrow$ **2. 2-Opt / Relocate / Tabu Search (Improvement)**.
""")

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("🛠️ Problem Configuration")

data_source = st.sidebar.radio(
    "Choose Dataset Source:",
    ["Bangalore Rakhi Example (Lecture Slide)", "Upload Assignment File (tinyurl.com/tsp-data)", "Generate Random Cities"]
)

coords = None
dist_matrix = None
node_names = None

if data_source == "Bangalore Rakhi Example (Lecture Slide)":
    node_names = BANGALORE_NODE_NAMES
    coords = np.array(list(BANGALORE_NODES.values()))
    dist_matrix = create_bangalore_dist_matrix()
    st.sidebar.success("Loaded Bangalore Rakhi 8-Node network (Slide 2-8)")

elif data_source == "Upload Assignment File (tinyurl.com/tsp-data)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV, TSV, or TXT", type=["csv", "tsv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.sidebar.write("Preview of uploaded dataset:", df.head())
            # Check if columns x, y or lat, lon exist
            col_x = [c for c in df.columns if str(c).lower() in ['x', 'lon', 'longitude', 'xcord'] ]
            col_y = [c for c in df.columns if str(c).lower() in ['y', 'lat', 'latitude', 'ycord'] ]
            
            if col_x and col_y:
                coords = df[[col_x[0], col_y[0]]].values
                n = len(coords)
                node_names = [f"City {i+1}" for i in range(n)]
                # Euclidean distance matrix
                diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
                st.sidebar.success(f"Parsed {n} city coordinates successfully!")
            else:
                # Assume full distance matrix
                dist_matrix = df.values
                n = len(dist_matrix)
                node_names = [f"City {i+1}" for i in range(n)]
                coords = np.random.uniform(10, 90, size=(n, 2))  # Dummy coords for display
                st.sidebar.success(f"Parsed {n}x{n} Distance Matrix successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")
    else:
        st.sidebar.info("Upload your downloaded assignment file from https://tinyurl.com/tsp-data")

elif data_source == "Generate Random Cities":
    num_cities = st.sidebar.slider("Number of Cities / Stops:", min_value=5, max_value=60, value=15, step=1)
    seed = st.sidebar.number_input("Random Seed:", value=42, step=1)
    np.random.seed(seed)
    coords = np.random.uniform(5, 95, size=(num_cities, 2))
    node_names = ["Home"] + [f"C{i}" for i in range(1, num_cities)]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])

# ==============================================================================
# ALGORITHM EXECUTION & DASHBOARD
# ==============================================================================
if dist_matrix is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Algorithm Parameters")
    tabu_tenure = st.sidebar.slider("Tabu Search Tenure:", min_value=2, max_value=20, value=5)
    tabu_iters = st.sidebar.slider("Tabu Search Max Iterations:", min_value=20, max_value=200, value=80)
    
    # 1. Construction: Nearest Neighbor
    nn_tour, nn_dist = solve_nearest_neighbor(dist_matrix, start_node=0)
    
    # 2. Improvement: 2-Opt
    opt_tour, opt_dist, opt_hist = solve_2opt(nn_tour, dist_matrix)
    
    # 3. Improvement: Relocate (Escape 1)
    rel_tour, rel_dist, rel_hist = solve_relocate(nn_tour, dist_matrix)
    
    # 4. Improvement: Tabu Search (Escape 2)
    tabu_tour, tabu_dist, tabu_hist = solve_tabu_search(nn_tour, dist_matrix, tabu_tenure=tabu_tenure, max_iterations=tabu_iters)
    
    # Identify Overall Best
    sol_dict = {
        "Nearest Neighbor (Start)": (nn_tour, nn_dist),
        "2-Opt (Edge Inversion)": (opt_tour, opt_dist),
        "Relocate (Node Reinsertion)": (rel_tour, rel_dist),
        "Tabu Search (Swap with Memory)": (tabu_tour, tabu_dist)
    }
    best_algo = min(sol_dict.keys(), key=lambda k: sol_dict[k][1])
    best_tour, best_dist = sol_dict[best_algo]
    
    # KPI Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nearest Neighbor (Baseline)", f"{nn_dist:.1f} km")
    col2.metric("2-Opt", f"{opt_dist:.1f} km", delta=f"{nn_dist - opt_dist:.1f} km saved" if nn_dist != opt_dist else "0 km")
    col3.metric("Relocate (Escape 1)", f"{rel_dist:.1f} km", delta=f"{nn_dist - rel_dist:.1f} km saved" if nn_dist != rel_dist else "0 km")
    col4.metric("Tabu Search (Escape 2)", f"{tabu_dist:.1f} km", delta=f"{nn_dist - tabu_dist:.1f} km saved" if nn_dist != tabu_dist else "0 km")
    
    st.success(f"🏆 **Hall of Fame Result:** Best Solution Found by **{best_algo}** = **{best_dist:.2f} km**")

    # ==============================================================================
    # ROUTE VISUALIZATION
    # ==============================================================================
    st.subheader("🗺️ Route Comparison & Visual Map")
    
    selected_view = st.selectbox(
        "Select Route to Visualize on Map:",
        ["Nearest Neighbor (Start)", "2-Opt (Edge Inversion)", "Relocate (Node Reinsertion)", "Tabu Search (Swap with Memory)", "Best Overall"]
    )
    
    active_tour = best_tour if selected_view == "Best Overall" else sol_dict[selected_view][0]
    active_dist = best_dist if selected_view == "Best Overall" else sol_dict[selected_view][1]
    
    # Format readable tour string
    tour_str = " → ".join([node_names[i] for i in active_tour]) + f" → {node_names[active_tour[0]]}"
    st.info(f"**Visiting Sequence ({active_dist:.2f} km):** `{tour_str}`")

    # Plot Route Map
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw path edges
    n_nodes = len(active_tour)
    for idx in range(n_nodes):
        u = active_tour[idx]
        v = active_tour[(idx + 1) % n_nodes]
        ax.annotate(
            "",
            xy=(coords[v][0], coords[v][1]),
            xytext=(coords[u][0], coords[u][1]),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2, mutation_scale=15, alpha=0.85)
        )
    
    # Draw Nodes
    ax.scatter(coords[:, 0], coords[:, 1], c="#3b82f6", s=180, edgecolors="black", zorder=4)
    # Highlight Home node
    ax.scatter(coords[0, 0], coords[0, 1], c="#ef4444", s=260, marker="*", edgecolors="black", zorder=5, label="Home")
    
    # Add Node Labels
    for i in range(len(coords)):
        ax.text(coords[i, 0], coords[i, 1] + 1.8, node_names[i], ha="center", fontsize=9, weight="bold")

    ax.set_title(f"{selected_view} Route Map (Total Length: {active_dist:.2f} km)", fontsize=13, weight="bold")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    st.pyplot(fig)

    # Detailed Table Breakdown
    with st.expander("📊 View Turn-by-Turn Leg Distances"):
        leg_data = []
        cum_dist = 0.0
        for idx in range(n_nodes):
            u = active_tour[idx]
            v = active_tour[(idx + 1) % n_nodes]
            leg_len = dist_matrix[u][v]
            cum_dist += leg_len
            leg_data.append({
                "Leg": f"{idx+1}",
                "From": node_names[u],
                "To": node_names[v],
                "Distance (km)": f"{leg_len:.2f}",
                "Cumulative (km)": f"{cum_dist:.2f}"
            })
        st.dataframe(pd.DataFrame(leg_data), use_container_width=True)

else:
    st.warning("Please upload a file or select a dataset from the sidebar to start.")

