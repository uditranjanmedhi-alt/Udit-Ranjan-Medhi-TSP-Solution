"""
Traveling Salesman Problem — Optimization Simulator & Visualizer
================================================================
Streamlit web application for the TSP Challenge (pr152 and other TSPLIB instances).

Algorithms implemented from scratch (no optimization libraries):
  • Construction: Nearest Neighbor (with multi-start)
  • Improvement: 2-Opt, Or-Opt (relocate/segment move), Tabu Search
"""

import streamlit as st
import numpy as np
import math
import time
import random
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────────────
# 1.  TSP FILE PARSER
# ─────────────────────────────────────────────────────

def parse_tsp_file(filepath):
    """Parse a TSPLIB .tsp file and return city coords as dict {id: (x,y)}."""
    coords = {}
    name = ""
    reading_coords = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(":")[-1].strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line == "EOF":
                break
            if reading_coords:
                parts = line.split()
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                coords[node_id] = (x, y)
    return name, coords


def euclidean_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def build_distance_matrix(coords):
    """Build a full NxN distance matrix (numpy array) and index mappings."""
    ids = sorted(coords.keys())
    n = len(ids)
    id_to_idx = {city_id: i for i, city_id in enumerate(ids)}
    idx_to_id = {i: city_id for i, city_id in enumerate(ids)}

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(coords[ids[i]], coords[ids[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist, ids, id_to_idx, idx_to_id


def tour_length(tour_idx, dist):
    """Calculate total tour length (closed loop). tour_idx uses 0-based indices."""
    total = 0.0
    n = len(tour_idx)
    for i in range(n):
        total += dist[tour_idx[i]][tour_idx[(i + 1) % n]]
    return total


def ids_to_idx(tour_ids, id_to_idx):
    return [id_to_idx[c] for c in tour_ids]


def idx_to_ids(tour_idx, idx_to_id):
    return [idx_to_id[i] for i in tour_idx]


# ─────────────────────────────────────────────────────
# 2.  CONSTRUCTION: NEAREST NEIGHBOR
# ─────────────────────────────────────────────────────

def nearest_neighbor(dist, n, start=0):
    """Build a tour using the nearest-neighbor heuristic. Uses 0-based indices."""
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
    return tour


# ─────────────────────────────────────────────────────
# 3.  IMPROVEMENT: 2-OPT (first-improvement, keeps looping)
# ─────────────────────────────────────────────────────

def two_opt_improve(tour, dist, max_iter=500):
    """
    Run 2-opt with first-improvement pivoting (faster than best-improvement).
    Each 'iteration' is one full pass; we keep going until a pass has no improvement.
    Returns (tour, cost, history).
    """
    n = len(tour)
    tour = list(tour)
    cost = tour_length(tour, dist)
    history = [(list(tour), cost, "Start of 2-Opt")]

    for iteration in range(1, max_iter + 1):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Delta calculation
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-10:
                    # Reverse the segment between i+1 and j
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    cost += delta
                    improved = True
                    break  # first improvement — restart inner loop
            if improved:
                break

        history.append((list(tour), cost, f"2-Opt pass {iteration}"))
        if not improved:
            break
    return tour, cost, history


# ─────────────────────────────────────────────────────
# 4.  IMPROVEMENT: OR-OPT (RELOCATE segments of 1, 2, 3)
# ─────────────────────────────────────────────────────

def or_opt_improve(tour, dist, max_iter=500):
    """
    Or-opt: try relocating segments of length 1, 2, 3.
    Uses first-improvement pivoting.
    """
    n = len(tour)
    tour = list(tour)
    cost = tour_length(tour, dist)
    history = [(list(tour), cost, "Start of Or-Opt")]

    for iteration in range(1, max_iter + 1):
        improved = False
        for seg_len in [1, 2, 3]:
            if improved:
                break
            for i in range(n):
                if improved:
                    break
                # Segment: tour[i], tour[i+1], ..., tour[i+seg_len-1] (mod n)
                seg_indices = [(i + k) % n for k in range(seg_len)]
                prev_seg = tour[(i - 1) % n]
                after_seg = tour[(i + seg_len) % n]
                seg_cities = [tour[si] for si in seg_indices]

                # Cost of removing segment
                remove_cost = dist[prev_seg][seg_cities[0]]
                for k in range(seg_len - 1):
                    remove_cost += dist[seg_cities[k]][seg_cities[k + 1]]
                remove_cost += dist[seg_cities[-1]][after_seg]
                reconnect_cost = dist[prev_seg][after_seg]
                saving = remove_cost - reconnect_cost

                # Build reduced tour (without the segment)
                if i + seg_len <= n:
                    reduced = tour[:i] + tour[i + seg_len :]
                else:
                    wrap = (i + seg_len) % n
                    reduced = tour[wrap:i]

                # Try inserting segment at every position in reduced tour
                nr = len(reduced)
                for j in range(nr):
                    a = reduced[j]
                    b = reduced[(j + 1) % nr]
                    insert_cost = (
                        dist[a][seg_cities[0]]
                        + sum(
                            dist[seg_cities[k]][seg_cities[k + 1]]
                            for k in range(seg_len - 1)
                        )
                        + dist[seg_cities[-1]][b]
                        - dist[a][b]
                    )
                    delta = insert_cost - saving
                    if delta < -1e-10:
                        # Apply: insert segment after position j in reduced
                        new_tour = (
                            reduced[: j + 1] + seg_cities + reduced[j + 1 :]
                        )
                        tour = new_tour
                        cost += delta
                        improved = True
                        break

        history.append((list(tour), cost, f"Or-Opt pass {iteration}"))
        if not improved:
            break
    return tour, cost, history


# ─────────────────────────────────────────────────────
# 5.  IMPROVEMENT: TABU SEARCH (2-opt neighbourhood)
# ─────────────────────────────────────────────────────

def tabu_search(tour, dist, max_iter=300, tabu_tenure=15, callback=None):
    """Tabu search with 2-opt neighbourhood and aspiration criterion."""
    n = len(tour)
    current = list(tour)
    current_cost = tour_length(current, dist)
    best_tour = list(current)
    best_cost = current_cost
    tabu_list = {}
    history = [(list(current), current_cost, "Start of Tabu")]

    no_improve_count = 0

    for iteration in range(1, max_iter + 1):
        best_move = None
        best_delta = float("inf")

        # Evaluate neighbourhood
        candidates = []
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                candidates.append((i, j))

        # Sample if too many candidates
        if len(candidates) > 5000:
            candidates = random.sample(candidates, 5000)

        for i, j in candidates:
            a, b = current[i], current[i + 1]
            c, d = current[j], current[(j + 1) % n]
            delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

            edge_key = (min(a, c), max(a, c))
            is_tabu = edge_key in tabu_list and tabu_list[edge_key] > iteration

            # Aspiration: accept tabu move only if it beats global best
            if is_tabu and current_cost + delta >= best_cost - 1e-10:
                continue

            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)

        if best_move is None:
            break

        i, j = best_move
        # Record tabu
        a_old, b_old = current[i], current[i + 1]
        c_old = current[j]

        # Apply move
        current[i + 1 : j + 1] = reversed(current[i + 1 : j + 1])
        current_cost += best_delta

        # Add edges to tabu list
        tabu_list[(min(a_old, b_old), max(a_old, b_old))] = iteration + tabu_tenure
        tabu_list[(min(c_old, current[(j + 1) % n]), max(c_old, current[(j + 1) % n]))] = iteration + tabu_tenure

        if current_cost < best_cost - 1e-10:
            best_tour = list(current)
            best_cost = current_cost
            no_improve_count = 0
        else:
            no_improve_count += 1

        label = f"Tabu iter {iteration} (cost={current_cost:.1f})"
        if current_cost <= best_cost + 1e-10:
            label += " ★"
        history.append((list(current), current_cost, label))

        # Early stop if no improvement for many iterations
        if no_improve_count > max_iter // 2:
            break

    # Return best found tour
    return best_tour, best_cost, history


# ─────────────────────────────────────────────────────
# 6.  COMBINED OPTIMISATION PIPELINE
# ─────────────────────────────────────────────────────

def run_pipeline(dist, n, start_idx, methods, params):
    """Run construction + selected improvement methods in sequence."""
    all_history = []

    # Step 1: Construction — NN
    tour = nearest_neighbor(dist, n, start=start_idx)
    cost = tour_length(tour, dist)
    all_history.append(("Nearest Neighbor", [(list(tour), cost, "NN Construction")]))

    # Step 2: Improvements
    current = list(tour)
    for method in methods:
        if method == "2-Opt":
            current, cost, hist = two_opt_improve(
                current, dist, max_iter=params.get("2opt_iters", 500)
            )
            all_history.append(("2-Opt", hist))
        elif method == "Or-Opt (Relocate)":
            current, cost, hist = or_opt_improve(
                current, dist, max_iter=params.get("oropt_iters", 500)
            )
            all_history.append(("Or-Opt", hist))
        elif method == "Tabu Search":
            current, cost, hist = tabu_search(
                current,
                dist,
                max_iter=params.get("tabu_iters", 300),
                tabu_tenure=params.get("tabu_tenure", 15),
            )
            all_history.append(("Tabu Search", hist))
        elif method == "2-Opt + Or-Opt (Repeat)":
            # Alternate 2-opt and or-opt until stable
            for rep in range(params.get("repeat_rounds", 5)):
                current, cost, hist2 = two_opt_improve(current, dist, max_iter=200)
                all_history.append((f"2-Opt (round {rep+1})", hist2))
                prev_cost = cost
                current, cost, hist3 = or_opt_improve(current, dist, max_iter=200)
                all_history.append((f"Or-Opt (round {rep+1})", hist3))
                if abs(prev_cost - cost) < 1e-6:
                    break

    return current, cost, all_history


# ─────────────────────────────────────────────────────
# 7.  VISUALIZATION (Plotly)
# ─────────────────────────────────────────────────────

def make_tour_figure(tour_idx, coords, idx_to_id, title="TSP Tour", cost=None):
    """Create a Plotly figure showing the tour on a 2D map."""
    tour_ids = [idx_to_id[i] for i in tour_idx]
    xs = [coords[c][0] for c in tour_ids] + [coords[tour_ids[0]][0]]
    ys = [coords[c][1] for c in tour_ids] + [coords[tour_ids[0]][1]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Tour",
            hoverinfo="skip",
        )
    )

    city_xs = [coords[c][0] for c in tour_ids]
    city_ys = [coords[c][1] for c in tour_ids]
    city_labels = [str(c) for c in tour_ids]
    fig.add_trace(
        go.Scatter(
            x=city_xs, y=city_ys,
            mode="markers",
            marker=dict(size=6, color="#ef4444", line=dict(width=1, color="white")),
            text=city_labels,
            hoverinfo="text",
            name="Cities",
        )
    )

    display_title = title
    if cost is not None:
        display_title += f"  |  Distance: {cost:,.1f}"

    fig.update_layout(
        title=dict(text=display_title, font=dict(size=16)),
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y"),
        showlegend=True,
        height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
    )
    return fig


def make_convergence_figure(all_history):
    """Create a convergence plot showing cost vs. iteration across all phases."""
    fig = go.Figure()
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
              "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#a855f7"]
    offset = 0
    for idx, (phase_name, history) in enumerate(all_history):
        iters = list(range(offset, offset + len(history)))
        costs = [h[1] for h in history]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=iters, y=costs,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(color=color, width=2),
                name=phase_name,
                hovertemplate=f"{phase_name}<br>Step %{{x}}<br>Cost: %{{y:,.1f}}<extra></extra>",
            )
        )
        offset += len(history)

    fig.update_layout(
        title=dict(text="Optimization Convergence", font=dict(size=16)),
        xaxis_title="Cumulative Step",
        yaxis_title="Tour Distance",
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def make_best_cost_figure(all_history):
    """Show the running best cost across all steps."""
    fig = go.Figure()
    steps = []
    best_costs = []
    best_so_far = float("inf")
    step = 0
    for phase_name, history in all_history:
        for _, c, _ in history:
            if c < best_so_far:
                best_so_far = c
            steps.append(step)
            best_costs.append(best_so_far)
            step += 1

    fig.add_trace(
        go.Scatter(
            x=steps, y=best_costs,
            mode="lines",
            line=dict(color="#10b981", width=3),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.1)",
            name="Best Cost",
        )
    )
    fig.update_layout(
        title="Running Best Distance",
        xaxis_title="Step",
        yaxis_title="Best Tour Distance",
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────
# 8.  STREAMLIT APP
# ─────────────────────────────────────────────────────

def get_tsp_files():
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSP_Challenge"),
    ]
    files = {}
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".tsp"):
                    files[f] = os.path.join(d, f)
    return files


def main():
    st.set_page_config(page_title="TSP Optimizer", page_icon="🗺️", layout="wide")

    st.title("🗺️ Traveling Salesman Problem — Optimization Simulator")
    st.markdown(
        "Build a tour with **Nearest Neighbor**, then improve it with "
        "**2-Opt**, **Or-Opt (Relocate)**, and **Tabu Search**. "
        "All algorithms implemented from scratch — no optimization libraries."
    )

    # ── Sidebar ────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    tsp_files = get_tsp_files()
    if not tsp_files:
        st.error("No .tsp files found. Place them next to this script.")
        return

    file_names = sorted(tsp_files.keys())
    default_idx = file_names.index("pr152.tsp") if "pr152.tsp" in file_names else 0
    selected_file = st.sidebar.selectbox("📂 TSP Instance", file_names, index=default_idx)

    name, coords = parse_tsp_file(tsp_files[selected_file])
    dist, ids, id_to_idx, idx_to_id_map = build_distance_matrix(coords)
    n = len(ids)
    st.sidebar.markdown(f"**Instance:** {name}  |  **Cities:** {n}")

    start_city = st.sidebar.selectbox("🏁 Starting City (NN)", ids, index=0)
    start_idx = id_to_idx[start_city]

    st.sidebar.subheader("🔧 Improvement Methods")
    methods = st.sidebar.multiselect(
        "Select methods (applied in order):",
        ["2-Opt", "Or-Opt (Relocate)", "Tabu Search", "2-Opt + Or-Opt (Repeat)"],
        default=["2-Opt", "Or-Opt (Relocate)", "Tabu Search"],
    )

    params = {}
    with st.sidebar.expander("🎛️ Algorithm Parameters", expanded=False):
        if "2-Opt" in methods or "2-Opt + Or-Opt (Repeat)" in methods:
            params["2opt_iters"] = st.slider("2-Opt max iterations", 10, 1000, 500)
        if "Or-Opt (Relocate)" in methods or "2-Opt + Or-Opt (Repeat)" in methods:
            params["oropt_iters"] = st.slider("Or-Opt max iterations", 10, 1000, 500)
        if "Tabu Search" in methods:
            params["tabu_iters"] = st.slider("Tabu Search max iterations", 50, 2000, 500)
            params["tabu_tenure"] = st.slider("Tabu tenure", 5, 50, 15)
        if "2-Opt + Or-Opt (Repeat)" in methods:
            params["repeat_rounds"] = st.slider("Repeat rounds", 1, 20, 5)

    random_seed = st.sidebar.number_input("🎲 Random Seed", value=42, step=1)

    run_clicked = st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True)

    # ── Main Area ──────────────────────────────────
    if run_clicked:
        random.seed(random_seed)
        np.random.seed(random_seed)

        with st.spinner("Running optimization pipeline… this may take a moment for large instances."):
            t0 = time.time()
            final_tour, final_cost, all_history = run_pipeline(
                dist, n, start_idx, methods, params
            )
            elapsed = time.time() - t0

        nn_cost = all_history[0][1][0][1]
        improvement_pct = (1 - final_cost / nn_cost) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NN Tour Distance", f"{nn_cost:,.1f}")
        col2.metric("Optimized Distance", f"{final_cost:,.1f}")
        col3.metric("Improvement", f"{improvement_pct:.2f}%")
        col4.metric("Time", f"{elapsed:.2f}s")

        st.divider()

        tab_tour, tab_conv, tab_steps, tab_compare, tab_data = st.tabs(
            ["🗺️ Final Tour", "📈 Convergence", "🎬 Step-by-Step", "🔄 Before / After", "📊 Data"]
        )

        with tab_tour:
            fig = make_tour_figure(
                final_tour, coords, idx_to_id_map,
                title=f"Optimized Tour — {name}", cost=final_cost
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_conv:
            c1, c2 = st.columns([3, 2])
            with c1:
                fig_conv = make_convergence_figure(all_history)
                st.plotly_chart(fig_conv, use_container_width=True)
            with c2:
                fig_best = make_best_cost_figure(all_history)
                st.plotly_chart(fig_best, use_container_width=True)

            # Phase summary
            rows = []
            for phase_name, history in all_history:
                start_c = history[0][1]
                end_c = history[-1][1]
                rows.append({
                    "Phase": phase_name,
                    "Start Cost": f"{start_c:,.1f}",
                    "End Cost": f"{end_c:,.1f}",
                    "Improvement": f"{start_c - end_c:,.1f}",
                    "Steps": len(history),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with tab_steps:
            st.markdown("### Step through the optimization process")
            flat = []
            for phase_name, history in all_history:
                for tour_snap, cost_val, label in history:
                    flat.append((phase_name, tour_snap, cost_val, label))

            step_num = st.slider("Optimization Step", 0, len(flat) - 1, len(flat) - 1)
            phase, snap_tour, snap_cost, snap_label = flat[step_num]
            st.markdown(
                f"**Phase:** {phase}  |  **{snap_label}**  |  "
                f"**Distance:** {snap_cost:,.1f}"
            )

            fig_step = make_tour_figure(
                snap_tour, coords, idx_to_id_map,
                title=f"Step {step_num}: {snap_label}", cost=snap_cost
            )
            st.plotly_chart(fig_step, use_container_width=True)

            # Auto-play
            if st.button("▶️ Auto-play animation"):
                placeholder = st.empty()
                prog = st.progress(0)
                for s in range(len(flat)):
                    ph, st_tour, st_cost, st_label = flat[s]
                    fig_anim = make_tour_figure(
                        st_tour, coords, idx_to_id_map,
                        title=f"Step {s}: {st_label}", cost=st_cost
                    )
                    placeholder.plotly_chart(fig_anim, use_container_width=True)
                    prog.progress((s + 1) / len(flat))
                    time.sleep(0.3)

        with tab_compare:
            col_a, col_b = st.columns(2)
            nn_tour = all_history[0][1][0][0]
            with col_a:
                st.markdown("### Before (Nearest Neighbor)")
                fig_before = make_tour_figure(
                    nn_tour, coords, idx_to_id_map,
                    title="NN Construction", cost=nn_cost
                )
                st.plotly_chart(fig_before, use_container_width=True)
            with col_b:
                st.markdown("### After (Optimized)")
                fig_after = make_tour_figure(
                    final_tour, coords, idx_to_id_map,
                    title="Optimized", cost=final_cost
                )
                st.plotly_chart(fig_after, use_container_width=True)

        with tab_data:
            st.markdown("### Tour Order (City IDs)")
            tour_ids = [idx_to_id_map[i] for i in final_tour]
            st.code(" → ".join(map(str, tour_ids)) + f" → {tour_ids[0]}", language=None)
            st.markdown(f"**Total distance:** {final_cost:,.1f}")
            st.markdown(f"**Number of cities:** {len(final_tour)}")

            # City coordinates table
            import pandas as pd
            df = pd.DataFrame([
                {"City": idx_to_id_map[i], "X": coords[idx_to_id_map[i]][0], "Y": coords[idx_to_id_map[i]][1]}
                for i in final_tour
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Store results
        st.session_state["last_result"] = {
            "final_tour": final_tour,
            "final_cost": final_cost,
            "all_history": all_history,
            "nn_cost": nn_cost,
            "name": name,
            "coords": coords,
            "idx_to_id_map": idx_to_id_map,
        }

    elif "last_result" in st.session_state:
        r = st.session_state["last_result"]
        nn_cost = r["nn_cost"]
        final_cost = r["final_cost"]
        improvement_pct = (1 - final_cost / nn_cost) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("NN Tour Distance", f"{nn_cost:,.1f}")
        col2.metric("Optimized Distance", f"{final_cost:,.1f}")
        col3.metric("Improvement", f"{improvement_pct:.2f}%")
        st.info("Adjust settings and click **Run Optimization** to re-run.")
    else:
        city_xs = [c[0] for c in coords.values()]
        city_ys = [c[1] for c in coords.values()]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=city_xs, y=city_ys,
                mode="markers",
                marker=dict(size=7, color="#3b82f6"),
                text=[str(k) for k in coords.keys()],
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=f"Cities — {name} ({len(coords)} cities)",
            xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y"),
            height=550,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("👈 Configure the algorithm in the sidebar and click **Run Optimization**.")


if __name__ == "__main__":
    main()
