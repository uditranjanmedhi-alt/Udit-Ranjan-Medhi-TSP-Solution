const fs = require('fs');

const coords = require('D:\\IMI\\TERM 5\\DDDM VS\\pr152_coords.json');
const n = coords.length;

function dist(c1, c2) {
    const dx = c1[0] - c2[0];
    const dy = c1[1] - c2[1];
    return Math.hypot(dx, dy);
}

// Precompute distance matrix
const D = new Float64Array(n * n);
for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
        D[i * n + j] = dist(coords[i], coords[j]);
    }
}

function getD(i, j) {
    return D[i * n + j];
}

function tourLen(tour) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += getD(tour[i], tour[(i + 1) % n]);
    }
    return sum;
}

// Multi-start Nearest Neighbor
function solveNN(startNode) {
    const visited = new Uint8Array(n);
    const tour = [startNode];
    visited[startNode] = 1;
    let curr = startNode;

    for (let step = 1; step < n; step++) {
        let bestD = Infinity;
        let nxt = -1;
        for (let j = 0; j < n; j++) {
            if (!visited[j]) {
                const d = getD(curr, j);
                if (d < bestD) {
                    bestD = d;
                    nxt = j;
                }
            }
        }
        visited[nxt] = 1;
        tour.push(nxt);
        curr = nxt;
    }
    return tour;
}

// 2-Opt local search
function run2Opt(tour) {
    let bestTour = tour.slice();
    let improved = true;

    while (improved) {
        improved = false;
        for (let i = 1; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const a = bestTour[i - 1], b = bestTour[i];
                const c = bestTour[j], d = bestTour[(j + 1) % n];

                const delta = (getD(a, c) + getD(b, d)) - (getD(a, b) + getD(c, d));
                if (delta < -1e-5) {
                    let l = i, r = j;
                    while (l < r) {
                        const tmp = bestTour[l];
                        bestTour[l] = bestTour[r];
                        bestTour[r] = tmp;
                        l++; r--;
                    }
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
    return bestTour;
}

// 2.5-Opt / Node Reinsertion (Relocate Move - Escape 1)
function runRelocate(tour) {
    let bestTour = tour.slice();
    let improved = true;

    while (improved) {
        improved = false;
        for (let i = 1; i < n; i++) {
            const node = bestTour[i];
            const prev = bestTour[i - 1];
            const next = bestTour[(i + 1) % n];
            const remCost = getD(prev, next) - (getD(prev, node) + getD(node, next));

            for (let j = 1; j < n; j++) {
                if (j === i || j === i - 1) continue;
                const u = bestTour[j];
                const v = bestTour[(j + 1) % n];
                const insCost = (getD(u, node) + getD(node, v)) - getD(u, v);
                const delta = remCost + insCost;

                if (delta < -1e-5) {
                    const cand = bestTour.filter((_, k) => k !== i);
                    cand.splice(j, 0, node);
                    bestTour = cand;
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
    return bestTour;
}

// Double Bridge Kick (4-opt perturbation) to escape local minima without random destruction
function doubleBridge(tour) {
    const t = tour.slice();
    const cuts = [];
    const step = Math.floor(n / 4);
    cuts.push(1 + Math.floor(Math.random() * (step - 2)));
    cuts.push(step + Math.floor(Math.random() * (step - 2)));
    cuts.push(2 * step + Math.floor(Math.random() * (step - 2)));
    cuts.sort((a, b) => a - b);

    const p1 = cuts[0], p2 = cuts[1], p3 = cuts[2];
    const A = t.slice(0, p1);
    const B = t.slice(p1, p2);
    const C = t.slice(p2, p3);
    const D_part = t.slice(p3);

    return A.concat(D_part).concat(C).concat(B);
}

// Test multi-starts & ILS
console.log("Searching for solution below 75,000 km...");

let globalBestTour = null;
let globalBestLen = Infinity;

for (let s = 0; s < n; s += 2) {
    let t = solveNN(s);
    t = run2Opt(t);
    t = runRelocate(t);
    let len = tourLen(t);

    if (len < globalBestLen) {
        globalBestLen = len;
        globalBestTour = t;
        console.log(`[Start ${s}] New Best: ${len.toFixed(2)} km`);
    }
}

// Iterated Local Search (ILS) with Tabu/Double-Bridge kicks
let current = globalBestTour.slice();
for (let iter = 1; iter <= 200; iter++) {
    let perturbed = doubleBridge(current);
    perturbed = run2Opt(perturbed);
    perturbed = runRelocate(perturbed);
    let len = tourLen(perturbed);

    if (len < globalBestLen) {
        globalBestLen = len;
        globalBestTour = perturbed;
        current = perturbed;
        console.log(`[Iter ${iter}] KICK SUCCESS! New Global Best: ${len.toFixed(2)} km`);
    } else if (len < tourLen(current) * 1.01) {
        current = perturbed; // accept slightly worse to explore
    }
}

// Ensure tour starts at 0 (Home)
const idx0 = globalBestTour.indexOf(0);
globalBestTour = globalBestTour.slice(idx0).concat(globalBestTour.slice(0, idx0));

console.log(`\n========================================`);
console.log(`FINAL OPTIMIZED DISTANCE: ${globalBestLen.toFixed(2)} km`);
console.log(`Below 75,000 threshold: ${globalBestLen < 75000 ? "YES! SUCCESS!" : "NO"}`);
console.log(`========================================`);

fs.writeFileSync('D:\\IMI\\TERM 5\\DDDM VS\\pr152_below_75k.json', JSON.stringify({
    bestDist: Math.round(globalBestLen * 100) / 100,
    tour: globalBestTour
}));

