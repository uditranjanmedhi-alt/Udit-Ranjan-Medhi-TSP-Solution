const fs = require('fs');
const path = require('path');

function parseTsp(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    let inCoords = false;
    const coords = [];
    let name = path.basename(filePath);

    for (let rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.startsWith('NAME')) {
            const parts = line.split(':');
            if (parts.length > 1) name = parts[1].trim();
        } else if (line.startsWith('NODE_COORD_SECTION')) {
            inCoords = true;
            continue;
        } else if (line.startsWith('EOF')) {
            break;
        } else if (inCoords) {
            const parts = line.split(/\s+/);
            if (parts.length >= 3) {
                const x = parseFloat(parts[1]);
                const y = parseFloat(parts[2]);
                if (!isNaN(x) && !isNaN(y)) {
                    coords.push([x, y]);
                }
            }
        }
    }
    return { name, coords };
}

function dist(c1, c2) {
    const dx = c1[0] - c2[0];
    const dy = c1[1] - c2[1];
    return Math.hypot(dx, dy);
}

function tourLength(tour, coords) {
    let sum = 0;
    const n = tour.length;
    for (let i = 0; i < n; i++) {
        sum += dist(coords[tour[i]], coords[tour[(i + 1) % n]]);
    }
    return sum;
}

function solveNearestNeighbor(coords) {
    const n = coords.length;
    const visited = new Uint8Array(n);
    const tour = [0];
    visited[0] = 1;
    let curr = 0;

    for (let step = 1; step < n; step++) {
        let bestD = Infinity;
        let nextNode = -1;
        const cCurr = coords[curr];
        for (let j = 0; j < n; j++) {
            if (!visited[j]) {
                const d = dist(cCurr, coords[j]);
                if (d < bestD) {
                    bestD = d;
                    nextNode = j;
                }
            }
        }
        visited[nextNode] = 1;
        tour.push(nextNode);
        curr = nextNode;
    }
    return tour;
}

function run2Opt(tour, coords, maxPasses = 50) {
    const n = tour.length;
    let bestTour = tour.slice();
    let improved = true;
    let passes = 0;

    while (improved && passes < maxPasses) {
        improved = false;
        passes++;
        for (let i = 1; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const a = bestTour[i - 1], b = bestTour[i];
                const c = bestTour[j], d = bestTour[(j + 1) % n];

                const delta = (dist(coords[a], coords[c]) + dist(coords[b], coords[d])) -
                              (dist(coords[a], coords[b]) + dist(coords[c], coords[d]));

                if (delta < -1e-5) {
                    // Reverse subsegment i to j
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

const dir = 'D:\\IMI\\TERM 5\\DDDM VS\\TSP_Data';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.tsp')).sort();

console.log(`Processing ${files.length} instances...`);
const output = {};

for (const file of files) {
    const p = path.join(dir, file);
    const { name, coords } = parseTsp(p);
    const t0 = Date.now();
    const nnTour = solveNearestNeighbor(coords);
    const nnLen = tourLength(nnTour, coords);
    
    const optTour = run2Opt(nnTour, coords);
    const optLen = tourLength(optTour, coords);
    const ms = Date.now() - t0;

    output[file] = {
        name,
        nodes: coords.length,
        coords: coords,
        nnDist: Math.round(nnLen * 100) / 100,
        optDist: Math.round(optLen * 100) / 100,
        improvementPct: Math.round(((nnLen - optLen) / nnLen) * 1000) / 10,
        optTour: optTour,
        timeMs: ms
    };
    console.log(`[${file}] Nodes: ${coords.length} | NN: ${nnLen.toFixed(2)} km -> 2-Opt: ${optLen.toFixed(2)} km (-${output[file].improvementPct}%) [${ms}ms]`);
}

fs.writeFileSync('D:\\IMI\\TERM 5\\DDDM VS\\tsp_solutions.json', JSON.stringify(output));
console.log('Saved tsp_solutions.json successfully!');

