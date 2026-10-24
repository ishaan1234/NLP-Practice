import { box, overlaps, inside, inspect, movableIds, solid } from './fit-checker.mjs';

// A small occupancy grid estimates the largest connected area with walking clearance.
export function metrics(objects, room, original = objects) {
  const checks = inspect(objects, room), boxes = objects.filter(solid).map(box).filter(Boolean);
  let tight = 0;
  for (let i = 0; i < boxes.length; i++) for (let j = 0; j < i; j++) {
    if (overlaps(boxes[i], boxes[j], 0.6) && !overlaps(boxes[i], boxes[j])) tight++;
  }
  const nx = Math.min(30, Math.max(1, Math.ceil(room.width / 0.3)));
  const nz = Math.min(30, Math.max(1, Math.ceil(room.depth / 0.3)));
  const dx = room.width / nx, dz = room.depth / nz, [cx, cz] = room.center || [0, 0];
  const free = new Set();
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const probe = box({ position: [cx - room.width / 2 + (x + 0.5) * dx, 0, cz - room.depth / 2 + (z + 0.5) * dz], size: [0.6, 1.8, 0.6] });
    if (inside(probe, room) && !boxes.some((b) => overlaps(probe, b))) free.add(z * nx + x);
  }
  let connected = 0;
  while (free.size) {
    const queue = [free.values().next().value]; free.delete(queue[0]);
    for (let k = 0; k < queue.length; k++) {
      const n = queue[k], x = n % nx, z = Math.floor(n / nx);
      for (const next of [x > 0 ? n - 1 : -1, x < nx - 1 ? n + 1 : -1, z > 0 ? n - nx : -1, z < nz - 1 ? n + nx : -1]) {
        if (free.delete(next)) queue.push(next);
      }
    }
    connected = Math.max(connected, queue.length * dx * dz);
  }
  let movement = 0, moved = 0;
  for (const o of objects) {
    const before = original.find((v) => v.id === o.id);
    if (!before) continue;
    const distance = Math.hypot(o.position[0] - before.position[0], o.position[2] - before.position[2]);
    const angle = Math.abs(Math.atan2(Math.sin((o.rotationY || 0) - (before.rotationY || 0)), Math.cos((o.rotationY || 0) - (before.rotationY || 0))));
    movement += distance + angle * 0.2;
    if (distance > 0.001 || angle > 0.001) moved++;
  }
  return { ...checks, tight, connected, movement, moved,
    score: -10000 * (checks.collisions.length + checks.outside.length) + connected * 10 - tight * 2 - movement * 0.4 };
}

export async function optimize(objects, room, { cancelled = () => false, budgetMs = 1800 } = {}) {
  if (![room.width, room.depth, room.height].every((v) => Number.isFinite(v) && v > 0)) return { reason: 'Room dimensions are unavailable.' };
  if (objects.length > 45) return { reason: 'This demo supports up to 45 objects per room.' };
  const original = structuredClone(objects), baseline = metrics(original, room, original);
  if (baseline.unknown.length) return { reason: 'Some furniture dimensions are unavailable. Wait for models to load or enter their sizes.' };
  const ids = movableIds(original);
  if (!ids.size) return { reason: 'No movable floor furniture. Unlock furniture to try a suggestion.' };
  const start = performance.now(), [cx, cz] = room.center || [0, 0];
  let best = null, bestMetrics = baseline, operations = 0;
  const pause = async () => { if (++operations % 60 === 0) await new Promise((resolve) => setTimeout(resolve, 0)); };
  for (let attempt = 0; attempt < 8; attempt++) {
    const scene = structuredClone(original);
    const moving = scene.filter((o) => ids.has(o.id)).sort((a, b) => box(b).size[0] * box(b).size[2] - box(a).size[0] * box(a).size[2]);
    if (attempt % 2) moving.reverse();
    if (attempt > 1) moving.push(...moving.splice(0, attempt % moving.length));
    const placed = scene.filter((o) => !ids.has(o.id) && solid(o));
    let failed = false;
    for (const o of moving) {
      let candidate = null, candidateScore = -Infinity;
      const positions = [[...o.position]];
      const step = Math.max(0.4, Math.max(room.width, room.depth) / 14);
      for (let x = cx - room.width / 2; x <= cx + room.width / 2; x += step)
        for (let z = cz - room.depth / 2; z <= cz + room.depth / 2; z += step) positions.push([x, o.position[1], z]);
      for (const angle of [...new Set([o.rotationY || 0, 0, Math.PI / 2, Math.PI, -Math.PI / 2])]) {
        const extent = box({ ...o, position: [0, o.position[1], 0], rotationY: angle });
        const ex = Math.max(...extent.corners.map((p) => Math.abs(p[0]))), ez = Math.max(...extent.corners.map((p) => Math.abs(p[1])));
        for (const p of positions) {
          await pause();
          if (cancelled()) return { reason: 'Search cancelled.' };
          if (performance.now() - start > budgetMs) return best ? { objects: best, baseline, result: bestMetrics } : { reason: 'No valid improvement found within the search budget. Try locking fewer items or removing an obstruction.' };
          const position = [Math.max(cx - room.width / 2 + ex, Math.min(cx + room.width / 2 - ex, p[0])), p[1], Math.max(cz - room.depth / 2 + ez, Math.min(cz + room.depth / 2 - ez, p[2]))];
          const trial = { ...o, position, rotationY: angle }, b = box(trial);
          if (!inside(b, room) || placed.some((v) => overlaps(b, box(v)))) continue;
          const tight = placed.filter((v) => overlaps(b, box(v), 0.6)).length;
          const central = Math.hypot(position[0] - cx, position[2] - cz);
          const distance = Math.hypot(position[0] - o.position[0], position[2] - o.position[2]);
          const score = -tight * 3 + central * (attempt % 3 === 0 ? 0.15 : 0.7) - distance * 0.15;
          if (score > candidateScore) { candidate = trial; candidateScore = score; }
        }
      }
      if (!candidate) { failed = true; break; }
      Object.assign(o, candidate); placed.push(o);
    }
    if (!failed) {
      const score = metrics(scene, room, original);
      if (!score.collisions.length && !score.outside.length && score.score > bestMetrics.score + 0.01) { best = structuredClone(scene); bestMetrics = score; }
    }
    if (cancelled()) return { reason: 'Search cancelled.' };
  }
  return best ? { objects: best, baseline, result: bestMetrics } : { reason: 'No valid arrangement scored better. Your furniture has been kept in place.' };
}

// Preview snapshots contain transforms only; the caller chooses when to persist them.
export function transforms(objects) {
  return objects.map((o) => ({ id: o.id, position: [...o.position], rotationY: o.rotationY || 0 }));
}
