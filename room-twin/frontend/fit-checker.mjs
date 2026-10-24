// Geometry only: metres, Y up, Three.js positive-Y rotation convention.
export const EPS = 1e-6;
export const opening = (o) => ['door', 'window', 'opening'].includes(o.type || o.label);
export const solid = (o) => !o.hidden && !opening(o) && (o.type || o.label) !== 'rug';
export function box(o) {
  const size = (o.size || []).map((v, i) => v * Math.abs(o.scale?.[i] ?? 1));
  if (size.length !== 3 || !size.every((v) => Number.isFinite(v) && v > 0) ||
      !o.position?.every(Number.isFinite) || o.position.length !== 3 || !Number.isFinite(o.rotationY ?? 0)) return null;
  const a = o.rotationY || 0, c = Math.cos(a), s = Math.sin(a);
  const axes = [[c, -s], [s, c]];
  const corners = [];
  for (const x of [-1, 1]) for (const z of [-1, 1]) corners.push([
    o.position[0] + x * size[0] / 2 * c + z * size[2] / 2 * s,
    o.position[2] - x * size[0] / 2 * s + z * size[2] / 2 * c]);
  return { corners, axes, bottom: o.position[1], top: o.position[1] + size[1], size };
}
export function overlaps(a, b, padding = 0, vertical = true) {
  if (vertical && (a.top <= b.bottom + EPS || b.top <= a.bottom + EPS)) return false;
  return [...a.axes, ...b.axes].every(([x, z]) => {
    const p = a.corners.map((v) => v[0] * x + v[1] * z);
    const q = b.corners.map((v) => v[0] * x + v[1] * z);
    return Math.min(Math.max(...p), Math.max(...q)) - Math.max(Math.min(...p), Math.min(...q)) > -padding + EPS;
  });
}
export function inside(b, room) {
  const [cx, cz] = room.center || [0, 0];
  return b.bottom >= -EPS && b.top <= room.height + EPS && b.corners.every(([x, z]) =>
    x >= cx - room.width / 2 - EPS && x <= cx + room.width / 2 + EPS &&
    z >= cz - room.depth / 2 - EPS && z <= cz + room.depth / 2 + EPS);
}
export function inspect(objects, room) {
  const items = objects.filter(solid).map((o) => [o, box(o)]);
  const collisions = [], outside = [], unknown = [];
  for (let i = 0; i < items.length; i++) {
    const [o, b] = items[i];
    if (!b) { unknown.push(o.id); continue; }
    if (!inside(b, room)) outside.push(o.id);
    for (let j = 0; j < i; j++) if (items[j][1] && overlaps(b, items[j][1])) collisions.push([o.id, items[j][0].id]);
  }
  return { collisions, outside, unknown };
}
// Keep supporting furniture fixed: moving it alone would strand elevated decor.
export function movableIds(objects) {
  return new Set(objects.filter((o) => solid(o) && !o.locked && o.mount !== 'wall' &&
    Math.abs(o.position[1]) < EPS && box(o) && !objects.some((other) => {
      if (other.id === o.id || !solid(other) || !box(other)) return false;
      return other.position[1] > EPS && Math.abs(other.position[1] - box(o).top) < 0.1 &&
        overlaps(box(o), box(other), 0, false);
    })).map((o) => o.id));
}
