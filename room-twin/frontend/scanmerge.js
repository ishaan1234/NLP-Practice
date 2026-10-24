// Import a roomshift-layout JSON (RoomShift / Apple RoomPlan LiDAR scan) and, when the
// current project is a photo twin, merge the two: the scan gives exact walls and furniture
// boxes, the photos give the look.
//
// RoomPlan coordinates follow wherever the phone pointed when scanning started, so first
// the scan is straightened so its walls run along x / z. Then the photo twin (also wall-
// aligned) is matched to it: 4 possible quarter turns x a scale (the photo twin's scale
// rests on the assumed camera height) x an offset proposed by each pair of same-type objects.
// The best fit moves the photo objects onto the scan and calibrates the camera height.

// RoomPlan CapturedRoom.Object.Category -> our labels / library types
const CATEGORY = { storage: 'cabinet', television: 'tv', sofa: 'sofa', table: 'table', chair: 'chair',
  bed: 'bed', fireplace: 'cabinet', refrigerator: 'wardrobe', stove: 'cabinet', oven: 'cabinet',
  dishwasher: 'cabinet', washerDryer: 'cabinet', sink: 'cabinet', toilet: 'stool', bathtub: 'bed', stairs: 'box' };
const GROUPS = [['desk', 'table', 'cabinet', 'nightstand', 'shelf', 'wardrobe'], ['chair', 'armchair', 'stool', 'ottoman', 'bench'],
  ['sofa', 'armchair', 'bed', 'bench'], ['tv', 'monitor', 'picture frame', 'mirror']];
const compatible = (a, b) => a === b || a === 'object' || b === 'object' || GROUPS.some((g) => g.includes(a) && g.includes(b));

// Three.js-style rotation about +Y: (x, z) -> (x cos + z sin, -x sin + z cos); yaw' = yaw + a
const rot = ([x, z], a) => [x * Math.cos(a) + z * Math.sin(a), -x * Math.sin(a) + z * Math.cos(a)];

export function normalizeLayout(j) {
  if (j.format !== 'roomshift-layout' || !j.room) throw new Error('Expected a roomshift-layout JSON file');
  const walls = (j.surfaces || []).filter((s) => s.kind === 'wall');
  // dominant wall direction, folded into a quarter turn (doubling the angle 4x averages mod 90 deg)
  let c = 0, s = 0;
  for (const w of walls) { const len = w.dimensions?.[0] || 1; c += len * Math.cos(4 * (w.rotation || 0)); s += len * Math.sin(4 * (w.rotation || 0)); }
  const straighten = walls.length ? -Math.atan2(s, c) / 4 : 0;
  const map = (e) => {
    const [x, z] = rot([e.position[0], e.position[2]], straighten);
    return { ...e, position: [x, e.position[1], z], rotation: (e.rotation || 0) + straighten };
  };
  const surfaces = (j.surfaces || []).map(map);
  const objects = (j.objectLocations || []).map(map).map((o) => ({ ...o, label: CATEGORY[o.kind] || o.kind || 'object' }));
  // room box from the straightened wall ends (falls back to the declared size)
  let room = { width: +j.room.width || 5, depth: +j.room.depth || 4, height: +j.room.height || 2.7, center: [0, 0] };
  if (walls.length) {
    const xs = [], zs = [];
    for (const w of surfaces.filter((q) => q.kind === 'wall')) {
      const half = (w.dimensions[0] || 0) / 2;
      for (const sgn of [-1, 1]) {
        const [dx, dz] = rot([sgn * half, 0], w.rotation);
        xs.push(w.position[0] + dx); zs.push(w.position[2] + dz);
      }
    }
    const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
    room = { ...room, width: x1 - x0, depth: z1 - z0, center: [(x0 + x1) / 2, (z0 + z1) / 2] };
  }
  return { room, openings: surfaces.filter((q) => ['door', 'window', 'opening'].includes(q.kind)), objects, straighten };
}

// Best similarity (quarter turn, scale, offset) taking photo-twin floor objects onto scan objects.
export function alignTwin(photoObjs, scanObjs) {
  const P = photoObjs.filter((o) => o.mount !== 'wall');
  if (P.length < 2 || scanObjs.length < 2) return null;
  let best = null;
  for (let q = 0; q < 4; q++) {
    const a = (q * Math.PI) / 2;
    for (let k = 0.6; k <= 1.6001; k += 0.02) {
      for (const p of P) for (const s of scanObjs) {
        if (!compatible(p.label, s.label)) continue;
        const [px, pz] = rot([p.position[0] * k, p.position[2] * k], a);
        const t = [s.position[0] - px, s.position[2] - pz];
        let score = 0;
        const pairs = [];
        for (const p2 of P) {
          const [x, z] = rot([p2.position[0] * k, p2.position[2] * k], a);
          let bestD = 0.6, match = null;
          for (const s2 of scanObjs) {
            if (!compatible(p2.label, s2.label)) continue;
            const d = Math.hypot(x + t[0] - s2.position[0], z + t[1] - s2.position[2]);
            if (d < bestD) { bestD = d; match = s2; }
          }
          if (match) { score += 1 - bestD / 0.6 + (match.label === p2.label ? 0.5 : 0); pairs.push([p2, match]); }
        }
        if (!best || score > best.score) best = { score, angle: a, scale: k, offset: t, pairs };
      }
    }
  }
  if (!best || best.pairs.length < 2) return null;
  // refine scale + offset by least squares on the matched pairs (the search used 0.02 steps)
  const A = best.pairs.map(([p]) => rot([p.position[0], p.position[2]], best.angle));
  const B = best.pairs.map(([, s]) => [s.position[0], s.position[2]]);
  const mean = (v) => [v.reduce((t, q) => t + q[0], 0) / v.length, v.reduce((t, q) => t + q[1], 0) / v.length];
  const [ax, az] = mean(A), [bx, bz] = mean(B);
  let num = 0, den = 0;
  A.forEach(([x, z], i) => { num += (x - ax) * (B[i][0] - bx) + (z - az) * (B[i][1] - bz); den += (x - ax) ** 2 + (z - az) ** 2; });
  if (den > 1e-6) {
    const k = Math.min(1.8, Math.max(0.5, num / den));
    best.scale = k;
    best.offset = [bx - k * ax, bz - k * az];
  }
  return best;
}

export function applyAlignment(project, fit) {
  const { angle, scale, offset } = fit;
  const move = ([x, z]) => { const [rx, rz] = rot([x * scale, z * scale], angle); return [rx + offset[0], rz + offset[1]]; };
  for (const o of project.objects) {
    const [x, z] = move([o.position[0], o.position[2]]);
    o.position = [x, o.position[1] * scale, z];
    o.rotationY = (o.rotationY || 0) + angle;
    if (o.size) o.size = o.size.map((v) => v * scale);
  }
  const r = project.room;
  const [cx, cz] = move([0, 0]);
  r.cameraPos = [cx, cz];
  r.cameraHeight = +(r.cameraHeight * scale).toFixed(3); // the scan calibrates the camera height
  r.panoYaw = (r.panoYaw || 0) - angle;
}
