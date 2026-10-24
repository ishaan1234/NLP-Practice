// Procedural furniture for the "Add items" library. Every builder returns a
// Group standing on y=0, centred on x/z, facing +Z, sized in metres.
import * as THREE from 'three';

const mat = (color, rough = 0.8) => new THREE.MeshStandardMaterial({ color, roughness: rough });

function box(w, h, d, m, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y + h / 2, z);
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

function cyl(rt, rb, h, m, x = 0, y = 0, z = 0, seg = 20) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), m);
  mesh.position.set(x, y + h / 2, z);
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}

function legs(g, w, d, h, r, m, inset = 0.04) {
  for (const sx of [-1, 1]) for (const sz of [-1, 1])
    g.add(cyl(r, r, h, m, sx * (w / 2 - inset), 0, sz * (d / 2 - inset), 10));
}

export const LIBRARY = {
  chair: { icon: '🪑', color: '#8a6a4f', size: [0.45, 0.9, 0.5], build(c) {
    const g = new THREE.Group(), m = mat(c);
    legs(g, 0.45, 0.45, 0.45, 0.018, m);
    g.add(box(0.45, 0.05, 0.45, m, 0, 0.45, 0));
    g.add(box(0.45, 0.45, 0.04, m, 0, 0.5, -0.205));
    return g;
  } },
  table: { icon: '🟫', color: '#9b7b5c', size: [1.4, 0.75, 0.8], build(c) {
    const g = new THREE.Group(), m = mat(c, 0.6);
    legs(g, 1.4, 0.8, 0.71, 0.025, m, 0.07);
    g.add(box(1.4, 0.04, 0.8, m, 0, 0.71, 0));
    return g;
  } },
  desk: { icon: '🖥️', color: '#d8d2c6', size: [1.2, 0.75, 0.6], build(c) {
    const g = new THREE.Group(), m = mat(c, 0.5);
    g.add(box(0.03, 0.72, 0.6, m, -0.585, 0, 0));
    g.add(box(0.4, 0.72, 0.58, m, 0.4, 0, 0));
    g.add(box(1.2, 0.03, 0.6, m, 0, 0.72, 0));
    return g;
  } },
  sofa: { icon: '🛋️', color: '#5b6f86', size: [2.0, 0.85, 0.9], build(c) {
    const g = new THREE.Group(), m = mat(c, 0.95);
    g.add(box(2.0, 0.42, 0.9, m, 0, 0.0, 0));
    g.add(box(2.0, 0.45, 0.22, m, 0, 0.4, -0.34));
    g.add(box(0.2, 0.25, 0.9, m, -0.9, 0.42, 0));
    g.add(box(0.2, 0.25, 0.9, m, 0.9, 0.42, 0));
    for (const x of [-0.45, 0.45]) g.add(box(0.78, 0.12, 0.62, mat(c, 1), x, 0.42, 0.08));
    return g;
  } },
  bed: { icon: '🛏️', color: '#c9c3b8', size: [1.6, 0.9, 2.05], build(c) {
    const g = new THREE.Group(), frame = mat('#6e5844', 0.7);
    g.add(box(1.6, 0.3, 2.05, frame, 0, 0, 0));
    g.add(box(1.52, 0.22, 1.95, mat(c, 1), 0, 0.3, 0.02));
    g.add(box(1.6, 0.9, 0.06, frame, 0, 0, -1.0));
    for (const x of [-0.38, 0.38]) g.add(box(0.6, 0.12, 0.35, mat('#ffffff', 1), x, 0.52, -0.72));
    return g;
  } },
  shelf: { icon: '📚', color: '#b9a385', size: [0.8, 1.8, 0.35], build(c) {
    const g = new THREE.Group(), m = mat(c, 0.7);
    g.add(box(0.03, 1.8, 0.35, m, -0.385, 0, 0));
    g.add(box(0.03, 1.8, 0.35, m, 0.385, 0, 0));
    g.add(box(0.8, 1.8, 0.02, m, 0, 0, -0.165));
    for (let i = 0; i < 5; i++) g.add(box(0.74, 0.025, 0.33, m, 0, i * 0.44, 0));
    return g;
  } },
  cabinet: { icon: '🗄️', color: '#e5e1d8', size: [0.9, 0.85, 0.45], build(c) {
    const g = new THREE.Group(), m = mat(c, 0.5);
    g.add(box(0.9, 0.85, 0.45, m));
    const k = mat('#555', 0.3);
    for (const x of [-0.05, 0.05]) g.add(box(0.015, 0.12, 0.02, k, x, 0.6, 0.23));
    return g;
  } },
  lamp: { icon: '💡', color: '#f1e3c2', size: [0.4, 1.6, 0.4], build(c) {
    const g = new THREE.Group(), metal = mat('#333', 0.4);
    g.add(cyl(0.16, 0.16, 0.03, metal));
    g.add(cyl(0.012, 0.012, 1.35, metal, 0, 0.03));
    const shade = cyl(0.12, 0.2, 0.25, new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.35, side: THREE.DoubleSide }), 0, 1.35);
    g.add(shade);
    const light = new THREE.PointLight(0xffe2b0, 0.6, 4);
    light.position.y = 1.4;
    g.add(light);
    return g;
  } },
  plant: { icon: '🪴', color: '#4f7f4a', size: [0.5, 1.1, 0.5], build(c) {
    const g = new THREE.Group();
    g.add(cyl(0.2, 0.15, 0.35, mat('#b8643f', 0.9)));
    const leaf = mat(c, 0.9);
    for (let i = 0; i < 7; i++) {
      const s = new THREE.Mesh(new THREE.IcosahedronGeometry(0.17 + (i % 3) * 0.04, 0), leaf);
      const a = (i / 7) * Math.PI * 2;
      s.position.set(Math.cos(a) * 0.12, 0.55 + (i % 3) * 0.17, Math.sin(a) * 0.12);
      s.castShadow = true;
      g.add(s);
    }
    return g;
  } },
  tv: { icon: '📺', color: '#202226', size: [1.25, 0.75, 0.2], build(c) {
    const g = new THREE.Group();
    g.add(box(0.4, 0.04, 0.2, mat('#2a2a2a')));
    g.add(box(0.05, 0.08, 0.05, mat('#2a2a2a'), 0, 0.04));
    g.add(box(1.25, 0.7, 0.04, mat(c, 0.3), 0, 0.1));
    return g;
  } },
  rug: { icon: '🟪', color: '#a7685a', size: [2.0, 0.01, 1.4], build(c) {
    const g = new THREE.Group();
    const r = box(2.0, 0.01, 1.4, mat(c, 1));
    r.castShadow = false;
    g.add(r);
    return g;
  } },
  box: { icon: '📦', color: '#c8a46e', size: [0.5, 0.5, 0.5], build(c) {
    const g = new THREE.Group();
    g.add(box(0.5, 0.5, 0.5, mat(c)));
    return g;
  } },
};
