import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { LIBRARY } from './furniture.js';
import { normalizeLayout, alignTwin, applyAlignment } from './scanmerge.js';

THREE.Cache.enabled = true;
const $ = (id) => document.getElementById(id);
// On a shared deployment a room belongs to the browser that made it; this random key says which one.
const owner = (() => {
  const make = () => [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, '0')).join('');
  try { return (localStorage.roomTwinOwner ||= make()); } catch { return make(); }
})();
const api = (path, opts = {}) => fetch(path, { ...opts, headers: { ...opts.headers, 'X-Owner': owner } }).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});
let caps = {}; // what the server can do, from /api/capabilities
const deg = THREE.MathUtils.radToDeg, rad = THREE.MathUtils.degToRad;
const SNAP = 0.05;

// ---------------------------------------------------------------- three.js setup
const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x15171a : 0xeceae4);

const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 200);
camera.position.set(5, 6, 7);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.target.set(0, 0.8, 0);

scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8475, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, 7, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 0.5, far: 25 });
sun.shadow.bias = -0.0005;
scene.add(sun);

const roomGroup = new THREE.Group();
const objectsGroup = new THREE.Group();
scene.add(roomGroup, objectsGroup);

const transform = new TransformControls(camera, renderer.domElement);
transform.setTranslationSnap(SNAP);
transform.setRotationSnap(rad(15));
transform.showY = false;
scene.add(transform);

const selBox = new THREE.BoxHelper(new THREE.Object3D(), 0x2f9f84);
const hoverBox = new THREE.BoxHelper(new THREE.Object3D(), 0x999999);
hoverBox.material.transparent = true;
hoverBox.material.opacity = 0.6;
selBox.visible = hoverBox.visible = false;
scene.add(selBox, hoverBox);

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
renderer.setAnimationLoop(() => { orbit.update(); updateOverlays(); renderer.render(scene, camera); });

// ---------------------------------------------------------------- state
let project = null; // scene.json contents
const nodes = new Map(); // object id -> THREE.Group
let selected = null;
let hovered = null;
const undoStack = [];
const redoStack = [];
let saveTimer = null;

function updateOverlays() {
  const n = selected && nodes.get(selected);
  const o = n && byId(selected);
  selBox.visible = !!n && n.visible;
  const label = $('dimLabel');
  if (selBox.visible) {
    selBox.setFromObject(n);
    const b = new THREE.Box3().setFromObject(n);
    const s = o.size?.every((v) => v > 0) ? scaledSize(o) : b.getSize(new THREE.Vector3()).toArray();
    const top = new THREE.Vector3((b.min.x + b.max.x) / 2, b.max.y + 0.05, (b.min.z + b.max.z) / 2).project(camera);
    const visible = top.z < 1 && Math.abs(top.x) < 1 && Math.abs(top.y) < 1;
    label.classList.toggle('hidden', !visible || !!drag?.moved);
    label.style.left = `${((top.x + 1) / 2) * viewport.clientWidth}px`;
    label.style.top = `${((1 - top.y) / 2) * viewport.clientHeight}px`;
    label.textContent = `${o.locked ? '🔒 ' : ''}${o.label} · ${s[0].toFixed(2)} × ${s[2].toFixed(2)} × ${s[1].toFixed(2)} m`;
  } else label.classList.add('hidden');
  const h = hovered && hovered !== selected && nodes.get(hovered);
  hoverBox.visible = !!h && h.visible;
  if (hoverBox.visible) hoverBox.setFromObject(h);
}

function updateTips() {
  const tips = $('tips');
  tips.classList.toggle('hidden', !project);
  const o = selected && byId(selected);
  if (!o) tips.innerHTML = 'Click an object to select it · drag empty space to orbit · drag furniture in from the right · <kbd>?</kbd> shortcuts';
  else if (o.locked) tips.innerHTML = `<b>${escapeHtml(o.label)}</b> is locked · press <kbd>L</kbd> to unlock`;
  else tips.innerHTML = 'Drag to move · <kbd>Shift</kbd>+drag rotate · arrows nudge · <kbd>Alt</kbd> no wall snap · <kbd>Del</kbd> delete';
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function toast(msg, ms = 2500) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add('hidden'), ms);
}

function checkpoint() {
  if (!project) return;
  undoStack.push(JSON.stringify(project.objects));
  if (undoStack.length > 60) undoStack.shift();
  redoStack.length = 0;
  updateHistoryButtons();
}

function updateHistoryButtons() {
  $('undo').disabled = !undoStack.length;
  $('redo').disabled = !redoStack.length;
}

function markDirty() {
  if (!project) return;
  $('save').textContent = 'Save*';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 2500);
}

async function save() {
  if (!project) return;
  clearTimeout(saveTimer);
  project.name = $('sceneName').value || 'Untitled room';
  try {
    await api(`/api/projects/${project.id}/scene`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(project),
    });
  } catch (err) {
    $('save').textContent = 'Save*';
    return toast(`Not saved: ${err.message}`, 5000);
  }
  $('save').textContent = 'Saved';
  setTimeout(() => ($('save').textContent = 'Save'), 1200);
}

// ---------------------------------------------------------------- room
const panoShader = {
  vertexShader: `varying vec3 vWorld;
    void main() { vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `uniform sampler2D pano; uniform vec3 eye; uniform float panoYaw; varying vec3 vWorld;
    void main() {
      vec3 d = normalize(vWorld - eye);
      // the scene was turned by panoYaw to line the room up with the axes; undo it for the lookup
      float u = fract((atan(d.x, -d.z) - panoYaw) / 6.28318530718 + 0.5);
      float v = asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5;
      gl_FragColor = texture2D(pano, vec2(u, v));
      #include <colorspace_fragment>
    }`,
};
let panoTexture = null;

// Where the photos were taken from (moves when a LiDAR scan re-aligns the twin).
const camXZ = (r, y) => [r.cameraPos?.[0] || 0, y, r.cameraPos?.[1] || 0];

// Room box in world coordinates; the capture camera is at the origin, not necessarily the centre.
function roomBox() {
  const r = project.room;
  const [cx, cz] = r.center || [0, 0];
  return { cx, cz, x0: cx - r.width / 2, x1: cx + r.width / 2, z0: cz - r.depth / 2, z1: cz + r.depth / 2 };
}

function buildRoom() {
  roomGroup.clear();
  if (!project) return;
  const r = project.room;
  const usePano = $('photoWalls').checked && panoTexture;
  const wallMat = usePano
    ? new THREE.ShaderMaterial({ ...panoShader, side: THREE.BackSide,
        uniforms: { pano: { value: panoTexture }, eye: { value: new THREE.Vector3(...camXZ(r, r.cameraHeight)) },
          panoYaw: { value: r.panoYaw || 0 } } })
    : new THREE.MeshStandardMaterial({ color: r.wallColor, side: THREE.BackSide, roughness: 0.95 });
  const floorMat = usePano ? wallMat : new THREE.MeshStandardMaterial({ color: r.floorColor, side: THREE.BackSide, roughness: 0.9 });
  const hidden = new THREE.MeshBasicMaterial({ visible: false });
  // BoxGeometry face order: +x, -x, +y (ceiling), -y (floor), +z, -z. BackSide = dollhouse view.
  const shell = new THREE.Mesh(new THREE.BoxGeometry(r.width, r.height, r.depth),
    [wallMat, wallMat, hidden, floorMat, wallMat, wallMat]);
  const box = roomBox();
  shell.position.set(box.cx, r.height / 2, box.cz);
  shell.receiveShadow = true;
  shell.name = 'room-shell';
  roomGroup.add(shell);
  if (usePano) { // the photo material is unlit, so catch shadows on a transparent overlay
    const catcher = new THREE.Mesh(new THREE.PlaneGeometry(r.width, r.depth), new THREE.ShadowMaterial({ opacity: 0.25 }));
    catcher.rotation.x = -Math.PI / 2;
    catcher.position.set(box.cx, 0.002, box.cz);
    catcher.receiveShadow = true;
    roomGroup.add(catcher);
  }
  const grid = new THREE.GridHelper(Math.max(r.width, r.depth), Math.round(Math.max(r.width, r.depth) * 2), 0x888888, 0x888888);
  grid.material.opacity = usePano ? 0.0 : 0.12;
  grid.material.transparent = true;
  grid.position.set(box.cx, 0.003, box.cz);
  grid.name = 'grid';
  roomGroup.add(grid);
  // capture point marker
  const marker = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.2, 16), new THREE.MeshBasicMaterial({ color: 0x2f6f5e }));
  marker.rotation.x = Math.PI;
  marker.position.set(...camXZ(r, r.cameraHeight));
  marker.name = 'camera-marker';
  roomGroup.add(marker);
  for (const [k, id] of [['width', 'rW'], ['depth', 'rD'], ['height', 'rH']]) $(id).value = r[k];
  $('rFloor').value = r.floorColor;
  $('rWall').value = r.wallColor;
}

// ---------------------------------------------------------------- objects
const texLoader = new THREE.TextureLoader();
const gltfLoader = new GLTFLoader();
const plyLoader = new PLYLoader();

function fitInto(inner, size) {
  // scale a loaded model uniformly to fit the given box (null dims are unconstrained; a flat
  // depth <= 6 cm is ignored), then stand it on the floor
  const box = new THREE.Box3().setFromObject(inner);
  const s = box.getSize(new THREE.Vector3());
  if (size) {
    const ratios = [[size[0], s.x], [size[1], s.y], [size[2] > 0.06 ? size[2] : null, s.z]]
      .filter(([want, have]) => want > 0 && have > 1e-6).map(([want, have]) => want / have);
    if (ratios.length) inner.scale.multiplyScalar(Math.min(...ratios));
  }
  const b2 = new THREE.Box3().setFromObject(inner);
  const c = b2.getCenter(new THREE.Vector3());
  inner.position.sub(new THREE.Vector3(c.x, b2.min.y, c.z));
}

function buildCutout(o) {
  const g = new THREE.Group();
  const tex = texLoader.load(o.cutout);
  tex.colorSpace = THREE.SRGBColorSpace;
  const [w, h] = o.size;
  const m = new THREE.MeshStandardMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.9,
    color: o.color || '#ffffff' });
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), m);
  plane.position.y = h / 2;
  plane.castShadow = true;
  plane.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: tex, alphaTest: 0.5 });
  g.add(plane);
  // soft footprint so the card reads as an object standing on the floor
  if (o.mount !== 'wall') {
    const foot = new THREE.Mesh(new THREE.CircleGeometry(Math.max(w, o.size[2]) * 0.5, 24),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.12, depthWrite: false }));
    foot.rotation.x = -Math.PI / 2;
    foot.position.y = 0.004;
    foot.scale.y = Math.max(0.3, o.size[2] / Math.max(w, o.size[2]));
    g.add(foot);
  }
  return g;
}

function buildNode(o) {
  const g = new THREE.Group();
  g.userData.id = o.id;
  let inner;
  if (o.kind === 'primitive') {
    const def = LIBRARY[o.type] || LIBRARY.box;
    inner = def.build(o.color || def.color);
    inner.scale.set(o.size[0] / def.size[0], o.size[1] / def.size[1], o.size[2] / def.size[2]);
  } else if (o.kind === 'mesh' || o.kind === 'pointcloud') {
    inner = new THREE.Group();
    if (o.cutout && o.size?.every((v) => v > 0)) inner.add(buildCutout(o)); // photo card until the model loads
    const onLoad = (obj) => {
      inner.clear();
      const holder = new THREE.Group();
      holder.add(obj);
      holder.rotation.x = rad(o.modelRotX || 0);
      holder.rotation.y = rad(o.modelRotY || 0);
      const wrap = new THREE.Group();
      wrap.add(holder);
      fitInto(wrap, o.size);
      inner.add(wrap);
      obj.traverse((c) => {
        if (!c.isMesh) return;
        c.castShadow = c.receiveShadow = true;
        if (o.color) c.material = [].concat(c.material).map((m) => { const t = m.clone(); t.color?.set(o.color); return t; })[0];
      });
      // record the model's real size once, so resizing and the size label work in metres
      if (!o.size || o.size.some((v) => !(v > 0))) {
        o.size = new THREE.Box3().setFromObject(wrap).getSize(new THREE.Vector3()).toArray().map((v) => +v.toFixed(3));
        if (selected === o.id) fillInspector();
      }
    };
    if (o.kind === 'mesh') gltfLoader.load(o.asset, (gl) => onLoad(gl.scene), undefined, () => toast(`Could not load ${o.label}`));
    else plyLoader.load(o.asset, (geo) => onLoad(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.012, vertexColors: true }))));
  } else {
    inner = buildCutout(o);
  }
  g.add(inner);
  applyTransform(g, o);
  return g;
}

function applyTransform(g, o) {
  g.position.fromArray(o.position);
  g.rotation.set(0, o.rotationY || 0, 0);
  g.scale.fromArray(o.scale || [1, 1, 1]);
  g.visible = !o.hidden;
}

function rebuildObjects() {
  transform.detach();
  objectsGroup.clear();
  nodes.clear();
  for (const o of project.objects) {
    const n = buildNode(o);
    nodes.set(o.id, n);
    objectsGroup.add(n);
  }
  if (selected && !nodes.has(selected)) selected = null;
  select(selected);
  renderOutliner();
}

function addObject(o, { select: sel = true } = {}) {
  checkpoint();
  project.objects.push(o);
  const n = buildNode(o);
  nodes.set(o.id, n);
  objectsGroup.add(n);
  renderOutliner();
  if (sel) select(o.id);
  markDirty();
}

function removeObject(id) {
  checkpoint();
  project.objects = project.objects.filter((o) => o.id !== id);
  objectsGroup.remove(nodes.get(id));
  nodes.delete(id);
  if (selected === id) select(null);
  renderOutliner();
  markDirty();
}

const byId = (id) => project?.objects.find((o) => o.id === id);
const newId = () => Math.random().toString(16).slice(2, 10);

function syncFromNode(id) {
  const o = byId(id), n = nodes.get(id);
  if (!o || !n) return;
  o.position = n.position.toArray().map((v) => +v.toFixed(3));
  o.rotationY = +n.rotation.y.toFixed(4);
  o.scale = n.scale.toArray().map((v) => +v.toFixed(3));
}

// ---------------------------------------------------------------- selection + inspector
function select(id) {
  selected = id;
  const n = id && nodes.get(id);
  const o = id && byId(id);
  if (n && !o.locked && !o.hidden) transform.attach(n); else transform.detach();
  $('inspector').classList.toggle('hidden', !n);
  document.querySelectorAll('#outliner li').forEach((li) => li.classList.toggle('active', li.dataset.id === id));
  if (n) fillInspector();
  updateTips();
}

function toggleFlag(id, flag) {
  const o = byId(id);
  if (!o) return;
  checkpoint();
  o[flag] = !o[flag];
  applyTransform(nodes.get(id), o);
  if (id === selected) select(id);
  renderOutliner();
  markDirty();
}

function focusSelected() {
  const n = selected && nodes.get(selected);
  if (!n) return;
  const b = new THREE.Box3().setFromObject(n);
  const c = b.getCenter(new THREE.Vector3());
  const r = Math.max(0.6, b.getSize(new THREE.Vector3()).length());
  const dir = camera.position.clone().sub(orbit.target).normalize();
  orbit.target.copy(c);
  camera.position.copy(c).addScaledVector(dir, r * 1.8);
  orbit.update();
}

function rotateSelected(deltaDeg) {
  const o = byId(selected);
  if (!o || o.locked) return;
  checkpoint();
  o.rotationY = rad(Math.round((deg(o.rotationY || 0) + deltaDeg) * 100) / 100);
  o.rotationY = Math.atan2(Math.sin(o.rotationY), Math.cos(o.rotationY));
  applyTransform(nodes.get(o.id), o);
  fillInspector();
  markDirty();
}

function nudgeSelected(dx, dz) {
  const o = byId(selected);
  if (!o || o.locked) return;
  checkpoint();
  const n = nodes.get(o.id);
  n.position.x += dx;
  n.position.z += dz;
  keepInRoom(n, false);
  syncFromNode(o.id);
  fillInspector();
  markDirty();
}

function fillInspector() {
  const o = byId(selected);
  if (!o) return;
  $('iLabel').value = o.label;
  [$('iX').value, $('iY').value, $('iZ').value] = o.position.map((v) => (+v).toFixed(2));
  $('iRot').value = Math.round(deg(o.rotationY || 0));
  $('iScale').value = o.scale[0];
  const dims = scaledSize(o);
  [$('iW').value, $('iH').value, $('iD').value] = dims.map((v) => (v > 0 ? v.toFixed(2) : ''));
  $('iColor').value = o.color || (o.kind === 'primitive' ? LIBRARY[o.type]?.color : '#ffffff') || '#ffffff';
  $('iLock').checked = !!o.locked;
  $('iHide').checked = !!o.hidden;
  for (const id of ['iX', 'iY', 'iZ', 'iRot', 'iScale', 'iRotL', 'iRotR', 'iDrop', 'iW', 'iD', 'iH']) $(id).disabled = !!o.locked;
  const src = $('iSource');
  src.innerHTML = '';
  const info = document.createElement('div');
  info.className = 'muted';
  info.textContent = `${o.kind}${o.score ? ` · score ${o.score}` : ''}${o.source?.view ? ` · from ${o.source.view}` : ''} · ${scaledSize(o).map((v) => v.toFixed(2)).join(' × ')} m`;
  src.appendChild(info);
  if (o.license) {
    const credit = document.createElement('div');
    credit.className = 'credit';
    credit.textContent = `Model: ${o.license} `;
    if (o.creditUrl) credit.append(Object.assign(document.createElement('a'), { href: o.creditUrl, target: '_blank', rel: 'noreferrer', textContent: 'source' }));
    src.appendChild(credit);
  }
  if (o.cutout) {
    const img = document.createElement('img');
    img.src = o.cutout;
    src.appendChild(img);
  }
  if (o.kind === 'mesh' || o.kind === 'pointcloud') {
    const b = document.createElement('button');
    b.textContent = 'Re-orient model (90° X)';
    b.onclick = () => { checkpoint(); o.modelRotX = ((o.modelRotX || 0) + 90) % 360; refreshNode(o); markDirty(); };
    src.appendChild(b);
  }
}

function refreshNode(o) {
  const old = nodes.get(o.id);
  const n = buildNode(o);
  objectsGroup.remove(old);
  objectsGroup.add(n);
  nodes.set(o.id, n);
  if (selected === o.id) transform.attach(n);
}

function renderOutliner() {
  const ul = $('outliner');
  ul.innerHTML = '';
  for (const o of project?.objects || []) {
    const li = document.createElement('li');
    li.dataset.id = o.id;
    li.title = `${o.kind} · double-click to focus`;
    const thumb = o.cutout ? Object.assign(document.createElement('img'), { src: o.cutout, alt: '' })
      : Object.assign(document.createElement('span'), { textContent: LIBRARY[o.type]?.icon || '🧊' });
    thumb.className = 'thumb';
    const name = Object.assign(document.createElement('span'), { className: 'name', textContent: o.label });
    const eye = Object.assign(document.createElement('button'), {
      className: `ico${o.hidden ? ' on' : ''}`, textContent: o.hidden ? '🙈' : '👁', title: o.hidden ? 'Show (H)' : 'Hide (H)' });
    const lock = Object.assign(document.createElement('button'), {
      className: `ico${o.locked ? ' on' : ''}`, textContent: o.locked ? '🔒' : '🔓', title: o.locked ? 'Unlock (L)' : 'Lock (L)' });
    eye.onclick = (e) => { e.stopPropagation(); toggleFlag(o.id, 'hidden'); };
    lock.onclick = (e) => { e.stopPropagation(); toggleFlag(o.id, 'locked'); };
    li.append(thumb, name, eye, lock);
    li.onclick = () => select(o.id);
    li.ondblclick = () => { select(o.id); focusSelected(); };
    if (o.hidden) li.classList.add('hidden-obj');
    if (o.id === selected) li.classList.add('active');
    ul.appendChild(li);
  }
  $('objCount').textContent = project ? `(${project.objects.length})` : '';
}

const onInspector = (fn) => (e) => {
  const o = byId(selected);
  if (!o) return;
  fn(o, e.target.value);
  applyTransform(nodes.get(o.id), o);
  markDirty();
};
for (const el of ['iX', 'iY', 'iZ', 'iRot', 'iScale', 'iLabel', 'iColor']) $(el).addEventListener('focus', checkpoint);
$('iX').oninput = onInspector((o, v) => (o.position[0] = +v));
$('iY').oninput = onInspector((o, v) => (o.position[1] = +v));
$('iZ').oninput = onInspector((o, v) => (o.position[2] = +v));
$('iRot').oninput = onInspector((o, v) => (o.rotationY = rad(+v)));
$('iScale').oninput = onInspector((o, v) => (o.scale = [+v, +v, +v]));
$('iLabel').oninput = onInspector((o, v) => { o.label = v; renderOutliner(); });
$('iColor').oninput = (e) => {
  const o = byId(selected);
  if (!o) return;
  o.color = e.target.value;
  refreshNode(o);
  markDirty();
};
$('iDel').onclick = () => selected && removeObject(selected);
$('iRotL').onclick = () => rotateSelected(-90);
$('iRotR').onclick = () => rotateSelected(90);
$('iLock').onchange = () => selected && toggleFlag(selected, 'locked');
$('iHide').onchange = () => selected && toggleFlag(selected, 'hidden');
$('iFocus').onclick = focusSelected;
$('iDup').onclick = duplicate;
$('iDrop').onclick = () => {
  const o = byId(selected);
  if (!o) return;
  checkpoint();
  o.position[1] = 0;
  applyTransform(nodes.get(o.id), o);
  fillInspector();
  markDirty();
};

function duplicate() {
  const o = byId(selected);
  if (!o) return;
  const copy = structuredClone(o);
  copy.id = newId();
  copy.position = [o.position[0] + 0.3, o.position[1], o.position[2] + 0.3];
  addObject(copy);
}

transform.addEventListener('dragging-changed', (e) => {
  orbit.enabled = !e.value;
  if (e.value) checkpoint();
  else if (selected) { syncFromNode(selected); fillInspector(); markDirty(); }
});
transform.addEventListener('objectChange', () => {
  if (!selected) return;
  const n = nodes.get(selected);
  if (transform.mode === 'scale') { // keep proportions: follow whichever handle is dragged
    const axis = { X: 'x', Y: 'y', Z: 'z' }[transform.axis] || 'x';
    n.scale.setScalar(Math.max(0.05, n.scale[axis]));
  }
  n.position.y = Math.max(0, n.position.y);
  if (transform.mode === 'translate') keepInRoom(n, false);
});

// ---------------------------------------------------------------- picking + drag-on-floor
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let drag = null;

function setNdc(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
}

function pickObject(e) {
  setNdc(e);
  const hit = raycaster.intersectObjects(objectsGroup.children.filter((c) => c.visible), true)
    .find((h) => h.object.visible && !(h.object.material?.opacity < 0.2));
  let o = hit?.object;
  while (o && !o.userData.id) o = o.parent;
  return o?.userData.id || null;
}

function floorPoint(e, y = 0) {
  setNdc(e);
  const p = new THREE.Vector3();
  return raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -y), p) ? p : null;
}

const WALL_SNAP = 0.15;

// Keep the object's footprint inside the room; within WALL_SNAP of a wall, pull it flush.
function keepInRoom(n, snapToWall = true) {
  if (!project) return;
  const r = roomBox();
  const b = new THREE.Box3().setFromObject(n);
  const s = snapToWall ? WALL_SNAP : 0;
  for (const [axis, lo, hi] of [['x', r.x0, r.x1], ['z', r.z0, r.z1]]) {
    if (b.max[axis] - b.min[axis] >= hi - lo) continue; // bigger than the room: leave it be
    if (b.min[axis] < lo + s) n.position[axis] += lo - b.min[axis];
    else if (b.max[axis] > hi - s) n.position[axis] += hi - b.max[axis];
  }
}

// "here" for the AI: a click (not a drag) on empty floor drops a marker
const hereMarker = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.17, 32),
  new THREE.MeshBasicMaterial({ color: 0xe0784d, side: THREE.DoubleSide, depthWrite: false }));
hereMarker.rotation.x = -Math.PI / 2;
hereMarker.visible = false;
scene.add(hereMarker);
let aiHere = null, downAt = null;
function setHere(p) {
  aiHere = p ? [+p.x.toFixed(2), +p.z.toFixed(2)] : null;
  hereMarker.visible = !!p;
  if (p) hereMarker.position.set(p.x, 0.01, p.z);
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || transform.axis) return; // gizmo has priority
  const id = pickObject(e);
  downAt = id ? null : { x: e.clientX, y: e.clientY };
  select(id);
  const o = id && byId(id);
  if (!o || o.locked) return;
  const n = nodes.get(id);
  if (e.shiftKey) {
    drag = { id, rotate: true, startX: e.clientX, startRot: n.rotation.y, moved: false };
  } else {
    if (transform.mode !== 'translate') return;
    const p = floorPoint(e, n.position.y);
    if (!p) return;
    drag = { id, offset: n.position.clone().sub(p), moved: false };
  }
  orbit.enabled = false;
  renderer.domElement.classList.add('grabbing');
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!drag) {
    if (e.buttons) return;
    const id = project ? pickObject(e) : null;
    hovered = id;
    renderer.domElement.classList.toggle('grab', !!id && !byId(id)?.locked);
    return;
  }
  const n = nodes.get(drag.id);
  if (drag.rotate) {
    if (!drag.moved) { checkpoint(); drag.moved = true; }
    let a = drag.startRot + (e.clientX - drag.startX) * 0.012;
    if ($('snap').checked) a = Math.round(a / rad(15)) * rad(15);
    n.rotation.y = Math.atan2(Math.sin(a), Math.cos(a));
    return;
  }
  const p = floorPoint(e, n.position.y);
  if (!p) return;
  if (!drag.moved) { checkpoint(); drag.moved = true; }
  p.add(drag.offset);
  if ($('snap').checked) { p.x = Math.round(p.x / SNAP) * SNAP; p.z = Math.round(p.z / SNAP) * SNAP; }
  n.position.x = p.x;
  n.position.z = p.z;
  keepInRoom(n, !e.altKey);
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (downAt && project && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) < 5) {
    const p = floorPoint(e);
    const r = roomBox();
    if (p && p.x > r.x0 && p.x < r.x1 && p.z > r.z0 && p.z < r.z1) setHere(p);
  }
  downAt = null;
  if (!drag) return;
  if (drag.moved) { syncFromNode(drag.id); fillInspector(); markDirty(); }
  drag = null;
  orbit.enabled = true;
  renderer.domElement.classList.remove('grabbing');
});
renderer.domElement.addEventListener('pointerleave', () => { if (!drag) hovered = null; });

// ---------------------------------------------------------------- library + new items
function spawnPoint() {
  // on the floor where the camera looks, clamped into the room
  const p = new THREE.Vector3();
  raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
  const r = roomBox();
  if (!raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p)) p.set(r.cx, 0, r.cz);
  p.x = THREE.MathUtils.clamp(p.x, r.x0 + 0.5, r.x1 - 0.5);
  p.z = THREE.MathUtils.clamp(p.z, r.z0 + 0.5, r.z1 - 0.5);
  return p;
}

// Nearest spot (spiralling out from p) where a w×d footprint doesn't overlap anything.
function freeSpot(p, w, d) {
  const r = roomBox();
  const taken = [...nodes.values()].filter((n) => n.visible).map((n) => new THREE.Box3().setFromObject(n));
  const fits = (x, z) => {
    const b = new THREE.Box3(new THREE.Vector3(x - w / 2, 0.02, z - d / 2), new THREE.Vector3(x + w / 2, 1, z + d / 2));
    return !taken.some((t) => t.intersectsBox(b));
  };
  const clamp = (v, lo, hi, s) => THREE.MathUtils.clamp(v, lo + s / 2, hi - s / 2);
  for (let ring = 0; ring < 14; ring++) {
    const steps = Math.max(1, ring * 8);
    for (let k = 0; k < steps; k++) {
      const a = (k / steps) * Math.PI * 2;
      const x = clamp(p.x + Math.cos(a) * ring * 0.3, r.x0, r.x1, w);
      const z = clamp(p.z + Math.sin(a) * ring * 0.3, r.z0, r.z1, d);
      if (fits(x, z)) return new THREE.Vector3(x, 0, z);
    }
  }
  return p;
}

function addPrimitive(type, at) {
  if (!project) return toast('Open or build a room first');
  const def = LIBRARY[type];
  const p = at || freeSpot(spawnPoint(), def.size[0], def.size[2]);
  const o = { id: newId(), label: type, kind: 'primitive', type, color: def.color, size: [...def.size],
    position: [p.x, def.elevation || 0, p.z], rotationY: 0, scale: [1, 1, 1], mount: def.mount || 'floor' };
  addObject(o);
  keepInRoom(nodes.get(o.id), false);
  syncFromNode(o.id);
}

async function addItemFromFile(file) {
  if (!project) return toast('Open or build a room first');
  file = await shrinkForUpload(file);
  if (caps.maxUpload && file.size > caps.maxUpload) return toast('That photo is too big to upload here');
  const fd = new FormData();
  fd.append('file', file, file.name || 'pasted.png');
  fd.append('prompt', $('itemPrompt').value);
  toast('Cutting out the item and building it in 3D…', 60000);
  try {
    const item = await api(`/api/projects/${project.id}/items`, { method: 'POST', body: fd });
    const p = freeSpot(spawnPoint(), item.size[0], item.size[2]);
    item.position = [p.x, 0, p.z];
    addObject(item);
    toast(`Added ${item.label} — drag it into place`);
  } catch (err) { toast(`Failed: ${err.message}`); }
}

for (const [type, def] of Object.entries(LIBRARY)) {
  const b = document.createElement('button');
  b.innerHTML = `<span>${def.icon}</span>${type}`;
  b.draggable = true;
  b.onclick = () => addPrimitive(type);
  b.ondragstart = (e) => e.dataTransfer.setData('text/room-item', type);
  $('library').appendChild(b);
}
// Viewport accepts library pieces (dropped where the cursor is) and image files.
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
let dragDepth = 0;
viewport.addEventListener('dragenter', (e) => {
  if (hasFiles(e)) { dragDepth++; $('dropOverlay').classList.remove('hidden'); }
  $('dropOverlay').textContent = project ? 'Drop a photo to add it as an item' : 'Drop room photos to start a new twin';
});
viewport.addEventListener('dragleave', (e) => {
  if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $('dropOverlay').classList.add('hidden'); }
});
viewport.addEventListener('dragover', (e) => e.preventDefault());
viewport.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('dropOverlay').classList.add('hidden');
  const type = e.dataTransfer.getData('text/room-item');
  if (type) return addPrimitive(type, floorPoint(e) || undefined);
  const images = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'));
  if (!images.length) return;
  if (project) images.forEach(addItemFromFile);
  else { setFiles(images); toast('Photos added — check directions, then press Build twin'); }
});
addEventListener('paste', (e) => {
  if (e.target.matches?.('input, textarea')) return;
  const img = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
  if (img) { e.preventDefault(); addItemFromFile(img); }
});

$('itemPhoto').onchange = (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) addItemFromFile(file);
};

$('importGlb').onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || !project) return toast('Open or build a room first');
  if (caps.maxUpload && file.size > caps.maxUpload) {
    return toast(`Models over ${(caps.maxUpload / 1e6).toFixed(1)} MB can't be uploaded here`, 5000);
  }
  const fd = new FormData();
  fd.append('file', file);
  try {
    const { asset, kind } = await api(`/api/projects/${project.id}/assets`, { method: 'POST', body: fd });
    const p = freeSpot(spawnPoint(), 1, 1);
    addObject({ id: newId(), label: file.name.replace(/\.\w+$/, ''), kind, asset, size: null,
      position: [p.x, 0, p.z], rotationY: 0, scale: [1, 1, 1], mount: 'floor' });
  } catch (err) { toast(`Failed: ${err.message}`); }
};

// ---------------------------------------------------------------- room panel
for (const [id, key] of [['rW', 'width'], ['rD', 'depth'], ['rH', 'height']]) {
  $(id).onchange = (e) => { project.room[key] = Math.max(1, +e.target.value); buildRoom(); markDirty(); };
}
// ---------------------------------------------------------------- layout JSON (RoomShift format)
// Shared with the RoomShift editor / its iPhone LiDAR (RoomPlan) exporter: metres, Y-up,
// positions are box centres, rotation is radians about Y.
async function importLayout(file) {
  let scan;
  try { scan = normalizeLayout(JSON.parse(await file.text())); } catch (err) { return toast(err.message || 'Not a valid layout file'); }
  if (!project) await startBlank();
  if (!project) return;
  checkpoint();
  const photoObjs = project.objects.filter((o) => o.kind !== 'primitive');
  const fit = alignTwin(photoObjs, scan.objects);
  const matched = new Set();
  if (fit) {
    applyAlignment(project, fit);
    // snap each matched photo object onto its scanned box (exact position and size),
    // keeping whichever of the box's four headings is closest to how the photo card faces
    const taken = new Set();
    for (const [p, sObj] of [...fit.pairs].sort((a, b) => (b[0].score || 0) - (a[0].score || 0))) {
      if (taken.has(sObj.id) || matched.has(p.id)) continue;
      taken.add(sObj.id);
      matched.add(p.id);
      let best = 0, bestErr = Infinity;
      for (let k = 0; k < 4; k++) {
        const r = sObj.rotation + (k * Math.PI) / 2;
        const err = Math.abs(Math.atan2(Math.sin(r - p.rotationY), Math.cos(r - p.rotationY)));
        if (err < bestErr) { bestErr = err; best = k; }
      }
      const [w, h, d] = sObj.dimensions;
      p.size = best % 2 ? [d, h, w] : [w, h, d];
      p.rotationY = sObj.rotation + (best * Math.PI) / 2;
      p.position = [sObj.position[0], 0, sObj.position[2]];
      p.scanMatched = true;
    }
    scan.objects = scan.objects.filter((o) => !taken.has(o.id));
  } else if (photoObjs.length) {
    toast('Could not match the scan to the photo twin automatically; showing the scan alongside it', 5000);
  }
  Object.assign(project.room, { width: +scan.room.width.toFixed(3), depth: +scan.room.depth.toFixed(3),
    height: +scan.room.height.toFixed(3), center: scan.room.center, layout: 'scanned' });
  const add = (item, mount, label) => {
    const [w, h, d] = item.dimensions || [0.5, 0.5, 0.5];
    const [x, y, z] = item.position || [0, h / 2, 0];
    const type = LIBRARY[label] ? label : 'box';
    project.objects.push({ id: newId(), label, kind: 'primitive', type, color: LIBRARY[type].color,
      size: [w, h, d], position: [x, Math.max(0, y - h / 2), z], rotationY: item.rotation || 0,
      scale: [1, 1, 1], mount, fromScan: true });
  };
  for (const o of scan.openings) add(o, 'wall', o.kind === 'window' ? 'window' : 'door');
  for (const o of scan.objects) add(o, 'floor', o.label);
  buildRoom();
  rebuildObjects();
  setView('pers');
  markDirty();
  toast(fit
    ? `Scan merged: ${matched.size} photo objects snapped to it, camera height calibrated to ${project.room.cameraHeight} m`
    : `Imported scan: ${scan.room.width.toFixed(2)} × ${scan.room.depth.toFixed(2)} m, ${scan.openings.length + scan.objects.length} items`, 6000);
}

function exportLayout() {
  if (!project) return;
  const r = project.room;
  const [cx, cz] = r.center || [0, 0];
  const items = project.objects.filter((o) => !o.hidden).map((o) => {
    const size = (o.size || [0.5, 0.5, 0.5]).map((v, i) => +(v * (o.scale?.[i] ?? 1)).toFixed(3));
    return { id: o.id, kind: o.type || o.label, dimensions: size,
      position: [+(o.position[0] - cx).toFixed(3), +(o.position[1] + size[1] / 2).toFixed(3), +(o.position[2] - cz).toFixed(3)],
      rotation: +(o.rotationY || 0).toFixed(4) };
  });
  const layout = { format: 'roomshift-layout', version: 1, units: 'meters', coordinateSystem: 'right-handed-y-up',
    source: 'room-twin', room: { width: r.width, depth: r.depth, height: r.height },
    surfaces: items.filter((i) => ['door', 'window'].includes(i.kind)),
    objectLocations: items.filter((i) => !['door', 'window'].includes(i.kind)) };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(layout, null, 1)], { type: 'application/json' }));
  a.download = `${(project.name || 'room').replace(/\W+/g, '_')}.layout.json`;
  a.click();
}
$('layoutFile').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) importLayout(f); };
$('layoutFile2').onchange = $('layoutFile').onchange;
$('layoutExport').onclick = exportLayout;

$('rFloor').oninput = (e) => { project.room.floorColor = e.target.value; buildRoom(); markDirty(); };
$('rWall').oninput = (e) => { project.room.wallColor = e.target.value; buildRoom(); markDirty(); };
$('photoWalls').onchange = buildRoom;
$('sceneName').onchange = markDirty;

// ---------------------------------------------------------------- toolbar + keys
function setMode(mode) {
  transform.setMode(mode);
  transform.showX = transform.showZ = true;
  transform.showY = mode !== 'translate';
  if (mode === 'rotate') { transform.showX = transform.showZ = false; transform.showY = true; }
  document.querySelectorAll('#modeSeg button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
}
setMode('translate');
document.querySelectorAll('#modeSeg button').forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$('snap').onchange = (e) => {
  transform.setTranslationSnap(e.target.checked ? SNAP : null);
  transform.setRotationSnap(e.target.checked ? rad(15) : null);
};

function setView(kind) {
  const r = project?.room || { width: 5, depth: 4, cameraHeight: 1.5 };
  const { cx, cz } = project ? roomBox() : { cx: 0, cz: 0 };
  const span = Math.max(r.width, r.depth);
  if (kind === 'top') { camera.position.set(cx, span * 1.6, cz + 0.001); orbit.target.set(cx, 0, cz); }
  else if (kind === 'eye') {
    const [ex, ey, ez] = camXZ(r, r.cameraHeight);
    camera.position.set(ex, ey, ez + 0.01); orbit.target.set(ex, ey, ez - 0.5);
  }
  else { camera.position.set(cx + span * 0.9, span * 1.1, cz + span * 1.2); orbit.target.set(cx, 0.6, cz); }
  orbit.update();
}
$('topView').onclick = () => setView('top');
$('persView').onclick = () => setView('pers');
$('eyeView').onclick = () => setView('eye');

function undo() {
  if (!project || !undoStack.length) return;
  redoStack.push(JSON.stringify(project.objects));
  project.objects = JSON.parse(undoStack.pop());
  rebuildObjects();
  updateHistoryButtons();
  markDirty();
}
function redo() {
  if (!project || !redoStack.length) return;
  undoStack.push(JSON.stringify(project.objects));
  project.objects = JSON.parse(redoStack.pop());
  rebuildObjects();
  updateHistoryButtons();
  markDirty();
}
$('undo').onclick = undo;
$('redo').onclick = redo;

function screenshot() {
  const hide = [selBox, hoverBox, transform].filter((o) => o.visible);
  hide.forEach((o) => (o.visible = false));
  renderer.render(scene, camera);
  const a = document.createElement('a');
  a.href = renderer.domElement.toDataURL('image/png');
  a.download = `${(project?.name || 'room').replace(/\W+/g, '_')}.png`;
  a.click();
  hide.forEach((o) => (o.visible = true));
}
$('shot').onclick = screenshot;
const toggleHelp = (show) => $('help').classList.toggle('hidden', !(show ?? $('help').classList.contains('hidden')));
$('helpBtn').onclick = () => toggleHelp();
$('helpClose').onclick = () => toggleHelp(false);
$('save').onclick = () => save().catch((e) => toast(e.message));

addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const k = e.key.toLowerCase();
  const mod = e.ctrlKey || e.metaKey;
  if (mod && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
  else if (mod && k === 'z') { e.preventDefault(); undo(); }
  else if (e.key === '?') toggleHelp();
  else if (k === '[' || k === '{') rotateSelected(e.shiftKey ? -90 : -15);
  else if (k === ']' || k === '}') rotateSelected(e.shiftKey ? 90 : 15);
  else if (k.startsWith('arrow') && selected) {
    e.preventDefault();
    const step = e.shiftKey ? 0.5 : SNAP;
    // nudge relative to the camera so "up" always moves away from you
    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
    fwd.normalize();
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
    const v = { arrowup: fwd, arrowdown: fwd.clone().negate(), arrowright: right, arrowleft: right.clone().negate() }[k];
    // snap the direction to the nearest room axis so nudges stay on the grid
    const ax = Math.abs(v.x) > Math.abs(v.z) ? [Math.sign(v.x), 0] : [0, Math.sign(v.z)];
    nudgeSelected(ax[0] * step, ax[1] * step);
  }
  else if (k === 'f') focusSelected();
  else if (k === 'h' && selected) toggleFlag(selected, 'hidden');
  else if (k === 'l' && selected) toggleFlag(selected, 'locked');
  else if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save(); }
  else if ((e.ctrlKey || e.metaKey) && k === 'd') { e.preventDefault(); duplicate(); }
  else if (k === 'w') setMode('translate');
  else if (k === 'e') setMode('rotate');
  else if (k === 'r') setMode('scale');
  else if (k === 't') setView('top');
  else if (k === 'p') setView('pers');
  else if (k === 'v') setView('eye');
  else if (k === 'escape') { toggleHelp(false); select(null); setHere(null); }
  else if (e.key === '/' && project) { e.preventDefault(); $('aiInput').focus(); }
  else if ((k === 'delete' || k === 'backspace') && selected) removeObject(selected);
});

$('exportGlb').onclick = () => {
  if (!project) return;
  const out = new THREE.Scene();
  const r = project.room;
  const { cx, cz } = roomBox();
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(r.width, r.depth), new THREE.MeshStandardMaterial({ color: r.floorColor }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(cx, 0, cz);
  floor.name = 'floor';
  out.add(floor);
  const wm = new THREE.MeshStandardMaterial({ color: r.wallColor, side: THREE.DoubleSide });
  for (const [w, x, z, ry] of [[r.width, 0, -r.depth / 2, 0], [r.width, 0, r.depth / 2, Math.PI],
    [r.depth, -r.width / 2, 0, Math.PI / 2], [r.depth, r.width / 2, 0, -Math.PI / 2]]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(w, r.height), wm);
    wall.position.set(cx + x, r.height / 2, cz + z);
    wall.rotation.y = ry;
    out.add(wall);
  }
  for (const n of objectsGroup.children) {
    const c = n.clone(true);
    c.name = byId(n.userData.id)?.label || 'object';
    out.add(c);
  }
  new GLTFExporter().parse(out, (buf) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([buf], { type: 'model/gltf-binary' }));
    a.download = `${(project.name || 'room').replace(/\W+/g, '_')}.glb`;
    a.click();
  }, (err) => toast(`Export failed: ${err}`), { binary: true });
};

// ---------------------------------------------------------------- projects + upload
let pending = [];

const DIRECTIONS = [[0, 'ahead'], [45, 'ahead-right'], [90, 'right'], [135, 'behind-right'],
  [180, 'behind'], [225, 'behind-left'], [270, 'left'], [315, 'ahead-left']];

// Which way each photo faces, relative to the first one; arrow shows it at a glance.
function directionPicker(p, i) {
  const wrap = document.createElement('div');
  wrap.className = 'dir';
  const arrow = Object.assign(document.createElement('span'), { className: 'arrow', textContent: '↑' });
  const sel = document.createElement('select');
  sel.title = 'Which way the camera faced, compared to the first photo';
  const opts = DIRECTIONS.some(([d]) => d === p.yaw) ? DIRECTIONS : [...DIRECTIONS, [p.yaw, `${p.yaw}°`]];
  for (const [d, name] of opts.sort((a, b) => a[0] - b[0])) sel.add(new Option(i === 0 && d === 0 ? 'first' : name, d));
  sel.value = p.yaw;
  sel.disabled = i === 0;
  const sync = () => (arrow.style.transform = `rotate(${p.yaw}deg)`);
  sel.onchange = () => { p.yaw = +sel.value; sync(); };
  sync();
  wrap.append(arrow, sel);
  return wrap;
}

// Camera tilt per photo: auto-detected from vertical edges unless the user says otherwise.
function tiltPicker(p) {
  const sel = document.createElement('select');
  sel.title = 'How the camera was tilted. Auto works it out from door frames, corners and furniture legs.';
  for (const [v, name] of [['', 'tilt: auto'], ['0', 'level'], ['-10', 'down 10°'], ['-20', 'down 20°'],
    ['-30', 'down 30°'], ['10', 'up 10°']]) sel.add(new Option(name, v));
  sel.value = p.pitch ?? '';
  sel.onchange = () => (p.pitch = sel.value === '' ? null : +sel.value);
  const wrap = document.createElement('div');
  wrap.className = 'dir';
  wrap.appendChild(sel);
  return wrap;
}

async function startBlank(e) {
  e?.preventDefault();
  const fd = new FormData();
  fd.append('room_width', $('roomW').value);
  fd.append('room_depth', $('roomD').value);
  try {
    const { id } = await api('/api/projects/blank', { method: 'POST', body: fd });
    await openProject(id);
    refreshProjects();
    toast('Blank room ready — add furniture from the right panel');
  } catch (err) { toast(err.message); }
}
$('blank').onclick = startBlank;
$('hintBlank').onclick = startBlank;

function renderThumbs() {
  const box = $('thumbs');
  box.innerHTML = '';
  const multi = $('mode').value !== '360';
  pending.forEach((p, i) => {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = p.url;
    fig.appendChild(img);
    if (multi && pending.length > 1) fig.appendChild(directionPicker(p, i));
    if (multi) fig.appendChild(tiltPicker(p));
    box.appendChild(fig);
  });
  $('build').disabled = !pending.length;
}

function setFiles(files) {
  pending.forEach((p) => URL.revokeObjectURL(p.url));
  const list = [...files].filter((f) => f.type.startsWith('image/'));
  pending = list.map((f, i) => ({ file: f, url: URL.createObjectURL(f), yaw: Math.round((i * 360) / list.length) }));
  renderThumbs();
}

function addFiles(files) {
  const list = [...files].filter((f) => f.type.startsWith('image/'));
  pending.push(...list.map((f) => ({ file: f, url: URL.createObjectURL(f), yaw: 0 })));
  pending.forEach((p, i) => (p.yaw = Math.round((i * 360) / pending.length))); // re-spread the defaults
  renderThumbs();
}

$('files').onchange = (e) => setFiles(e.target.files);

// ---------------------------------------------------------------- phone capture (QR pairing)
let phone = null; // { id, owner, seen:Set, timer }

async function startPhone() {
  stopPhone();
  try {
    const s = await api('/api/capture', { method: 'POST' });
    phone = { ...s, seen: new Set(), timer: null };
    const qr = qrcode(0, 'M');
    qr.addData(s.url);
    qr.make();
    $('qr').innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
    $('phoneLink').textContent = s.url.split('#')[0];
    $('phoneLink').href = s.url;
    $('phoneWarn').textContent = s.warning || '';
    $('phoneWarn').classList.toggle('hidden', !s.warning);
    $('phoneStatus').textContent = "Scan with your phone's camera";
    $('phonePanel').classList.remove('hidden');
    $('phoneBtn').classList.add('hidden');
    phone.timer = setInterval(pollPhone, 1500);
  } catch (err) { toast(`Couldn't start phone capture: ${err.message}`); }
}

async function pollPhone() {
  if (!phone) return;
  const headers = { 'X-Capture-Owner': phone.owner };
  let state;
  try { state = await api(`/api/capture/${phone.id}`, { headers }); }
  catch (err) { toast(`Phone session ended: ${err.message}`); stopPhone(); return; }
  const fresh = state.photos.filter((p) => !phone.seen.has(p.id));
  fresh.forEach((p) => phone.seen.add(p.id));
  const n = phone.seen.size;
  $('phoneStatus').textContent = state.phone_connected
    ? `📱 Phone connected · ${n} photo${n === 1 ? '' : 's'} received` : "Scan with your phone's camera";
  for (const p of fresh) {
    const r = await fetch(`/api/capture/${phone.id}/photos/${p.id}`, { headers });
    if (!r.ok) continue;
    const file = new File([await r.blob()], `phone-${p.id}.jpg`, { type: 'image/jpeg' });
    if (p.mode === 'item' && project) addItemFromFile(file);
    else {
      addFiles([file]);
      if (p.mode === 'item') toast('Open a room first to drop items in — added it to the photo list for now');
    }
  }
}

function stopPhone() {
  if (!phone) return;
  clearInterval(phone.timer);
  fetch(`/api/capture/${phone.id}`, { method: 'DELETE', headers: { 'X-Capture-Owner': phone.owner } }).catch(() => {});
  phone = null;
  $('phonePanel').classList.add('hidden');
  $('phoneBtn').classList.remove('hidden');
}
$('phoneBtn').onclick = startPhone;
$('phoneEnd').onclick = stopPhone;
$('mode').onchange = renderThumbs;
const drop = $('drop');
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); setFiles(e.dataTransfer.files); };

// The camera info (Exif) of a JPEG with the rotation tag reset, for pixels that are already upright.
async function exifSegment(file) {
  try {
    const b = new DataView(await file.slice(0, 256 * 1024).arrayBuffer());
    if (b.getUint16(0) !== 0xffd8) return null;
    for (let off = 2; off + 4 <= b.byteLength;) {
      const marker = b.getUint16(off), len = b.getUint16(off + 2);
      if (marker === 0xffda || (marker & 0xff00) !== 0xff00) return null; // image data: no Exif
      if (marker === 0xffe1 && b.getUint32(off + 4) === 0x45786966) { // "Exif"
        if (off + 2 + len > b.byteLength) return null;
        const seg = new Uint8Array(b.buffer.slice(off, off + 2 + len));
        const v = new DataView(seg.buffer), tiff = 10, le = v.getUint16(tiff) === 0x4949;
        const ifd = tiff + v.getUint32(tiff + 4, le);
        for (let i = 0, n = v.getUint16(ifd, le); i < n; i++) {
          const e = ifd + 2 + i * 12;
          if (v.getUint16(e, le) === 0x0112) v.setUint16(e + 8, 1, le); // Orientation: upright
        }
        return seg;
      }
      off += 2 + len;
    }
  } catch { /* unreadable: send the photo without it */ }
  return null;
}

// Vercel takes at most 4.5 MB per request, so photos are shrunk in the browser to the size the
// server works at anyway (1600 px wide, 4096 for a panorama). The camera info goes along so the
// lens can still be read from it.
async function shrinkForUpload(file) {
  if (!caps.maxUpload) return file;
  let bmp;
  try { bmp = await createImageBitmap(file); } catch { return file; } // applies the Exif rotation
  const maxW = bmp.width / bmp.height >= 1.9 ? 4096 : 1600;
  if (bmp.width <= maxW && file.size < 1.5e6) { bmp.close(); return file; }
  const s = Math.min(1, maxW / bmp.width);
  const canvas = Object.assign(document.createElement('canvas'),
    { width: Math.round(bmp.width * s), height: Math.round(bmp.height * s) });
  canvas.getContext('2d').drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  const jpeg = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.9));
  const exif = await exifSegment(file);
  return new File(exif ? [jpeg.slice(0, 2), exif, jpeg.slice(2)] : [jpeg],
    `${file.name.replace(/\.\w+$/, '') || 'photo'}.jpg`, { type: 'image/jpeg' });
}

$('build').onclick = async () => {
  const prog = $('progress');
  prog.querySelector('.bar').style.width = '0';
  prog.querySelector('span').textContent = 'uploading…';
  prog.classList.remove('hidden');
  $('build').disabled = true;
  const files = await Promise.all(pending.map((p) => shrinkForUpload(p.file)));
  if (caps.maxUpload && files.reduce((n, f) => n + f.size, 0) > caps.maxUpload) {
    toast(`Those photos come to more than ${(caps.maxUpload / 1e6).toFixed(1)} MB, the upload limit here. Use fewer.`, 6000);
    prog.classList.add('hidden');
    $('build').disabled = false;
    return;
  }
  // a deployed server builds the twin before it answers, so there is no progress to poll
  if (caps.deployed) prog.querySelector('span').textContent = 'building… (a minute or two)';
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  fd.append('name', pending[0]?.file.name.replace(/\.\w+$/, '') || 'My room');
  fd.append('mode', $('mode').value);
  fd.append('yaws', JSON.stringify(pending.map((p) => p.yaw)));
  fd.append('pitches', JSON.stringify(pending.map((p) => p.pitch ?? null)));
  if ($('lens').value) fd.append('hfov', $('lens').value);
  fd.append('camera_height', $('camH').value);
  fd.append('room_width', $('roomW').value);
  fd.append('room_depth', $('roomD').value);
  fd.append('prompts', $('prompts').value);
  fd.append('reconstruct', $('recon').checked);
  try {
    const { id } = await api('/api/projects', { method: 'POST', body: fd });
    await poll(id);
    await openProject(id);
  } catch (err) {
    toast(`Build failed: ${err.message}`, 6000);
    prog.classList.add('hidden');
  }
  $('build').disabled = false;
  refreshProjects();
};

async function poll(id) {
  const prog = $('progress');
  prog.classList.remove('hidden');
  for (;;) {
    const s = await api(`/api/projects/${id}/status`);
    prog.querySelector('.bar').style.width = `${s.progress * 100}%`;
    prog.querySelector('span').textContent = s.stage;
    if (s.stage === 'done') break;
    if (s.stage === 'error') throw new Error(s.error);
    await new Promise((r) => setTimeout(r, 1000));
  }
  setTimeout(() => prog.classList.add('hidden'), 1500);
}

async function openProject(id) {
  project = await api(`/api/projects/${id}/scene`);
  undoStack.length = redoStack.length = 0;
  updateHistoryButtons();
  selected = hovered = null;
  $('sceneName').value = project.name || '';
  $('sceneName').disabled = false;
  $('hint').classList.add('hidden');
  $('roomPanel').classList.remove('hidden');
  $('aiForm').classList.remove('hidden');
  setHere(null);
  aiHistory.length = 0;
  lastChanged = null;
  panoTexture = null;
  if (project.panorama) {
    panoTexture = await texLoader.loadAsync(project.panorama);
    panoTexture.colorSpace = THREE.SRGBColorSpace;
    panoTexture.minFilter = THREE.LinearFilter; // no mipmaps -> no seam where u wraps
    panoTexture.generateMipmaps = false;
  }
  $('photoWalls').parentElement.classList.toggle('hidden', !panoTexture);
  buildRoom();
  rebuildObjects();
  setView('pers');
  const m = project.models || {};
  $('caps').textContent = project.views?.length
    ? `segmenter: ${m.segmenter || '?'} · 3D: ${m.reconstructor || 'photo cutouts'}` : 'blank room';
  updateTips();
  document.querySelectorAll('#projects li').forEach((li) => li.classList.toggle('active', li.dataset.id === id));
  history.replaceState(null, '', `#${id}`);
}

async function refreshProjects() {
  const ul = $('projects');
  const list = await api('/api/projects');
  ul.innerHTML = list.length ? '' : '<li class="muted">No projects yet</li>';
  for (const p of list) {
    const li = document.createElement('li');
    li.dataset.id = p.id;
    li.innerHTML = '<span></span><span><span class="muted"></span> <button title="Delete">✕</button></span>';
    li.children[0].textContent = p.name || p.id;
    li.querySelector('.muted').textContent = p.stage === 'done' ? '' : p.stage;
    li.onclick = () => (p.stage === 'done' ? openProject(p.id) : poll(p.id).then(() => openProject(p.id)).catch((e) => toast(e.message)));
    li.querySelector('button').onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${p.name}"?`)) return;
      await api(`/api/projects/${p.id}`, { method: 'DELETE' });
      if (project?.id === p.id) location.hash = '';
      refreshProjects();
    };
    if (project?.id === p.id) li.classList.add('active');
    ul.appendChild(li);
  }
}

// ---------------------------------------------------------------- sizes
// An object's size as shown to the user: its box times its (possibly per-axis) scale.
function scaledSize(o) {
  return (o.size || [0, 0, 0]).map((v, i) => (v || 0) * (o.scale?.[i] ?? 1));
}

// Set an object's size in metres; null keeps that dimension.
function resizeTo(o, w, h, d) {
  const cur = scaledSize(o);
  const want = [w, h, d].map((v, i) => (v > 0 ? v : cur[i]));
  const base = o.size || want;
  o.scale = want.map((v, i) => +(v / (base[i] || v || 1)).toFixed(4));
}

for (const [id, axis] of [['iW', 0], ['iH', 1], ['iD', 2]]) {
  $(id).addEventListener('focus', checkpoint);
  $(id).onchange = (e) => {
    const o = byId(selected);
    if (!o || !(+e.target.value > 0)) return;
    const dims = [null, null, null];
    dims[axis] = +e.target.value;
    resizeTo(o, ...dims);
    applyTransform(nodes.get(o.id), o);
    fillInspector();
    markDirty();
  };
}

// ---------------------------------------------------------------- free model search (Poly Haven, CC0)
let searchTimer = null;
$('modelSearch').oninput = (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (!q) { $('modelResults').innerHTML = ''; return; }
  searchTimer = setTimeout(async () => {
    let hits = [];
    try { hits = await api(`/api/models?q=${encodeURIComponent(q)}`); } catch { /* offline */ }
    const box = $('modelResults');
    box.innerHTML = hits.length ? '' : '<span class="muted small">No matches - try the AI bar, it can find or make anything.</span>';
    for (const h of hits.slice(0, 12)) {
      const b = document.createElement('button');
      b.title = `${h.name} (CC0, Poly Haven)`;
      b.append(Object.assign(document.createElement('img'), { src: h.thumb, alt: '', loading: 'lazy' }), h.name);
      b.onclick = () => addCatalogModel(h);
      box.appendChild(b);
    }
  }, 250);
};

async function addCatalogModel(h) {
  if (!project) return toast('Open or build a room first');
  toast(`Fetching ${h.name}…`, 20000);
  try {
    const info = await api(`/api/models/${encodeURIComponent(h.id)}`, { method: 'POST' });
    const p = aiHere ? new THREE.Vector3(aiHere[0], 0, aiHere[1]) : freeSpot(spawnPoint(), 0.8, 0.8);
    addObject({ id: newId(), label: h.name, kind: 'mesh', asset: info.asset, size: null, license: info.license,
      position: [p.x, 0, p.z], rotationY: faceRoomCentre(p.x, p.z), scale: [1, 1, 1], mount: 'floor' });
    toast(`Added ${h.name}`);
  } catch (err) { toast(`Couldn't fetch that model: ${err.message}`); }
}

function faceRoomCentre(x, z) {
  const { cx, cz } = roomBox();
  return Math.atan2(cx - x, cz - z);
}

// ---------------------------------------------------------------- AI assistant
const aiHistory = [];
let lastChanged = null;

function aiScene() {
  const r = roomBox();
  const round = (v) => +(+v).toFixed(2);
  const view = spawnPoint();
  return {
    room: { x_min: round(r.x0), x_max: round(r.x1), z_min: round(r.z0), z_max: round(r.z1), height: project.room.height },
    objects: project.objects.filter((o) => !o.hidden).slice(-150).map((o) => ({
      id: o.id, label: o.label, position: o.position.map(round), size: scaledSize(o).map(round),
      rotation_deg: Math.round(deg(o.rotationY || 0)), mount: o.mount, variant: o.variant || 0 })),
    selected, last_changed: lastChanged, here: aiHere, view_center: [round(view.x), round(view.z)],
  };
}

function objectFromAction(a) {
  const dims = [a.width, a.height, a.depth].map((v) => (v > 0 ? v : null));
  let o;
  if (a.item) { // generated: a cut-out (or SAM 3D mesh) made from a drawn picture
    o = { ...a.item };
    if (dims[1]) { const k = dims[1] / o.size[1]; o.size = o.size.map((v) => +(v * k).toFixed(3)); }
  } else if (a.asset) {
    o = { kind: 'mesh', asset: a.asset, size: dims.some((v) => v) ? dims : null };
  } else {
    const type = LIBRARY[a.library_type] ? a.library_type : 'box';
    const def = LIBRARY[type];
    o = { kind: 'primitive', type, color: a.color || def.color, size: dims.map((v, i) => v || def.size[i]) };
  }
  return { ...o, id: newId(), label: a.label || o.label || 'object', license: a.license, creditUrl: a.credit_url,
    variant: a.variant || 0, color: a.color || o.color, scale: [1, 1, 1],
    position: [a.x, a.elevation > 0 ? a.elevation : 0, a.z],
    rotationY: a.rotation_deg != null ? rad(a.rotation_deg) : faceRoomCentre(a.x, a.z),
    mount: a.elevation > 1.2 ? 'wall' : 'floor' };
}

function applyActions(actions) {
  if (!actions.length) return;
  checkpoint();
  const added = [];
  for (const a of actions) {
    const o = a.id && byId(a.id);
    if (a.op === 'add_object') {
      const n = objectFromAction(a);
      project.objects.push(n);
      added.push(n.id);
      lastChanged = n.id;
      continue;
    }
    if (!o) continue;
    lastChanged = o.id;
    if (a.op === 'resize_object') resizeTo(o, a.width, a.height, a.depth);
    else if (a.op === 'move_object') o.position = [a.x, a.elevation != null ? a.elevation : o.position[1], a.z];
    else if (a.op === 'rotate_object') o.rotationY = rad(a.rotation_deg);
    else if (a.op === 'recolor_object') o.color = a.color;
    else if (a.op === 'delete_object') { project.objects = project.objects.filter((x) => x !== o); lastChanged = null; }
    else if (a.op === 'replace_model' && (a.asset || a.item)) {
      const keep = { id: o.id, label: o.label, position: o.position, rotationY: o.rotationY, mount: o.mount };
      const target = scaledSize(o);
      const fresh = objectFromAction({ ...a, x: 0, z: 0, label: o.label });
      Object.keys(o).forEach((k) => delete o[k]);
      // keep the old height so the swap doesn't jump in size; the model keeps its proportions
      Object.assign(o, fresh, keep, { size: fresh.kind === 'mesh' ? [null, target[1] || null, null] : fresh.size, scale: [1, 1, 1] });
    }
  }
  rebuildObjects();
  for (const id of added) { const n = nodes.get(id); if (n && byId(id).mount === 'floor') { keepInRoom(n, false); syncFromNode(id); } }
  if (lastChanged && nodes.has(lastChanged)) select(lastChanged);
  markDirty();
}

function showReply(text, engine) {
  const el = $('aiReply');
  el.textContent = text;
  if (engine) el.appendChild(Object.assign(document.createElement('small'), { textContent: engine }));
  el.classList.remove('hidden');
  clearTimeout(el._h);
  el._h = setTimeout(() => el.classList.add('hidden'), 9000);
}

$('aiForm').onsubmit = async (e) => {
  e.preventDefault();
  const message = $('aiInput').value.trim();
  if (!message || !project) return;
  const form = $('aiForm');
  form.classList.add('busy');
  $('aiSend').disabled = true;
  showReply('Thinking…');
  try {
    const res = await api(`/api/projects/${project.id}/assistant`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, scene: aiScene(), history: aiHistory.slice(-6) }),
    });
    applyActions(res.actions || []);
    showReply(res.reply, res.engine && !String(res.engine).startsWith('claude') ? res.engine : '');
    aiHistory.push({ user: message, reply: res.reply });
    $('aiInput').value = '';
  } catch (err) {
    console.error(err);
    showReply(`Sorry - ${err.message}`);
  } finally {
    form.classList.remove('busy');
    $('aiSend').disabled = false;
  }
};

api('/api/capabilities').then((c) => {
  caps = c;
  if (c.assistant === false) $('aiInput').placeholder = "Ask: 'add a vase here', 'make it 20% bigger' (simple mode: set ANTHROPIC_API_KEY for full AI)";
  if (c.storage === 'temporary') toast('Rooms are not kept on this deployment yet: connect a Vercel Blob store to save them', 8000);
}).catch(() => {});

refreshProjects().then(() => {
  const id = location.hash.slice(1);
  if (id) openProject(id).catch(() => {});
});
