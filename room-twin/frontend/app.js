import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { LIBRARY } from './furniture.js';

THREE.Cache.enabled = true;
const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});
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

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
renderer.setAnimationLoop(() => { orbit.update(); renderer.render(scene, camera); });

// ---------------------------------------------------------------- state
let project = null; // scene.json contents
const nodes = new Map(); // object id -> THREE.Group
let selected = null;
const undoStack = [];
let saveTimer = null;

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
  await api(`/api/projects/${project.id}/scene`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(project),
  });
  $('save').textContent = 'Saved';
  setTimeout(() => ($('save').textContent = 'Save'), 1200);
}

// ---------------------------------------------------------------- room
const panoShader = {
  vertexShader: `varying vec3 vWorld;
    void main() { vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `uniform sampler2D pano; uniform vec3 eye; varying vec3 vWorld;
    void main() {
      vec3 d = normalize(vWorld - eye);
      float u = atan(d.x, -d.z) / 6.28318530718 + 0.5;
      float v = asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5;
      gl_FragColor = texture2D(pano, vec2(u, v));
      #include <colorspace_fragment>
    }`,
};
let panoTexture = null;

function buildRoom() {
  roomGroup.clear();
  if (!project) return;
  const r = project.room;
  const usePano = $('photoWalls').checked && panoTexture;
  const wallMat = usePano
    ? new THREE.ShaderMaterial({ ...panoShader, side: THREE.BackSide,
        uniforms: { pano: { value: panoTexture }, eye: { value: new THREE.Vector3(0, r.cameraHeight, 0) } } })
    : new THREE.MeshStandardMaterial({ color: r.wallColor, side: THREE.BackSide, roughness: 0.95 });
  const floorMat = usePano ? wallMat : new THREE.MeshStandardMaterial({ color: r.floorColor, side: THREE.BackSide, roughness: 0.9 });
  const hidden = new THREE.MeshBasicMaterial({ visible: false });
  // BoxGeometry face order: +x, -x, +y (ceiling), -y (floor), +z, -z. BackSide = dollhouse view.
  const shell = new THREE.Mesh(new THREE.BoxGeometry(r.width, r.height, r.depth),
    [wallMat, wallMat, hidden, floorMat, wallMat, wallMat]);
  shell.position.y = r.height / 2;
  shell.receiveShadow = true;
  shell.name = 'room-shell';
  roomGroup.add(shell);
  if (usePano) { // the photo material is unlit, so catch shadows on a transparent overlay
    const catcher = new THREE.Mesh(new THREE.PlaneGeometry(r.width, r.depth), new THREE.ShadowMaterial({ opacity: 0.25 }));
    catcher.rotation.x = -Math.PI / 2;
    catcher.position.y = 0.002;
    catcher.receiveShadow = true;
    roomGroup.add(catcher);
  }
  const grid = new THREE.GridHelper(Math.max(r.width, r.depth), Math.round(Math.max(r.width, r.depth) * 2), 0x888888, 0x888888);
  grid.material.opacity = usePano ? 0.0 : 0.12;
  grid.material.transparent = true;
  grid.position.y = 0.003;
  grid.name = 'grid';
  roomGroup.add(grid);
  // capture point marker
  const marker = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.2, 16), new THREE.MeshBasicMaterial({ color: 0x2f6f5e }));
  marker.rotation.x = Math.PI;
  marker.position.set(0, r.cameraHeight, 0);
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
  // scale a loaded model uniformly into the estimated box, stand it on the floor
  const box = new THREE.Box3().setFromObject(inner);
  const s = box.getSize(new THREE.Vector3());
  if (size) {
    const k = Math.min(size[0] / (s.x || 1), size[1] / (s.y || 1), (size[2] > 0.06 ? size[2] / (s.z || 1) : Infinity));
    inner.scale.multiplyScalar(isFinite(k) ? k : 1);
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
    const placeholder = buildCutout({ ...o, cutout: o.cutout || '' });
    if (o.cutout) inner.add(placeholder);
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
      obj.traverse((c) => { if (c.isMesh) c.castShadow = c.receiveShadow = true; });
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
  if (n) transform.attach(n); else transform.detach();
  $('inspector').classList.toggle('hidden', !n);
  document.querySelectorAll('#outliner li').forEach((li) => li.classList.toggle('active', li.dataset.id === id));
  if (n) fillInspector();
}

function fillInspector() {
  const o = byId(selected);
  if (!o) return;
  $('iLabel').value = o.label;
  [$('iX').value, $('iY').value, $('iZ').value] = o.position.map((v) => (+v).toFixed(2));
  $('iRot').value = Math.round(deg(o.rotationY || 0));
  $('iScale').value = o.scale[0];
  $('iColor').value = o.color || (o.kind === 'primitive' ? LIBRARY[o.type]?.color : '#ffffff') || '#ffffff';
  const src = $('iSource');
  src.innerHTML = '';
  const info = document.createElement('div');
  info.className = 'muted';
  info.textContent = `${o.kind}${o.score ? ` · score ${o.score}` : ''}${o.source?.view ? ` · from ${o.source.view}` : ''} · ${o.size.map((v) => v.toFixed(2)).join(' × ')} m`;
  src.appendChild(info);
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
    li.innerHTML = `<span></span><span class="muted"></span>`;
    li.children[0].textContent = o.label;
    li.children[1].textContent = o.kind;
    li.onclick = () => select(o.id);
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
  const hit = raycaster.intersectObjects(objectsGroup.children, true)
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

function clampToRoom(n) {
  if (!project) return;
  const r = project.room;
  n.position.x = THREE.MathUtils.clamp(n.position.x, -r.width / 2, r.width / 2);
  n.position.z = THREE.MathUtils.clamp(n.position.z, -r.depth / 2, r.depth / 2);
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || transform.axis) return; // gizmo has priority
  const id = pickObject(e);
  select(id);
  if (!id || transform.mode !== 'translate') return;
  const n = nodes.get(id);
  const p = floorPoint(e, n.position.y);
  if (!p) return;
  drag = { id, offset: n.position.clone().sub(p), moved: false };
  orbit.enabled = false;
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const n = nodes.get(drag.id);
  const p = floorPoint(e, n.position.y);
  if (!p) return;
  if (!drag.moved) { checkpoint(); drag.moved = true; }
  p.add(drag.offset);
  if ($('snap').checked) { p.x = Math.round(p.x / SNAP) * SNAP; p.z = Math.round(p.z / SNAP) * SNAP; }
  n.position.x = p.x;
  n.position.z = p.z;
  clampToRoom(n);
});
renderer.domElement.addEventListener('pointerup', () => {
  if (!drag) return;
  if (drag.moved) { syncFromNode(drag.id); fillInspector(); markDirty(); }
  drag = null;
  orbit.enabled = true;
});

// ---------------------------------------------------------------- library + new items
function spawnPoint() {
  // on the floor where the camera looks, clamped into the room
  const p = new THREE.Vector3();
  raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
  if (!raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p)) p.set(0, 0, 0);
  const r = project.room;
  p.x = THREE.MathUtils.clamp(p.x, -r.width / 2 + 0.5, r.width / 2 - 0.5);
  p.z = THREE.MathUtils.clamp(p.z, -r.depth / 2 + 0.5, r.depth / 2 - 0.5);
  return p;
}

function addPrimitive(type, at) {
  if (!project) return toast('Open or build a room first');
  const def = LIBRARY[type];
  const p = at || spawnPoint();
  addObject({ id: newId(), label: type, kind: 'primitive', type, color: def.color, size: [...def.size],
    position: [p.x, 0, p.z], rotationY: 0, scale: [1, 1, 1], mount: 'floor' });
}

for (const [type, def] of Object.entries(LIBRARY)) {
  const b = document.createElement('button');
  b.innerHTML = `<span>${def.icon}</span>${type}`;
  b.draggable = true;
  b.onclick = () => addPrimitive(type);
  b.ondragstart = (e) => e.dataTransfer.setData('text/room-item', type);
  $('library').appendChild(b);
}
renderer.domElement.addEventListener('dragover', (e) => e.preventDefault());
renderer.domElement.addEventListener('drop', (e) => {
  e.preventDefault();
  const type = e.dataTransfer.getData('text/room-item');
  if (type) addPrimitive(type, floorPoint(e) || undefined);
});

$('itemPhoto').onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || !project) return toast('Open or build a room first');
  const fd = new FormData();
  fd.append('file', file);
  fd.append('prompt', $('itemPrompt').value);
  toast('Segmenting + reconstructing item…', 60000);
  try {
    const item = await api(`/api/projects/${project.id}/items`, { method: 'POST', body: fd });
    const p = spawnPoint();
    item.position = [p.x, 0, p.z];
    addObject(item);
    toast(`Added ${item.label} (${item.kind})`);
  } catch (err) { toast(`Failed: ${err.message}`); }
};

$('importGlb').onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || !project) return toast('Open or build a room first');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const { asset, kind } = await api(`/api/projects/${project.id}/assets`, { method: 'POST', body: fd });
    const p = spawnPoint();
    addObject({ id: newId(), label: file.name.replace(/\.\w+$/, ''), kind, asset, size: null,
      position: [p.x, 0, p.z], rotationY: 0, scale: [1, 1, 1], mount: 'floor' });
  } catch (err) { toast(`Failed: ${err.message}`); }
};

// ---------------------------------------------------------------- room panel
for (const [id, key] of [['rW', 'width'], ['rD', 'depth'], ['rH', 'height']]) {
  $(id).onchange = (e) => { project.room[key] = Math.max(1, +e.target.value); buildRoom(); markDirty(); };
}
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
  const span = Math.max(r.width, r.depth);
  if (kind === 'top') { camera.position.set(0, span * 1.6, 0.001); orbit.target.set(0, 0, 0); }
  else if (kind === 'eye') { camera.position.set(0, r.cameraHeight, 0.01); orbit.target.set(0, r.cameraHeight, -0.5); }
  else { camera.position.set(span * 0.9, span * 1.1, span * 1.2); orbit.target.set(0, 0.6, 0); }
  orbit.update();
}
$('topView').onclick = () => setView('top');
$('persView').onclick = () => setView('pers');
$('eyeView').onclick = () => setView('eye');

function undo() {
  if (!project || !undoStack.length) return;
  project.objects = JSON.parse(undoStack.pop());
  rebuildObjects();
  markDirty();
}
$('undo').onclick = undo;
$('save').onclick = () => save().catch((e) => toast(e.message));

addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); undo(); }
  else if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save(); }
  else if ((e.ctrlKey || e.metaKey) && k === 'd') { e.preventDefault(); duplicate(); }
  else if (k === 'w') setMode('translate');
  else if (k === 'e') setMode('rotate');
  else if (k === 'r') setMode('scale');
  else if (k === 't') setView('top');
  else if (k === 'p') setView('pers');
  else if (k === 'v') setView('eye');
  else if (k === 'escape') select(null);
  else if ((k === 'delete' || k === 'backspace') && selected) removeObject(selected);
});

$('exportGlb').onclick = () => {
  if (!project) return;
  const out = new THREE.Scene();
  const r = project.room;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(r.width, r.depth), new THREE.MeshStandardMaterial({ color: r.floorColor }));
  floor.rotation.x = -Math.PI / 2;
  floor.name = 'floor';
  out.add(floor);
  const wm = new THREE.MeshStandardMaterial({ color: r.wallColor, side: THREE.DoubleSide });
  for (const [w, x, z, ry] of [[r.width, 0, -r.depth / 2, 0], [r.width, 0, r.depth / 2, Math.PI],
    [r.depth, -r.width / 2, 0, Math.PI / 2], [r.depth, r.width / 2, 0, -Math.PI / 2]]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(w, r.height), wm);
    wall.position.set(x, r.height / 2, z);
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

function renderThumbs() {
  const box = $('thumbs');
  box.innerHTML = '';
  const multi = $('mode').value !== '360';
  pending.forEach((p, i) => {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = p.url;
    fig.appendChild(img);
    if (multi && pending.length > 1) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.value = p.yaw;
      inp.title = 'Direction this photo faces, degrees clockwise from the first photo';
      inp.oninput = () => (p.yaw = +inp.value);
      fig.appendChild(inp);
    }
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

$('files').onchange = (e) => setFiles(e.target.files);
$('mode').onchange = renderThumbs;
const drop = $('drop');
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); setFiles(e.dataTransfer.files); };

$('build').onclick = async () => {
  const fd = new FormData();
  pending.forEach((p) => fd.append('files', p.file));
  fd.append('name', pending[0]?.file.name.replace(/\.\w+$/, '') || 'My room');
  fd.append('mode', $('mode').value);
  fd.append('yaws', JSON.stringify(pending.map((p) => p.yaw)));
  fd.append('camera_height', $('camH').value);
  fd.append('room_width', $('roomW').value);
  fd.append('room_depth', $('roomD').value);
  fd.append('prompts', $('prompts').value);
  fd.append('reconstruct', $('recon').checked);
  $('build').disabled = true;
  try {
    const { id } = await api('/api/projects', { method: 'POST', body: fd });
    await poll(id);
    await openProject(id);
  } catch (err) { toast(`Build failed: ${err.message}`, 6000); }
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
  undoStack.length = 0;
  selected = null;
  $('sceneName').value = project.name || '';
  $('sceneName').disabled = false;
  $('hint').classList.add('hidden');
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
  $('caps').textContent = `segmenter: ${m.segmenter || '?'} · 3D: ${m.reconstructor || 'photo cutouts'}`;
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

refreshProjects().then(() => {
  const id = location.hash.slice(1);
  if (id) openProject(id).catch(() => {});
});
