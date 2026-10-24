import assert from 'node:assert/strict';
import { setupSuggestions } from '../layout-suggestions.js';

// Exercise the real UI controller with a minimal DOM/Three adapter (no WebGL required).
const elements = new Map();
for (const id of ['optimizeRoom', 'layoutSuggestion', 'layoutMessage', 'layoutApply', 'layoutCancel', 'viewport', 'canvas', 'header', 'aside']) {
  const classes = new Set();
  elements.set(id, { id, style: {}, inert: false, textContent: '', focus() {}, contains(target) { return target === this; },
    classList: { add(c) { classes.add(c); }, remove(c) { classes.delete(c); }, toggle(c, value) { if (value) classes.add(c); else classes.delete(c); } } });
}
const listeners = new Map();
globalThis.document = { getElementById: (id) => elements.get(id), querySelector: () => elements.get('canvas'),
  querySelectorAll: () => [elements.get('header'), elements.get('aside')], addEventListener: (name, fn) => listeners.set(name, fn) };
class Group {
  children = [];
  add(n) { this.children.push(n); }
  remove(n) { this.children = this.children.filter((v) => v !== n); }
}
class BoxHelper { geometry = { dispose() {} }; material = { dispose() {} }; }
const objects = ['a', 'b'].map((id) => ({ id, position: [0, 0, 0], size: [1, 1, 1], scale: [1, 1, 1], rotationY: 0, mount: 'floor' }));
let project = { objects, room: { width: 5, depth: 4, height: 3 } };
const saved = JSON.stringify(project), nodes = new Map();
for (const o of objects) nodes.set(o.id, { position: { value: [...o.position], fromArray(a) { this.value = [...a]; } }, rotation: { y: 0 } });
let undo = [], dirty = 0, restored = 0;
const rebuild = () => { for (const o of project.objects) { nodes.get(o.id).position.fromArray(o.position); nodes.get(o.id).rotation.y = o.rotationY; } };
const tick = setupSuggestions({ THREE: { Group, BoxHelper }, getProject: () => project, nodes, scene: new Group(), transform: { detach() {} },
  checkpoint: () => { undo.push(JSON.stringify(project.objects)); }, rebuildObjects: rebuild,
  markDirty: () => dirty++, restoreSelection: () => restored++ });
await elements.get('optimizeRoom').onclick();
assert.equal(JSON.stringify(project), saved, 'Preview does not change saved project');
assert.notEqual(JSON.stringify([...nodes.values()].map((n) => n.position.value)), JSON.stringify(objects.map((o) => o.position)));
assert(elements.get('layoutMessage').textContent.includes('Reduced furniture collisions'));
elements.get('layoutCancel').onclick();
assert.equal(JSON.stringify(project), saved);
assert.equal(dirty, 0, 'Cancel never autosaves');
assert.equal(JSON.stringify([...nodes.values()].map((n) => n.position.value)), JSON.stringify(objects.map((o) => o.position)));
assert(restored > 0, 'Selection restored after Cancel');
await elements.get('optimizeRoom').onclick();
elements.get('layoutApply').onclick();
assert.equal(undo.length, 1, 'Apply is one history checkpoint');
assert.equal(dirty, 1);
const applied = JSON.stringify(project.objects);
assert.notEqual(applied, undo[0]);
assert.equal(undo[0], JSON.stringify(JSON.parse(saved).objects), 'History receives the original data before applying');
await elements.get('optimizeRoom').onclick();
project.objects[0].position[0] += 0.2;
tick();
assert.equal(elements.get('header').inert, false, 'External edits invalidate preview');
assert.equal(nodes.get('a').position.value[0], project.objects[0].position[0]);
assert.equal(dirty, 1);
console.log('Preview, Cancel, Apply, history checkpoint and invalidation checks passed');
