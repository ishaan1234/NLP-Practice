import { optimize, metrics, transforms } from './layout-optimizer.mjs';
import { inspect } from './fit-checker.mjs';

export function setupSuggestions({ THREE, getProject, nodes, scene, transform, checkpoint, rebuildObjects, markDirty, restoreSelection }) {
  const button = document.getElementById('optimizeRoom');
  const panel = document.getElementById('layoutSuggestion');
  const message = document.getElementById('layoutMessage');
  const apply = document.getElementById('layoutApply');
  const cancel = document.getElementById('layoutCancel');
  const outlines = new THREE.Group(); scene.add(outlines);
  let active = false, generation = 0, snapshot = '', owner = null, suggestion = null;
  const signature = () => JSON.stringify(getProject());
  function clearOutlines() {
    for (const child of [...outlines.children]) { child.geometry.dispose(); child.material.dispose(); outlines.remove(child); }
  }
  function setBusy(value) {
    document.querySelectorAll('header, aside').forEach((el) => { el.inert = value; });
    document.querySelector('#viewport canvas').style.pointerEvents = value ? 'none' : '';
    document.getElementById('viewport').classList.toggle('layout-preview', value);
  }
  function finish() {
    generation++; active = false; suggestion = null; setBusy(false); clearOutlines();
    const project = getProject();
    if (project) for (const o of project.objects) {
      const n = nodes.get(o.id);
      if (n) { n.position.fromArray(o.position); n.rotation.y = o.rotationY || 0; }
    }
    panel.classList.add('hidden');
    restoreSelection();
    button.focus();
  }
  cancel.onclick = finish;
  for (const name of ['pointerdown', 'drop', 'paste', 'dragover']) document.addEventListener(name, (event) => {
    if (active && !panel.contains(event.target)) { event.preventDefault(); event.stopImmediatePropagation(); }
  }, true);
  // Keyboard shortcuts must not write preview transforms back into project data.
  document.addEventListener('keydown', (event) => {
    if (!active) return;
    if (event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); finish(); }
    else if (!panel.contains(event.target)) { event.preventDefault(); event.stopImmediatePropagation(); }
    else event.stopImmediatePropagation();
  }, true);
  button.onclick = async () => {
    const project = getProject();
    if (!project || active) return;
    active = true; owner = project; snapshot = signature(); suggestion = null;
    const token = ++generation;
    transform.detach(); setBusy(true);
    panel.classList.remove('hidden'); apply.classList.add('hidden');
    cancel.textContent = 'Cancel'; message.textContent = 'Analyzing furniture and searching for a better arrangement…';
    cancel.focus();
    try {
      // Let the progress panel paint before starting geometry work.
      await new Promise((resolve) => setTimeout(resolve, 20));
      if (token !== generation) return;
      const found = await optimize(project.objects, project.room, { cancelled: () => token !== generation || signature() !== snapshot });
      if (token !== generation) return;
      if (signature() !== snapshot) { finish(); return; }
      if (!found.objects) { message.textContent = found.reason; cancel.textContent = 'Close'; return; }
      suggestion = found;
      for (const t of transforms(found.objects)) {
        const n = nodes.get(t.id); if (!n) continue;
        n.position.fromArray(t.position); n.rotation.y = t.rotationY;
        const before = project.objects.find((o) => o.id === t.id);
        if (Math.hypot(...t.position.map((v, i) => v - before.position[i])) > 0.001 || Math.abs(t.rotationY - (before.rotationY || 0)) > 0.001) {
          const outline = new THREE.BoxHelper(n, 0x5fb39b); outlines.add(outline);
        }
      }
      const improvements = [];
      if (found.result.collisions.length < found.baseline.collisions.length) improvements.push('Reduced furniture collisions');
      if (found.result.outside.length < found.baseline.outside.length) improvements.push('Brought furniture inside room boundaries');
      if (found.result.tight < found.baseline.tight) improvements.push('Reduced narrow furniture gaps');
      if (found.result.connected > found.baseline.connected + 0.01) improvements.push(`Connected walking space: ${found.baseline.connected.toFixed(1)} → ${found.result.connected.toFixed(1)} m²`);
      improvements.push(`Moved ${found.result.moved} furniture item${found.result.moved === 1 ? '' : 's'}`);
      message.textContent = `Preview • ${improvements.join(' • ')}. Green outlines show moved items. Apply to keep this layout; Undo restores it.`;
      apply.classList.remove('hidden');
    } catch (error) {
      if (token === generation) { message.textContent = `Could not generate a suggestion: ${error.message}`; cancel.textContent = 'Close'; }
    }
  };
  apply.onclick = () => {
    if (!suggestion || owner !== getProject() || signature() !== snapshot) { finish(); return; }
    const project = getProject(), checks = inspect(suggestion.objects, project.room);
    const score = metrics(suggestion.objects, project.room, project.objects);
    if (checks.unknown.length || checks.collisions.length || checks.outside.length || score.score <= suggestion.baseline.score + 0.01) { finish(); return; }
    const next = transforms(suggestion.objects);
    finish(); checkpoint();
    for (const t of next) Object.assign(project.objects.find((o) => o.id === t.id), { position: t.position, rotationY: t.rotationY });
    rebuildObjects(); markDirty();
  };
  return () => {
    button.disabled = !getProject();
    if (active && (owner !== getProject() || snapshot !== signature())) finish();
  };
}
