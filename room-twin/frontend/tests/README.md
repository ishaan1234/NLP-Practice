# Smart Room Layout Suggestions

From `room-twin`, run with Node.js 22 or newer (no npm install needed):

```powershell
node frontend/tests/fit-checker.test.mjs
node frontend/tests/layout-optimizer.test.mjs
node --experimental-default-type=module frontend/tests/layout-preview.test.mjs
```

The preview test uses a minimal DOM/Three adapter to exercise the real controller.
It verifies preview isolation, Cancel, Apply, the original-data history checkpoint,
and invalidation after an external change. It does not replace
testing WebGL, browser events, or backend saving in a real browser.

## Local demo

Use the project's normal setup, from `room-twin`:

```powershell
py launch.py --no-gpu
```

Open the URL printed by the launcher (normally http://localhost:8000). Start a
blank room or open an existing project, add furniture, and click **Optimize My
Room** in the Room panel. Overlap two items using the X/Z inspector to make a
clear demo case. Green outlines identify changed items in the preview. Apply
keeps the suggestion and uses the existing autosave; Cancel/Escape discards it.

## How it works

The checker uses SAT on scaled, rotated rectangular footprints and checks height
overlap. Edge touching is allowed within a small numerical tolerance. All four
footprint corners and vertical extents must fit inside the room, including an
offset room center. These are conservative furniture envelopes, not mesh tests.

The optimizer searches a coarse grid plus wall-clamped positions, trying the
current rotation and quarter turns in several placement orders. Candidate ranking
prefers fewer narrow gaps, less movement and space near walls. Complete layouts
are checked again and scored as:

`-10000 * (collision pairs + out-of-bounds items) + 10 * connected walking area - 2 * narrow pairs - 0.4 * movement`

Walking area is the largest connected component of a grid where a 0.6 m square,
1.8 m tall probe fits. Narrow pairs use a 0.6 m SAT gap approximation; movement
counts travel in meters plus a small rotation cost. This is a comfort heuristic,
not a navigation or accessibility guarantee. Baseline and candidate use the same
grid and score. Only valid layouts scoring strictly higher are offered.

Locked furniture, wall-mounted items, elevated items, and furniture supporting
elevated objects stay fixed. Hidden items, rugs and wall openings do not block
placement. Doors/windows remain unchanged; door swing and entry routes are not
modeled. Furniture sizes, colors, assets and elevations are preserved.

The search yields every 60 candidates, stops after roughly 1.8 seconds or eight
attempts, and supports at most 45 scene objects. The time limit is cooperative,
not a hard real-time guarantee. A coarse search may miss a feasible improvement;
the UI reports that without changing the room. Unknown dimensions prevent search.

Preview changes only rendered transforms. Editing controls are temporarily
disabled to prevent writing preview transforms back to scene data. Saving an
already-pending edit still saves the original project. External scene changes
invalidate the preview. Applying creates one existing Undo/Redo checkpoint.

## Manual browser checklist

- [ ] Overlap two items; preview reports reduced collisions and highlights moved items.
- [ ] Cancel and Escape restore positions, selection gizmo, and controls.
- [ ] Preview for longer than the autosave delay; reload and confirm the original layout remains.
- [ ] Apply; Undo restores the original; Redo restores the suggestion; save/reload retains it.
- [ ] Locked furniture stays fixed; all-locked and oversized rooms explain no suggestion.
- [ ] Rotate and resize furniture; test offset-center imported rooms and boundary crossings.
- [ ] Hidden objects, rugs, wall-mounted items, and elevated decor follow the documented rules.
- [ ] A table supporting elevated decor stays fixed.
- [ ] During preview, shortcuts, paste, and drops cannot edit furniture; Cancel remains responsive.
- [ ] Before and after preview, drag, rotate, resize, add/delete, camera controls, and autosave work normally.
- [ ] Check light/dark themes, small windows, keyboard focus, and loading model assets.
