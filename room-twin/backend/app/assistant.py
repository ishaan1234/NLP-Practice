"""Natural-language editing: "I want a table here", "maybe a vase", "add two cats on the rug",
"make the sofa 20 cm wider".

One Claude call per request. The editing operations are tools; Claude answers with tool calls
that the browser applies as one undoable step. The room itself (walls, size) is deliberately
not exposed as a tool.

New pieces come from, best first:
  1. Poly Haven - curated CC0 models at real-world scale (catalog.py)
  2. Objaverse-LVIS - 46k models in 1,156 everyday categories, cats to pianos (objaverse.py)
  3. generation - text -> picture -> cut-out / SAM 3D mesh, for anything else (generate.py)
  4. the procedural library pieces
Without Anthropic credentials a small rule-based parser covers the common phrasings.
"""
from __future__ import annotations

import json
import logging
import os
import re

from . import catalog, generate, objaverse

log = logging.getLogger(__name__)

MODEL = os.environ.get("ROOM_TWIN_AI_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ROOM_TWIN_AI_EFFORT", "medium")
LIBRARY_TYPES = ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "lamp", "plant", "tv", "rug",
                 "door", "window", "box"]

NUM = {"type": ["number", "null"]}
SOURCE = {
    "model_id": {"type": ["string", "null"], "description": "Poly Haven catalog id (best quality), or null"},
    "objaverse_category": {"type": ["string", "null"],
                           "description": "exact Objaverse category name, when no catalog model fits, or null"},
    "variant": {"type": ["integer", "null"], "description": "0 = default model; 1, 2... = another one"},
    "generate": {"type": ["string", "null"],
                 "description": "short visual description to create the object from scratch when neither "
                                "catalog has it, e.g. 'a ginger cat curled up asleep'; or null"},
    "library_type": {"type": ["string", "null"], "enum": [*LIBRARY_TYPES, None],
                     "description": "simple built-in piece, last resort"},
}
TOOLS = [
    {
        "name": "add_object",
        "description": "Place a new piece. Fill exactly one source: model_id, else objaverse_category, else "
                       "generate, else library_type. Sizes are real-world metres: always give them for "
                       "Objaverse, generated and library pieces (their files have no reliable scale); for a "
                       "Poly Haven model give them only if the user asked for a size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "short name shown to the user, e.g. 'ginger cat'"},
                **SOURCE,
                "x": {"type": "number", "description": "footprint centre, metres (room coordinates)"},
                "z": {"type": "number"},
                "elevation": {**NUM, "description": "base height above the floor (on a table, on a wall); "
                                                    "null = on the floor"},
                "rotation_deg": {**NUM, "description": "yaw, 0 faces +z; null = face the room centre"},
                "width": NUM, "height": NUM, "depth": NUM,
                "color": {"type": ["string", "null"], "description": "#rrggbb tint, or null"},
            },
            "required": ["label", *SOURCE, "x", "z", "elevation", "rotation_deg", "width", "height", "depth",
                         "color"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_model",
        "description": "Swap an object's 3D model for another (e.g. 'show me a different cat', 'a more modern "
                       "lamp'), keeping its place and size. Fill one source as for add_object.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, **SOURCE},
            "required": ["id", *SOURCE],
            "additionalProperties": False,
        },
    },
    {
        "name": "resize_object",
        "description": "Change an existing object's size in metres. Null keeps that dimension; for "
                       "'bigger'/'smaller' scale all three proportionally.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "width": NUM, "height": NUM, "depth": NUM},
            "required": ["id", "width", "height", "depth"],
            "additionalProperties": False,
        },
    },
    {
        "name": "move_object",
        "description": "Move an object to a floor position (metres). elevation null keeps its height.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "x": {"type": "number"}, "z": {"type": "number"},
                           "elevation": NUM},
            "required": ["id", "x", "z", "elevation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rotate_object",
        "description": "Set an object's yaw in degrees (absolute; 0 faces +z).",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "rotation_deg": {"type": "number"}},
            "required": ["id", "rotation_deg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recolor_object",
        "description": "Tint an object (#rrggbb).",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "color": {"type": "string"}},
            "required": ["id", "color"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_object",
        "description": "Remove an object.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = """You arrange things in a 3D twin of the user's room, using the tools. The user can ask for
anything - furniture, a vase, a doghouse, their cats - and you find or make it.

Coordinates: metres, Y up, floor y = 0. Positions are the centre of an object's footprint; sizes are
width (x before rotation) x height x depth (z); rotation_deg is the yaw, 0 faces +z. The scene message
gives the room bounds, every object (id, label, position, size, rotation), the selected object, and
"here": the floor point the user last clicked (if any).

- "here"/"there" is the clicked point; "it"/"this" is the selected object, else the one added or
  changed most recently.
- Sources for new pieces, best first: a Poly Haven model_id from the catalog below; else the closest
  objaverse_category (pets, toys, instruments, appliances...); else generate with a short visual
  description; library_type only as a last resort. Give realistic real-world sizes for everything but
  Poly Haven models (a house cat is ~0.45 m long and 0.25 m tall; a vase 0.3 m tall).
- Put things where they belong: small objects on a table, shelf or the floor (elevation = the top of
  what they stand on), pictures on walls, furniture back against a wall when that is normal. Keep pieces
  inside the room and clear of each other unless asked; face seating towards what it is used with.
- Several of something = several add_object calls with different positions.
- Never try to change the room itself (walls, size); say so if asked.
- If the request is ambiguous in a way that matters (which of two sofas?), ask one short question
  instead of guessing.
- After the tool calls, reply in one short sentence saying what you did."""


def _catalog_text() -> str:
    ph = [f"{i['id']}: {i['name']} [{', '.join(i['categories'][:3])}] ~{' x '.join(str(v) for v in i['size'])} m"
          for i in catalog.catalog()]
    cats = objaverse.categories()
    return ("Poly Haven catalog (id: name [categories] ~bounding box, axis order varies):\n" + "\n".join(ph)
            + "\n\nObjaverse categories (use the exact name):\n" + ", ".join(cats))


def _client():
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
    except Exception:
        return None
    # credentials resolve from the environment (API key, auth token or an `ant auth login` profile)
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or getattr(client, "api_key", None) or getattr(client, "auth_token", None)):
        return None
    return client


def ask(message: str, scene: dict, history: list[dict], make_item=None) -> dict:
    """make_item(description) -> item dict, for generated pieces (needs the project)."""
    client = _client()
    if client is None:
        result = rule_based(message, scene, make_item)
        result["engine"] = "offline rules (set ANTHROPIC_API_KEY for the AI assistant)"
        return result
    import anthropic

    messages = []
    for turn in history[-6:]:  # plain-text memory of recent requests and replies
        if turn.get("user") and turn.get("reply"):
            messages += [{"role": "user", "content": str(turn["user"])},
                         {"role": "assistant", "content": str(turn["reply"])}]
    messages.append({"role": "user", "content": [
        {"type": "text", "text": "Scene:\n" + json.dumps(scene, separators=(",", ":"))},
        {"type": "text", "text": message},
    ]})
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a declined request is re-run on Anthropic's recommended fallback model
            output_config={"effort": EFFORT},
            # the instructions + catalogs are the same on every request: cache them
            system=[{"type": "text", "text": SYSTEM},
                    {"type": "text", "text": _catalog_text(), "cache_control": {"type": "ephemeral"}}],
            tools=[{**t, "strict": True} for t in TOOLS],
            messages=messages,
        )
    except anthropic.AuthenticationError:
        result = rule_based(message, scene, make_item)
        result["engine"] = "offline rules (the Anthropic API key was rejected)"
        return result
    except anthropic.RateLimitError:
        return {"reply": "The AI is rate limited right now; try again in a minute.", "actions": []}
    except anthropic.APIConnectionError:
        result = rule_based(message, scene, make_item)
        result["engine"] = "offline rules (couldn't reach the Anthropic API)"
        return result
    except anthropic.APIStatusError as e:
        log.exception("assistant request failed")
        return {"reply": f"The AI request failed ({e.status_code}).", "actions": []}

    if response.stop_reason == "refusal":
        return {"reply": "The assistant declined that request.", "actions": []}
    actions, texts = [], []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            actions.append({"op": block.name, **dict(block.input)})
    if response.stop_reason == "max_tokens" and not actions:
        return {"reply": "That request was too long to finish; try asking for less at once.", "actions": []}
    return {"reply": " ".join(t.strip() for t in texts if t.strip()) or "Done.",
            "actions": resolve_models(actions, make_item), "engine": MODEL}


def resolve_models(actions: list[dict], make_item=None) -> list[dict]:
    """Turn each add/replace source into a concrete asset, falling through the sources in order."""
    for a in actions:
        if a["op"] not in ("add_object", "replace_model"):
            continue
        a["source_used"] = None
        if a.get("model_id"):
            try:
                info = catalog.fetch(a["model_id"])
                a.update(asset=info["asset"], license=info["license"], source_used="polyhaven")
            except Exception as e:
                log.warning("Poly Haven model %s unavailable: %s", a["model_id"], e)
        if not a["source_used"] and a.get("objaverse_category"):
            try:
                info = objaverse.fetch(a["objaverse_category"], int(a.get("variant") or 0))
                a.update(asset=info["asset"], license=info["license"], source_used="objaverse",
                         credit_url=info.get("source"))
            except Exception as e:
                log.warning("Objaverse %s unavailable: %s", a["objaverse_category"], e)
        if not a["source_used"] and a.get("generate") and make_item and generate.available():
            try:
                a["item"] = make_item(a["generate"])
                a["source_used"] = "generated"
            except Exception as e:
                log.warning("generation failed: %s", e)
        if not a["source_used"]:
            # nothing found: try the label against Objaverse, then a library piece
            guess = _objaverse_guess(a.get("label", "") + " " + (a.get("generate") or ""))
            if guess:
                try:
                    info = objaverse.fetch(guess, int(a.get("variant") or 0))
                    a.update(asset=info["asset"], license=info["license"], source_used="objaverse",
                             credit_url=info.get("source"))
                except Exception:
                    pass
        if not a["source_used"]:
            a["library_type"] = a.get("library_type") if a.get("library_type") in LIBRARY_TYPES \
                else _guess_type(a.get("label", "")) or "box"
            a["source_used"] = "library"
    return actions


def _objaverse_guess(text: str) -> str | None:
    cats = set(objaverse.categories())
    words = re.findall(r"[a-z]+", text.lower())
    for n in (2, 1):  # "teddy bear" before "bear"
        for i in range(len(words) - n + 1):
            for cand in ("_".join(words[i:i + n]), "_".join(words[i:i + n]).rstrip("s")):
                if cand in cats:
                    return cand
    return None


# ---------------------------------------------------------------- offline fallback

def _guess_type(text: str) -> str | None:
    text = text.lower()
    aliases = {"couch": "sofa", "armchair": "chair", "stool": "chair", "bookshelf": "shelf", "bookcase": "shelf",
               "dresser": "cabinet", "wardrobe": "cabinet", "nightstand": "cabinet", "television": "tv",
               "carpet": "rug", "light": "lamp", "flower": "plant"}
    for word in re.findall(r"[a-z]+", text):
        if word in LIBRARY_TYPES:
            return word
        if word.rstrip("s") in LIBRARY_TYPES:
            return word.rstrip("s")
        if word in aliases:
            return aliases[word]
    return None


def _target(scene: dict) -> dict | None:
    objs = scene.get("objects", [])
    sel = scene.get("selected") or scene.get("last_changed")
    return next((o for o in objs if o["id"] == sel), None) or (objs[-1] if objs else None)


def _find(text: str, scene: dict) -> dict | None:
    if re.search(r"\b(it|this|that|selected)\b", text):
        return _target(scene)
    for o in scene.get("objects", []):
        if o["label"].lower() in text:
            return o
    return _target(scene)


def rule_based(message: str, scene: dict, make_item=None) -> dict:
    """add/put/want <thing> [here]; bigger/smaller/N%; N cm wider/taller/deeper; turn N; delete."""
    text = message.lower().strip()
    here = scene.get("here") or scene.get("view_center") or [0, 0]
    if re.search(r"\b(add|put|place|want|need|spawn|drop|bring|maybe|get me)\b", text):
        thing = re.sub(r"\b(i|want|need|add|put|place|spawn|drop|bring|get|me|maybe|even|a|an|the|some|here|there|"
                       r"please|in|on|of|models?|pet|could|you|can)\b", " ", text).strip() or "box"
        hits = catalog.search(thing, 1)
        exact = hits and any(w in hits[0]["name"].lower() for w in thing.split())
        action = {"op": "add_object", "label": thing, "model_id": hits[0]["id"] if exact else None,
                  "objaverse_category": _objaverse_guess(thing), "variant": None,
                  "generate": thing, "library_type": _guess_type(thing), "x": here[0], "z": here[1],
                  "elevation": None, "rotation_deg": None, "width": None, "height": None, "depth": None,
                  "color": None}
        if not exact:
            action["height"] = 0.5  # unknown real size: a sensible default the user can resize
        return {"reply": f"Added {thing}.", "actions": resolve_models([action], make_item)}
    obj = _find(text, scene)
    if obj is None:
        return {"reply": "There's nothing to change yet - add something first.", "actions": []}
    w, h, d = obj["size"]
    if re.search(r"\b(delete|remove|get rid of)\b", text):
        return {"reply": f"Removed the {obj['label']}.", "actions": [{"op": "delete_object", "id": obj["id"]}]}
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(cm|m)\b.*?\b(wider|narrower|taller|shorter|deeper|shallower|longer)", text)
    if m:
        amount = float(m.group(1)) / (100 if m.group(2) == "cm" else 1)
        word = m.group(3)
        sign = -1 if word in ("narrower", "shorter", "shallower") else 1
        key = {"wider": "width", "narrower": "width", "longer": "width", "taller": "height", "shorter": "height",
               "deeper": "depth", "shallower": "depth"}[word]
        dims = {"width": None, "height": None, "depth": None}
        dims[key] = round(max(0.05, {"width": w, "height": h, "depth": d}[key] + sign * amount), 3)
        return {"reply": f"Made the {obj['label']} {word}.", "actions": [{"op": "resize_object", "id": obj["id"], **dims}]}
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    factor = None
    if re.search(r"\b(bigger|larger|grow)\b", text):
        factor = 1 + (float(m.group(1)) / 100 if m else 0.2)
    elif re.search(r"\b(smaller|shrink)\b", text):
        factor = 1 - (float(m.group(1)) / 100 if m else 0.2)
    if factor:
        return {"reply": f"Resized the {obj['label']}.", "actions": [
            {"op": "resize_object", "id": obj["id"], "width": round(w * factor, 3),
             "height": round(h * factor, 3), "depth": round(d * factor, 3)}]}
    m = re.search(r"(?:turn|rotate|spin)\D*(-?\d+)", text)
    if m:
        return {"reply": f"Turned the {obj['label']}.", "actions": [
            {"op": "rotate_object", "id": obj["id"], "rotation_deg": obj.get("rotation_deg", 0) + float(m.group(1))}]}
    if re.search(r"\b(another|different|other one)\b", text):
        action = {"op": "replace_model", "id": obj["id"], "model_id": None,
                  "objaverse_category": _objaverse_guess(obj["label"]), "variant": int(obj.get("variant") or 0) + 1,
                  "generate": None, "library_type": None}
        return {"reply": f"Here's a different {obj['label']}.", "actions": resolve_models([action], make_item)}
    if re.search(r"\b(move|over here|here)\b", text) and scene.get("here"):
        return {"reply": f"Moved the {obj['label']}.", "actions": [
            {"op": "move_object", "id": obj["id"], "x": here[0], "z": here[1], "elevation": None}]}
    return {"reply": "I can add things, resize, move, turn or remove them - e.g. 'add a vase here' or "
                     "'make the sofa 20 cm wider'. Set ANTHROPIC_API_KEY for free-form requests.", "actions": []}
