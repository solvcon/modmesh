# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The single command schema for driving a ``World`` from an AI agent.

The schema is expressed in JSON Schema (draft 2020-12) and authored here as
Python dicts, so it is one machine-readable contract that every front-end
rides on: native tool-calling (Opus), the constrained text protocol (the
small model), and the MCP server all produce and consume these same commands.
There is no second schema.

A command is a JSON object ``{"op": <name>, ...args}``. ``COMMAND_SCHEMAS``
maps each op to its JSON Schema; ``SCHEMA`` is the combined document that
accepts any one command. ``validate_command`` validates an instance with the
``jsonschema`` library and raises ``CommandError`` on any violation.

The contract is symmetric: every op also declares the shape of the value it
returns. ``RESULT_SCHEMAS`` maps each op to a closed JSON Schema for its
result, so an MCP server can publish an ``outputSchema`` and a harness can
rely on exactly the fields named there. ``render_png`` returns a
transport-ready image (base64 PNG plus ``mime_type``), never raw bytes, so
the result is JSON-serializable across any transport.

Every command is tagged with a CRUD ``category`` so the vocabulary covers the
full lifecycle of a shape: ``create`` (``add_*``), ``read`` (``get_shape`` and
the world-wide queries), ``update`` (``translate_shape``, the only in-place
edit the ``World`` backend supports), and ``delete`` (``remove_shape``,
``clear``). ``log`` is the lone non-CRUD, bookkeeping op.

Coordinates are world coordinates in the ``World``'s native units, math
convention with +Y pointing up.
"""

import copy
from collections import namedtuple

import jsonschema


class CommandError(ValueError):
    """A command failed schema validation or could not be applied."""


# Reusable JSON Schema fragments. These are plain JSON Schema objects shared
# by reference across command schemas; treat them as immutable.
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
INTEGER = {"type": "integer"}
POSITIVE_INT = {"type": "integer", "exclusiveMinimum": 0}
BOOLEAN = {"type": "boolean"}
STRING = {"type": "string"}
# A 2D or 3D point; a missing z is treated as 0 when the command is applied.
POINT = {"type": "array", "items": {"type": "number"},
         "minItems": 2, "maxItems": 3}
# A 2D view transform (math-convention +Y up). Mirrors ViewTransform2dFp64.
VIEW = {"type": "object",
        "properties": {"pan_x": NUMBER, "pan_y": NUMBER, "zoom": POSITIVE},
        "additionalProperties": False}


# Builders for argument properties that carry a human description. Models call
# tools far more reliably when each parameter is described, so the catalog
# below uses these instead of the bare fragments above wherever a word of
# context helps. The fragments stay bare for use inside composite schemas.
def _num(description):
    """A number property with a human description."""
    return {"type": "number", "description": description}


def _pos(description):
    """A strictly-positive number property with a description."""
    return {"type": "number", "exclusiveMinimum": 0,
            "description": description}


def _int(description):
    """An integer property with a human description."""
    return {"type": "integer", "description": description}


def _point(description):
    """A 2D or 3D point property with a human description."""
    return {"type": "array", "items": {"type": "number"},
            "minItems": 2, "maxItems": 3, "description": description}


# Output-side fragments mirroring the JSON the ``World`` serializes. These feed
# the per-op result schemas so a consumer knows the exact return shape.
_BBOX = {"type": "array", "items": NUMBER, "minItems": 4, "maxItems": 4,
         "description": "Axis-aligned bounds [min_x, min_y, max_x, max_y]."}
_SEG_LIST = {"type": "array",
             "items": {"type": "array", "items": NUMBER,
                       "minItems": 4, "maxItems": 4},
             "description": "Line segments, each [x0, y0, x1, y1]."}
_CURVE_LIST = {"type": "array",
               "items": {"type": "array",
                         "items": {"type": "array", "items": NUMBER,
                                   "minItems": 2, "maxItems": 2},
                         "minItems": 4, "maxItems": 4},
               "description": "Cubic Beziers, each four [x, y] points."}
_POINT_LIST = {"type": "array",
               "items": {"type": "array", "items": NUMBER,
                         "minItems": 2, "maxItems": 2},
               "description": "Free points, each [x, y]."}

# One serialized shape, as returned by ``get_shape`` and nested in the world
# state. Mirrors WorldShapeState in cpp/solvcon/universe/World.hpp.
SHAPE = {"type": "object",
         "description": "One shape: id, type, bounds, and 2D geometry.",
         "properties": {
             "id": _int("Stable id assigned when the shape was created."),
             "type": {**STRING, "description": "Lower-case shape type name."},
             "bbox": _BBOX,
             "segments": _SEG_LIST,
             "curves": _CURVE_LIST},
         "required": ["id", "type", "bbox", "segments", "curves"],
         "additionalProperties": False}

# The whole visible world, as returned by ``describe_state``. Mirrors
# WorldState in cpp/solvcon/universe/World.hpp.
STATE = {"type": "object",
         "description": "The whole visible world: shapes plus bare geometry.",
         "properties": {
             "shapes": {"type": "array", "items": SHAPE,
                        "description": "Every live shape."},
             "segments": _SEG_LIST,
             "curves": _CURVE_LIST,
             "points": _POINT_LIST},
         "required": ["shapes", "segments", "curves", "points"],
         "additionalProperties": False}

IMAGE = {"type": "object",
         "description": "A rendered raster image, transport-ready.",
         "properties": {
             "data": {"type": "string", "contentEncoding": "base64",
                      "contentMediaType": "image/png",
                      "description": "Base64-encoded PNG bytes."},
             "mime_type": {"const": "image/png",
                           "description": "Media type of the encoded image."},
             "width": _int("Image width in pixels."),
             "height": _int("Image height in pixels.")},
         "required": ["data", "mime_type", "width", "height"],
         "additionalProperties": False}


# One command's variable parts. The op name and CRUD category are not stored
# here: they come from the keys of the table below, so neither is repeated.
# ``required`` defaults to None, meaning every property is required. ``result``
# names the command's output properties; None means it returns an empty object.
_Spec = namedtuple("_Spec",
                   ["description", "properties", "required", "result"],
                   defaults=(None, None))


# The command catalog, grouped by CRUD role. The outer key is the category,
# the inner key is the op, and the value spells out just that command's
# description and arguments. This grouping is the single source of truth for
# both COMMAND_SCHEMAS and CRUD_CATEGORIES.
_CATALOG = {
    "create": {
        "add_point": _Spec(
            "Add a free point at world (x, y, z); z is dropped in 2D.",
            {"x": _num("World x of the point."),
             "y": _num("World y of the point (+Y up)."),
             "z": {**NUMBER, "default": 0.0,
                   "description": "World z; ignored in 2D."}},
            ["x", "y"],
            {"npoint": _int("Total free points after the add.")}),
        "add_segment": _Spec(
            "Add a bare line segment between two world points.",
            {"p0": _point("Start point [x, y] or [x, y, z]."),
             "p1": _point("End point [x, y] or [x, y, z].")},
            result={"nsegment": _int("Total bare segments after the add.")}),
        "add_line": _Spec(
            "Add a line shape from (x0, y0) to (x1, y1) in world coordinates.",
            {"x0": _num("World x of the start point."),
             "y0": _num("World y of the start point."),
             "x1": _num("World x of the end point."),
             "y1": _num("World y of the end point.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_triangle": _Spec(
            "Add a triangle shape through its three corners (+Y up).",
            {"x0": _num("World x of corner 0."),
             "y0": _num("World y of corner 0."),
             "x1": _num("World x of corner 1."),
             "y1": _num("World y of corner 1."),
             "x2": _num("World x of corner 2."),
             "y2": _num("World y of corner 2.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_rectangle": _Spec(
            "Add an axis-aligned rectangle from lower-left to upper-right.",
            {"x_min": _num("Lower-left corner x."),
             "y_min": _num("Lower-left corner y."),
             "x_max": _num("Upper-right corner x."),
             "y_max": _num("Upper-right corner y.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_square": _Spec(
            "Add an axis-aligned square from its lower-left corner.",
            {"x_min": _num("Lower-left corner x."),
             "y_min": _num("Lower-left corner y."),
             "size": _pos("Edge length; must be positive.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_ellipse": _Spec(
            "Add an ellipse shape centered at (cx, cy).",
            {"cx": _num("Center x."), "cy": _num("Center y."),
             "rx": _pos("Semi-axis along x; must be positive."),
             "ry": _pos("Semi-axis along y; must be positive.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_circle": _Spec(
            "Add a circle shape centered at (cx, cy).",
            {"cx": _num("Center x."), "cy": _num("Center y."),
             "r": _pos("Radius; must be positive.")},
            result={"shape_id": _int("Id of the new shape.")}),
        "add_bezier": _Spec(
            "Add a bare cubic Bezier from four control points.",
            {"p0": _point("Start anchor [x, y] or [x, y, z]."),
             "p1": _point("First control point."),
             "p2": _point("Second control point."),
             "p3": _point("End anchor.")},
            result={"nbezier": _int("Total bare Beziers after the add.")}),
        "add_bezier_shape": _Spec(
            "Add a cubic Bezier shape from four control points.",
            {"p0": _point("Start anchor [x, y] or [x, y, z]."),
             "p1": _point("First control point."),
             "p2": _point("Second control point."),
             "p3": _point("End anchor.")},
            result={"shape_id": _int("Id of the new shape.")}),
    },
    "read": {
        "get_shape": _Spec(
            "Read one shape's id, type, bbox, and geometry by its id.",
            {"shape_id": _int("Id of the shape to read.")},
            result={"shape": SHAPE}),
        "shape_type_of": _Spec(
            "Name the type of the shape with the given id.",
            {"shape_id": _int("Id of the shape to inspect.")},
            result={"type": {**STRING,
                             "description": "Lower-case shape type name."}}),
        "nshape": _Spec(
            "Count the live shapes in the world.", {},
            result={"nshape": _int("Number of live shapes.")}),
        "query_visible": _Spec(
            "List the ids of shapes overlapping a query box.",
            {"min_x": _num("Query box lower-left x."),
             "min_y": _num("Query box lower-left y."),
             "max_x": _num("Query box upper-right x."),
             "max_y": _num("Query box upper-right y.")},
            result={"shape_ids": {
                "type": "array", "items": INTEGER,
                "description": "Ids of shapes overlapping the box."}}),
        "describe_state": _Spec(
            "Serialize the visible 2D geometry to a state object.",
            {"level": {"type": "string", "enum": ["basic"],
                       "default": "basic",
                       "description": "Level of detail; only 'basic'."}},
            [],
            {"state": STATE}),
        "render_png": _Spec(
            "Render the world to a PNG via the offscreen renderer.",
            {"width": {**POSITIVE_INT,
                       "description": "Image width in pixels."},
             "height": {**POSITIVE_INT,
                        "description": "Image height in pixels."},
             "view": {**VIEW, "default": {"pan_x": 0.0, "pan_y": 0.0,
                                          "zoom": 1.0},
                      "description": "2D view transform (+Y up)."},
             "antialiasing": {**BOOLEAN, "default": False,
                              "description": "Enable antialiased edges."}},
            ["width", "height"],
            {"image": IMAGE}),
    },
    "update": {
        "translate_shape": _Spec(
            "Translate the shape with the given id by (dx, dy).",
            {"shape_id": _int("Id of the shape to move."),
             "dx": _num("World displacement along x."),
             "dy": _num("World displacement along y.")}),
    },
    "delete": {
        "remove_shape": _Spec(
            "Remove the shape with the given id.",
            {"shape_id": _int("Id of the shape to remove.")}),
        "clear": _Spec("Remove every shape and bare primitive.", {}),
    },
    "log": {
        "log": _Spec(
            "Record a free-text note in the command log.",
            {"message": {**STRING,
                         "description": "Free-text note to record."}}),
    },
}

CRUD_CATEGORIES = tuple(_CATALOG)


def _build_schema(op, category, spec):
    """Assemble one command's JSON Schema from its catalog entry.

    The ``op`` field is pinned to a constant and always required; ``category``
    rides along as an annotation keyword that validators ignore. A ``None``
    ``spec.required`` means every argument is required.
    """
    required = (list(spec.properties) if spec.required is None
                else spec.required)
    return {
        "title": op,
        "description": spec.description,
        "category": category,
        "type": "object",
        "properties": {"op": {"const": op}, **spec.properties},
        "required": ["op"] + required,
        "additionalProperties": False,
    }


def _build_result(op, result):
    """Assemble one command's result JSON Schema from its catalog entry.

    A ``None`` ``spec.result`` means the command returns an empty object.
    Every declared result property is required and the object is closed, so a
    consumer can rely on exactly the fields named here.
    """
    properties = result or {}
    return {
        "title": f"{op}_result",
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


COMMAND_SCHEMAS = {
    op: _build_schema(op, category, spec)
    for category, specs in _CATALOG.items()
    for op, spec in specs.items()
}


# Result schemas, keyed by op like COMMAND_SCHEMAS. The output half of the
# contract: what each command returns on success.
RESULT_SCHEMAS = {
    op: _build_result(op, spec.result)
    for _category, specs in _CATALOG.items()
    for op, spec in specs.items()
}


# The combined document: a valid instance is exactly one command. Useful for
# consumers that want a single schema describing the whole command language.
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Agent Draw command",
    "description": "Any single command in the Agent Draw schema.",
    "oneOf": list(COMMAND_SCHEMAS.values()),
}

# One compiled validator per op, built once at import. Reused on every call so
# validation does not recompile the schema each time.
_VALIDATORS = {op: jsonschema.Draft202012Validator(schema)
               for op, schema in COMMAND_SCHEMAS.items()}

# Result validators, compiled once like the command validators above. Used by
# the executor's opt-in result checking to enforce the output contract.
_RESULT_VALIDATORS = {op: jsonschema.Draft202012Validator(schema)
                      for op, schema in RESULT_SCHEMAS.items()}


def validate_command(command):
    """Validate ``command`` against its op's JSON Schema.

    Returns the command unchanged on success; raises ``CommandError`` for a
    non-object, a missing/unknown ``op``, or any JSON Schema violation.
    """
    if not isinstance(command, dict):
        raise CommandError(
            f"command must be an object, got {type(command).__name__}")
    op = command.get("op")
    if not isinstance(op, str):
        raise CommandError("command is missing a string 'op' field")
    validator = _VALIDATORS.get(op)
    if validator is None:
        raise CommandError(
            f"unknown op '{op}'; valid ops: {sorted(COMMAND_SCHEMAS)}")
    try:
        validator.validate(command)
    except jsonschema.ValidationError as exc:
        raise CommandError(f"{op}: {exc.message}") from exc
    return command


def validate_result(op, value):
    """Validate a command's result against ``RESULT_SCHEMAS[op]``.

    Returns the value unchanged on success; raises ``CommandError`` for an
    unknown op or any violation of the declared output schema. The executor
    calls this when result checking is enabled so the published output
    contract is enforced at runtime, not only in tests.
    """
    validator = _RESULT_VALIDATORS.get(op)
    if validator is None:
        raise CommandError(
            f"unknown op '{op}'; valid ops: {sorted(RESULT_SCHEMAS)}")
    try:
        validator.validate(value)
    except jsonschema.ValidationError as exc:
        raise CommandError(f"{op} result: {exc.message}") from exc
    return value


def validate_script(commands):
    """Validate a list of commands, returning it unchanged on success."""
    if not isinstance(commands, list):
        raise CommandError("a script must be a list of commands")
    for command in commands:
        validate_command(command)
    return commands


def apply_defaults(command):
    """Return a copy of ``command`` with omitted optional args filled in.

    JSON Schema records defaults but validators do not apply them, so the
    executor calls this after validating to get a fully-populated command.
    For a dict-valued default (such as ``render_png``'s ``view``), sub-fields
    the caller omitted are filled from the default too, so a partial object
    reaches the handler complete.
    """
    schema = COMMAND_SCHEMAS[command["op"]]
    out = dict(command)
    for name, prop in schema["properties"].items():
        if not isinstance(prop, dict) or "default" not in prop:
            continue
        default = prop["default"]
        if name not in out:
            out[name] = copy.deepcopy(default)
        elif isinstance(default, dict) and isinstance(out[name], dict):
            out[name] = {**default, **out[name]}
    return out


def tool_definitions():
    """Describe every command as a name + input/output schema tool definition.

    Front-ends (MCP, native tool-calling) build their tool surface from this
    so the JSON Schema stays the single source of truth. The ``op`` field is
    dropped from each input schema because the tool name already carries it.
    ``outputSchema`` is the declared result shape, ready for an MCP server to
    publish verbatim.
    """
    tools = []
    for op, schema in COMMAND_SCHEMAS.items():
        properties = {name: prop for name, prop in schema["properties"].items()
                      if name != "op"}
        required = [name for name in schema["required"] if name != "op"]
        tools.append({
            "name": op,
            "category": schema["category"],
            "description": schema["description"],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "outputSchema": RESULT_SCHEMAS[op],
        })
    return tools


def commands_by_category():
    """Group op names by their CRUD ``category``."""
    return {category: list(specs.keys())
            for category, specs in _CATALOG.items()}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
