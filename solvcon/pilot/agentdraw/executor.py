# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Apply validated Agent Draw commands to a ``World``.

The executor validates each command against the JSON Schema in ``schema.py``,
fills in defaults, and dispatches to a handler that calls the ``World`` API.
Rendering is delegated to an injected, caller-supplied renderer so this layer
stays headless and deterministic (no Qt, no event loop); the offscreen QImage
renderer plugs in there. Errors are captured into a failed ``CommandResult``
rather than raised, so a recording harness can log every step.
"""

import json
import base64
from dataclasses import dataclass

from .schema import (
    CommandError,
    apply_defaults,
    validate_command,
    validate_result,
    validate_script,
)


def _world_point(coords):
    import solvcon
    z = coords[2] if len(coords) > 2 else 0.0
    return solvcon.Point3dFp64(coords[0], coords[1], z)


def _require_live(world, shape_id):
    """Raise a uniform CommandError when no live shape has the given id.

    The World API raises C++-flavored IndexError/ValueError for a missing or
    DEAD id; preflighting with shape_is_live lets every by-id handler fail the
    same clean way get_shape does, so agents and log parsers see one error
    shape for one logical error.
    """
    if not world.shape_is_live(shape_id):
        raise CommandError(f"no live shape with id {shape_id}")


def _h_add_point(world, a, ctx):
    world.add_point(a["x"], a["y"], a["z"])
    return {"npoint": world.npoint}


def _h_add_segment(world, a, ctx):
    world.add_segment(_world_point(a["p0"]), _world_point(a["p1"]))
    return {"nsegment": world.nsegment}


def _h_add_line(world, a, ctx):
    return {"shape_id": world.add_line(a["x0"], a["y0"], a["x1"], a["y1"])}


def _h_add_triangle(world, a, ctx):
    return {"shape_id": world.add_triangle(
        a["x0"], a["y0"], a["x1"], a["y1"], a["x2"], a["y2"])}


def _h_add_rectangle(world, a, ctx):
    return {"shape_id": world.add_rectangle(
        a["x_min"], a["y_min"], a["x_max"], a["y_max"])}


def _h_add_square(world, a, ctx):
    return {"shape_id": world.add_square(a["x_min"], a["y_min"], a["size"])}


def _h_add_ellipse(world, a, ctx):
    return {"shape_id": world.add_ellipse(a["cx"], a["cy"], a["rx"], a["ry"])}


def _h_add_circle(world, a, ctx):
    return {"shape_id": world.add_circle(a["cx"], a["cy"], a["r"])}


def _h_add_bezier(world, a, ctx):
    world.add_bezier(_world_point(a["p0"]), _world_point(a["p1"]),
                     _world_point(a["p2"]), _world_point(a["p3"]))
    return {"nbezier": world.nbezier}


def _h_add_bezier_shape(world, a, ctx):
    return {"shape_id": world.add_bezier_shape(
        _world_point(a["p0"]), _world_point(a["p1"]),
        _world_point(a["p2"]), _world_point(a["p3"]))}


def _h_translate_shape(world, a, ctx):
    _require_live(world, a["shape_id"])
    world.translate_shape(a["shape_id"], a["dx"], a["dy"])
    return {}


def _h_remove_shape(world, a, ctx):
    _require_live(world, a["shape_id"])
    world.remove_shape(a["shape_id"])
    return {}


def _h_clear(world, a, ctx):
    world.clear()
    return {}


def _h_nshape(world, a, ctx):
    return {"nshape": world.nshape}


def _h_shape_type_of(world, a, ctx):
    _require_live(world, a["shape_id"])
    return {"type": world.shape_type_of(a["shape_id"])}


def _h_get_shape(world, a, ctx):
    # The World has no read-one accessor, so filter the rendered state by id.
    # A dead or never-created id is a failed read, not an empty success.
    shape_id = a["shape_id"]
    state = json.loads(world.describe_state())
    for shape in state["shapes"]:
        if shape["id"] == shape_id:
            return {"shape": shape}
    raise CommandError(f"no live shape with id {shape_id}")


def _h_query_visible(world, a, ctx):
    ids = world.query_visible(a["min_x"], a["min_y"], a["max_x"], a["max_y"])
    return {"shape_ids": list(ids)}


def _h_describe_state(world, a, ctx):
    return {"state": json.loads(world.describe_state(level=a["level"]))}


def _h_render_png(world, a, ctx):
    if ctx.renderer is None:
        raise CommandError(
            "render_png needs a renderer; none configured. The harness and "
            "MCP front-ends inject the offscreen QImage renderer.")
    png = ctx.renderer(world, a["view"], a["width"], a["height"],
                       a["antialiasing"])
    # Base64-encode so the result is plain JSON over any transport (MCP image
    # content, a recorded log) rather than raw, non-serializable bytes.
    return {"image": {"data": base64.b64encode(png).decode("ascii"),
                      "mime_type": "image/png",
                      "width": a["width"], "height": a["height"]}}


def _h_log(world, a, ctx):
    ctx.append_log(a["message"])
    return {}


_HANDLERS = {
    "add_point": _h_add_point,
    "add_segment": _h_add_segment,
    "add_line": _h_add_line,
    "add_triangle": _h_add_triangle,
    "add_rectangle": _h_add_rectangle,
    "add_square": _h_add_square,
    "add_ellipse": _h_add_ellipse,
    "add_circle": _h_add_circle,
    "add_bezier": _h_add_bezier,
    "add_bezier_shape": _h_add_bezier_shape,
    "translate_shape": _h_translate_shape,
    "remove_shape": _h_remove_shape,
    "clear": _h_clear,
    "nshape": _h_nshape,
    "shape_type_of": _h_shape_type_of,
    "get_shape": _h_get_shape,
    "query_visible": _h_query_visible,
    "describe_state": _h_describe_state,
    "render_png": _h_render_png,
    "log": _h_log,
}


@dataclass
class CommandResult:
    """The outcome of applying one command."""
    op: str
    ok: bool
    value: object = None
    error: str = None


def _op_of(command):
    if isinstance(command, dict):
        return command.get("op", "?")
    return "?"


class Executor:
    """Apply validated commands to a ``World``, recording a structured log.

    ``renderer`` is a callable
    ``renderer(world, view, width, height, antialiasing) -> bytes`` supplied
    by the harness/MCP front-ends; with no renderer, ``render_png`` returns a
    failed result rather than touching a GUI.

    With ``validate_results=True`` each success value is checked against its
    op's ``RESULT_SCHEMAS`` entry, so the published output contract is enforced
    at runtime: a handler or ``World`` change that returns an off-contract
    value becomes a failed result instead of silently reporting ``ok``. It is
    off by default to keep the hot path free of per-call schema validation.
    """

    def __init__(self, world, renderer=None, validate_results=False):
        self.world = world
        self.renderer = renderer
        self.validate_results = validate_results
        self._log = []

    def append_log(self, message):
        self._log.append(message)

    @property
    def log(self):
        return list(self._log)

    def run(self, command):
        """Validate and apply one command, returning its ``CommandResult``.

        Validation and application errors are captured into a failed result
        so a recording harness can log every step instead of aborting.
        """
        try:
            validate_command(command)
        except CommandError as exc:
            return CommandResult(_op_of(command), False, error=str(exc))
        op = command["op"]
        args = apply_defaults(command)
        try:
            value = _HANDLERS[op](self.world, args, self)
            if self.validate_results:
                validate_result(op, value)
        except CommandError as exc:
            return CommandResult(op, False, error=str(exc))
        except Exception as exc:
            return CommandResult(
                op, False, error=f"{type(exc).__name__}: {exc}")
        return CommandResult(op, True, value=value)

    def run_script(self, commands):
        """Validate a whole script, then apply each command in order."""
        validate_script(commands)
        return [self.run(c) for c in commands]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
