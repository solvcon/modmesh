# `RDomainWidget` for Field Analysis

## Goal

Build **`RDomainWidget`**, a new pilot widget for **interactive and
programmable** analysis of **2D and 3D spatial domains and fields** on
unstructured meshes. Make it fully controllable from Python in both `solvcon`
module code and interactive console operations. Python is the primary control
surface: instantiate the widget, embed it in the pilot, populate the scene
(domains, fields, world geometry, orientation guide), drive the camera, toggle
layers, recolor the field, and grab the framebuffer, all from Python. C++
receives the same API, though the style may differ in a way natural to the
language difference. C++ does the numerics and rendering; Python orchestrates
everything the widget does.

The widget renders through [QRhi](https://doc.qt.io/qt-6/qrhi.html), Qt's
portable graphics abstraction, hosted in a
[`QRhiWidget`](https://doc.qt.io/qt-6/qrhiwidget.html). This replaces the
pilot's current Qt 3D path, which has been [deprecated and dropped from stock Qt
at 6.8](https://doc.qt.io/qt-6/whatsnew68.html) so that the pilot no longer
builds against a current Qt binary release (building from source still works).

**The existing `R*` 3D classes are a prototype**, not production-proven. They
should be used as a **behavior reference**. We reimplement cleanly on QRhi under
new classes whose names are kept distinct from the prototype's `R*` classes, so
the prototype keeps building and running while the new widget is developed. The
to-be-developed **`RDomainWidget`** corresponds to **`R3DWidget`** in the
existing prototype.

The work stays confined to the pilot's domain-rendering code under
`cpp/solvcon/pilot/` and its pybind11 bindings. After the new **`RDomainWidget`**
is done, the old **`R3DWidget`** prototype based on Qt 3D should be removed.

### No change to existing 2D code

Docking, the Python console (`RPythonConsoleDockWidget`), the existing 2D viewer
(`R2DWidget`, `RWorldRenderer2d`), and the manager use plain Qt widgets that do
not touch Qt 3D and are unaffected. `RDomainWidget` brings its own 2D and 3D
domain rendering; the existing 2D viewer is independent and stays as is.

### Why not Qt Quick 3D

Qt's high-level successor, [Qt Quick
3D](https://doc.qt.io/qt-6/qtquick3d-index.html), requires QML to assemble the
scene, which would move the viewer out of C++ and away from a clean Python
control surface. QRhi is instead an imperative, buffer-and-draw-call API that
matches how solvcon's numerics already think, keeps everything in C++ behind
pybind11, adds no new dependency, and is Qt's actively developed rendering
core.

## Design of the new system

:::{admonition} Distinct names from the prototype
:class: note

The new classes take names distinct from the prototype's existing `R*` classes:
`R3DWidget`, `RScene`, `RStaticMesh`, `RColorField`, `RBoundary`, `RLines`,
`RVertices`, `RAxisMark`, and the `RCameraController` family. Some keep an
`RDomain*` prefix; others take a free or more descriptive name (`RDrawable`,
`RMaterial`, `RMeshFrame`, `RField`, `RMeshBoundary`, `RAxisGizmo`).
The only rule is that the names do not collide, so the Qt 3D prototype keeps
building and running side by side throughout development; the prototype's names
are freed only when it is retired (slice 8). Slice 9 then reclaims the two
cleaner ones, renaming `RMeshBoundary` to `RBoundary` and `RDomainScene` to
`RScene`.
:::

`RDomainWidget` is meant to grow into a tool for analyzing spatial domains and
fields. The design in this document is only for its core:

- **Construct and embed**: the pilot manager (`RManager`) creates
  `RDomainWidget` (a [`QWidget`](https://doc.qt.io/qt-6/qwidget.html)) and
  shows it as a sub-window in its central
  [`QMdiArea`](https://doc.qt.io/qt-6/qmdiarea.html), the same place the
  `R3DWidget` prototype lives (not a dock widget), then hands it to Python; the
  dockable analysis widgets sit around it. This follows today's
  `RManager::add3DWidget` factory.
- **Show 2D and 3D unstructured-mesh domains** from `SimpleArray` buffers: the
  mesh wireframe (lines), the cells (2D faces or 3D triangles), and boundary
  highlights.
- **Color a field over the domain** per vertex, swappable at runtime.
- **Show an orientation guide** (a 2- or 3-axis mark with labels) on its own
  camera.
- **Navigate the domain** with one camera controller offering pan/zoom (2D) and
  first-person (FPS, 3D) modes, both interactively (mouse/keys) and
  programmatically (set/get pose, switch mode, fit-to-scene).
- **Grab the framebuffer offscreen** for the pytest screenshot tests and for
  scripted capture.

:::{note}
Richer analytical operations (selection, query, measurement, slicing) and the
additional dockable widgets that will cooperate with `RDomainWidget` are
deliberately left unplanned here. World geometry, the auxiliary lines and
points drawn through `updateWorld` (the SVG/canvas feature), is rendered
ad-hoc on the generic `RDrawable` for the initial work; dedicated line and point
primitive classes are designed later. The class design only needs to leave room
for these extensions.
:::

The `R3DWidget` prototype meets the 3D side with Qt 3D custom geometry
([`QGeometry`](https://doc.qt.io/qt-6/qt3dcore-qgeometry.html) /
[`QAttribute`](https://doc.qt.io/qt-6/qt3dcore-qattribute.html) /
[`QBuffer`](https://doc.qt.io/qt-6/qt3dcore-qbuffer.html)),
[`QPerVertexColorMaterial`](https://doc.qt.io/qt-6/qt3dextras-qpervertexcolormaterial.html),
[`QConeMesh`](https://doc.qt.io/qt-6/qt3dextras-qconemesh.html) /
[`QExtrudedTextMesh`](https://doc.qt.io/qt-6/qt3dextras-qextrudedtextmesh.html),
a hand-rolled frame graph, and the
[`Qt6::3DInput`](https://doc.qt.io/qt-6/qt3dinput.html) declarative axis/action
system; none of that carries forward, but the `SimpleArray` domain and field
data does. The new design fulfills the requirements with the classes
below. C++ owns rendering, and the classes are arranged so that
**`RDomainWidget` is the single Python-facing control object**, with the rest
driven through it:

- **`RDomainWidget`** (prototype: `R3DWidget`): a
  [`QRhiWidget`](https://doc.qt.io/qt-6/qrhiwidget.html) subclass; the widget
  Python instantiates and controls. Owns the
  [`QRhi`](https://doc.qt.io/qt-6/qrhi.html), the per-frame render target and
  depth buffer, the main and overlay passes, the input dispatch, and the
  `RScene`. Renders into its own backing texture and composes with
  sibling widgets (unlike the
  [`createWindowContainer`](https://doc.qt.io/qt-6/qwidget.html#createWindowContainer)
  the `R3DWidget` prototype uses). Hosts the entire pybind11-exposed API.
- **`RScene`** (prototype: `RScene`, the name reclaimed in slice 9 after
  developing with the temporary name `RDomainScene`): owns the `RDrawable`s,
  the domain bounding box, and the active camera and controller; selects an
  orthographic projection for 2D domains and a perspective projection for 3D,
  and computes the fit-to-scene framing.
- **`RDrawable`** (new): abstract base for a renderable: owns its
  [`QRhiBuffer`](https://doc.qt.io/qt-6/qrhibuffer.html) vertex/index data and
  references an `RMaterial`. Buffer uploads go through a
  [`QRhiResourceUpdateBatch`](https://doc.qt.io/qt-6/qrhiresourceupdatebatch.html).
  Concrete kinds:
  - **`RMeshFrame`** (prototype: `RStaticMesh`): the domain wireframe
    (`Lines` topology), 2D or 3D.
  - **`RField`** (prototype: `RColorField`): the field-colored cells
    (2D faces / 3D surface).
  - **`RBoundary`** (prototype: `RBoundary`, the name reclaimed in slice 9
    after developing with the temporary name `RMeshBoundary`): the boundary
    highlight.
  - **`RAxisGizmo`** (prototype: `RAxisMark`): the orientation guide
    (2- or 3-axis arrows and labels), drawn in the overlay pass.
- **`RMaterial`** (new): wraps a
  [`QRhiGraphicsPipeline`](https://doc.qt.io/qt-6/qrhigraphicspipeline.html) and
  its [`QShader`](https://doc.qt.io/qt-6/qshader.html) set, in flat-color,
  per-vertex-color, and line variants. GLSL is baked to `.qsb` with the
  [`qsb`](https://doc.qt.io/qt-6/qtshadertools-qsb.html) tool already in the
  dependency prefix. This is the whole "material" system the widget needs.
- **`RDomainCameraController`** (prototype: `RCameraController`): a single
  controller holding the camera as view/projection
  [`QMatrix4x4`](https://doc.qt.io/qt-6/qmatrix4x4.html) and supporting both
  **pan/zoom** (2D) and **first-person/FPS** (3D) navigation as internal modes,
  with no separate per-mode camera classes. Driven by ordinary
  [`QWidget`](https://doc.qt.io/qt-6/qwidget.html) mouse/key events and by
  Python commands. Replaces the `R3DWidget` prototype's
  [`Qt6::3DInput`](https://doc.qt.io/qt-6/qt3dinput.html) declarative graph
  outright.

The numerics and buffer management stay in C++ on the zero-copy `SimpleArray`
path. No QML is introduced.

### Python control surface

`RDomainWidget` exposes, through pybind11, the full set of operations needed to
drive it headlessly or interactively from Python (representative methods):

- construction and embedding as a `QWidget` (via PySide6)
- `updateMesh(mesh)`, `showMesh(on)`, `showBoundary(ibc, on)`
- `updateColorField(vertices, colors, indices)`
- `showAxis(on)` for the orientation guide
- camera control: `setCameraMode(mode)` (pan/zoom or first-person), get/set
  camera pose, `fitCameraToScene()`
- `grabImage()` -> [`QImage`](https://doc.qt.io/qt-6/qimage.html) for offscreen
  capture

Selected new types may also be bound directly for advanced scripting, but
`RDomainWidget` should provide intuitive helper functions.

:::{admonition} PySide6 compatibility
:class: note

`RDomainWidget` is a [`QWidget`](https://doc.qt.io/qt-6/qwidget.html) bridged
through libpyside6, so it embeds in PySide6 layouts and is scriptable from the
Python console with no architectural special-casing. PySide6 need not wrap
[`QRhiWidget`](https://doc.qt.io/qt-6/qrhiwidget.html) itself; it sees
`RDomainWidget` as a `QWidget`, as it does the `R3DWidget` prototype today.
The heavy data path crosses the C++/pybind11 boundary as zero-copy
`SimpleArray`; no QML is involved. Full Python control is thus the combination
of the pybind11 `RDomainWidget` API and the libpyside6 `QWidget` bridge.
:::

### Event loop

The move changes how frames are produced. The `R3DWidget` prototype, on Qt 3D,
ran its own continuous, multithreaded render-and-aspect loop alongside the Qt
event loop, producing frames independently of event processing and grabbing
offscreen images asynchronously (via
[`QRenderCapture`](https://doc.qt.io/qt-6/qt3drender-qrendercapture.html)).
`RDomainWidget` instead renders on demand inside the **single** Qt event loop:
[`update()`](https://doc.qt.io/qt-6/qwidget.html#update) schedules a repaint and
the QRhi `render()` callback runs synchronously when that paint event is
dispatched, so one frame is one turn of the loop. The widget needs no engine
loop of its own, and the existing Python control
([`RManager.exec()`](https://doc.qt.io/qt-6/qapplication.html#exec) or
[`QApplication::processEvents()`](https://doc.qt.io/qt-6/qcoreapplication.html#processEvents))
drives both rendering and a synchronous `grabImage()` deterministically, with no
background render thread to wait on, which also makes the pytest screenshots
reproducible.

## Implementation plan

The work is organized as **vertical capability slices**. Each slice is a
self-contained, reviewable change that leaves the widget runnable, adds one
visible capability, **and exposes that capability to Python in the same slice**,
so programmable control grows with the widget rather than being bolted on at
the end. `RDomainWidget` is developed alongside the `R3DWidget` prototype and
replaces it once it reaches parity; gating the two behind a build flag (e.g.
`PILOT_QRHI`) is optional, since the prototype does not build on current Qt
anyway.

We expect unexpected needs to surface, and will adjust the plan and this
document as that happens.

### 1. Widget foundation

Stand up `RDomainWidget` as a [`QRhiWidget`](https://doc.qt.io/qt-6/qrhiwidget.html)
with the render loop, clear, and depth buffer, drawing one trivial primitive
through an `RMaterial` (pipeline plus a GLSL shader pair baked by
[`qsb`](https://doc.qt.io/qt-6/qtshadertools-qsb.html)). Bind it so Python can
construct it, embed it via PySide6, and call `grabImage()`, the binding over
[`QRhiWidget::grabFramebuffer()`](https://doc.qt.io/qt-6/qrhiwidget.html).
Settle the build wiring (`Qt6::GuiPrivate`, `#include <rhi/qrhi.h>`, the `qsb`
step). This slice establishes both the render foundation and the Python control
spine.

### 2. Domain rendering (2D and 3D)

Add `RDrawable` and `RMeshFrame`, render the unstructured-mesh domain wireframe
from the `SimpleArray` buffers for both 2D and 3D meshes, and bind `updateMesh`
/ `showMesh`. Validate against the unstructured sample mesh exercised in
`solvcon/pilot/_oblique.py`, driven from Python.

### 3. Field coloring and boundary highlight

Add `RField` and `RMeshBoundary` with the per-vertex-color `RMaterial` variant,
and bind `updateColorField` / `showBoundary`. The field is swappable at
runtime from Python.

### 4. Scene and framing

Assemble `RDomainScene` with its domain bounding box and per-dimension
projection, implement fit-to-scene, and bind `fitCameraToScene`.

### 5. Camera interaction and control

Implement `RDomainCameraController` as a single controller with pan/zoom (2D)
and first-person (FPS, 3D) modes (no per-mode camera classes), driven by
[`QWidget`](https://doc.qt.io/qt-6/qwidget.html) mouse/key events feeding the
[`QMatrix4x4`](https://doc.qt.io/qt-6/qmatrix4x4.html) view/projection. Bind the
mode switch and programmatic pose get/set so Python navigates the domain as well
as the mouse does.

### 6. Orientation-guide overlay

Render `RAxisGizmo` as a second pass on its own camera in a corner
viewport: a hand-built cone for the arrows and texture billboards for the labels
(e.g. [`QPainter`](https://doc.qt.io/qt-6/qpainter.html) into a
[`QImage`](https://doc.qt.io/qt-6/qimage.html) uploaded to a
[`QRhiTexture`](https://doc.qt.io/qt-6/qrhitexture.html)), as a 2- or 3-axis
guide. Bind `showAxis`.

### 7. Consolidate the Python API and tests

Finalize and document the `RDomainWidget` pybind11 surface, point `RManager`'s
widget factory (`add3DWidget` and the MDI hosting) at `RDomainWidget`, update
its Python callers (`solvcon/pilot/_mesh.py`, `_canvas_gui.py`) and the GUI
tests, and route the screenshot path through `RDomainWidget.grabImage()`.
Confirm `make run_pilot_pytest` and `tests/test_pilot_mesh_info.py` pass,
exercising the widget through its Python control surface.

### 8. Retire the 3D prototype

Delete the Qt 3D prototype classes (`R3DWidget`, `RScene`, `RStaticMesh`,
`RColorField`, `RBoundary`, `RLines`, `RVertices`, `RAxisMark`, the
`RCameraController` family, and the rest) and the `Qt6::3D*`
`find_package` calls from `cpp/binary/pilot/CMakeLists.txt`, drop the build flag
if one was used, and remove the Qt 3D modules from the dependency build. Only
dead prototype code is removed; `RDomainWidget` has already taken over the 3D
view.

### 9. Reclaim class names

With the `R3DWidget` prototype retired, its names are free again. Reclaim the
cleaner ones: rename `RMeshBoundary` to `RBoundary` and `RDomainScene` to
`RScene`.

## References

- [QRhi](https://doc.qt.io/qt-6/qrhi.html)
- [QRhiWidget (Qt 6.7+, Qt Widgets)](https://doc.qt.io/qt-6/qrhiwidget.html)
- [QRhiBuffer](https://doc.qt.io/qt-6/qrhibuffer.html)
- [QRhiGraphicsPipeline](https://doc.qt.io/qt-6/qrhigraphicspipeline.html)
- [QRhiResourceUpdateBatch](https://doc.qt.io/qt-6/qrhiresourceupdatebatch.html)
- [QShader](https://doc.qt.io/qt-6/qshader.html) /
  [qsb tool](https://doc.qt.io/qt-6/qtshadertools-qsb.html)
- [QOpenGLWidget (fallback host)](https://doc.qt.io/qt-6/qopenglwidget.html)
- [What's New in Qt 6.8 (Qt 3D deprecation
  announcement)](https://doc.qt.io/qt-6/whatsnew68.html)
- [Changes to Qt 3D](https://doc.qt.io/qt-6/qt3d-changes-qt6.html)
- [Qt Quick 3D (rejected target)](https://doc.qt.io/qt-6/qtquick3d-index.html)
- [PySide6.QtWidgets.QRhiWidget](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QRhiWidget.html)

## Appendix: QRhi API stability

The QRhi API is semi-public and may change, but across its whole lifetime (Qt
6.6 through 6.11) it has evolved almost entirely by addition. The single
documented backward-incompatible change is binary-only and would not have
forced a code edit, and because solvcon builds Qt and the pilot from source
together, even that one change costs nothing. It is a safe foundation for
`RDomainWidget`.

:::{note}
The [QRhi class documentation](https://doc.qt.io/qt-6/qrhi.html) states that
the QRhi family (with [`QShader`](https://doc.qt.io/qt-6/qshader.html) and
`QShaderDescription`) offers "no source or binary compatibility guarantees,"
that the API "is only guaranteed to work with the Qt version the application
was developed against," and that "source incompatible changes are however
aimed to be kept at a minimum and will only be made in minor releases (6.7,
6.8, and so on)." The headers live under the `rhi/` prefix
(`#include <rhi/qrhi.h>`) and require linking `Qt::GuiPrivate`. Treat a Qt
minor-version bump as the moment to re-check, not as something the compiler
guarantees.
:::

A diff of the public installed header `src/gui/rhi/qrhi.h` between tag
**v6.6.0** (when QRhi became semi-public) and **v6.11.0**, comparing every
method signature and enum enumerator and cross-checking the per-release notes
in the `qtreleasenotes` repository for 6.6 through 6.11, found **zero removed
or renamed public methods and zero removed or renamed enum values** over five
minor releases. The only backward-incompatible changes were:

1. **`QRhi::create()`, binary-incompatible but source-compatible** (Qt 6.10,
   [QTBUG-139778](https://bugreports.qt.io/browse/QTBUG-139778), the only
   change Qt explicitly tagged `BIC`). A graphics-adapter parameter was added
   for the new `enumerateAdapters()` feature. As of 6.11 it is a second
   overload, so the original four-argument `create(Implementation,
   QRhiInitParams*, Flags, QRhiNativeHandles*)` is untouched and existing call
   sites still compile. The break is ABI only: recompile, no code edit.
2. **Removed internal constants and a helper**: `BINDING_PREALLOC`,
   `LAYOUT_DESC_ENTRIES_PER_BINDING`, `MAX_MIP_LEVELS`,
   `MAX_TEX_SAMPLER_ARRAY_SIZE`, and `runCleanup()` left the public header.
   They are implementation details (for example, `BINDING_PREALLOC` only sized
   an internal `QVarLengthArray` member), not APIs application code calls.
3. **RHI-adjacent, for completeness** (not QRhi core): Qt 6.6 changed the
   `QSGMaterialShader::GraphicsPipelineState` struct (the scene-graph layer),
   and Qt 6.11 dropped `QShader` HLSL output and handwritten-HLSL
   geometry-shader injection (shader tooling, niche).

Everything else QRhi gained in 6.7 through 6.11 was additive: short, ushort,
Int, and HalfFloat vertex attributes, partial-region readbacks, no-copy
`QByteArray` upload overloads, `enumerateAdapters()`, exotic integer texture
formats, variable-rate shading, and completed multiview.

Because solvcon recompiles against every Qt version anyway, none of these
changes affect the project.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
