# Pilot 3D Rendering Migration Plan

## Introduction

The solvcon pilot GUI is built on Qt 3D (the
[`Qt6::3DCore`](https://doc.qt.io/qt-6/qt3dcore.html),
[`Qt6::3DRender`](https://doc.qt.io/qt-6/qt3drender.html),
[`Qt6::3DInput`](https://doc.qt.io/qt-6/qt3dinput.html), and
[`Qt6::3DExtras`](https://doc.qt.io/qt-6/qt3dextras.html) modules). But Qt 3D
has been [officially deprecated, put into maintenance-only status, and dropped
from the official Qt binary releases at Qt 6.8 (October
2024)](https://doc.qt.io/qt-6/whatsnew68.html). The latest Qt (6.11) does not
ship it, so the pilot GUI no longer builds against a stock Qt.

To move forward, we migrate to [Qt Quick
3D](https://doc.qt.io/qt-6/qtquick3d-index.html), Qt's supported successor. It
renders through [QRhi](https://doc.qt.io/qt-6/qrhi.html), the portable graphics
abstraction shared by all of Qt 6 (OpenGL, Vulkan, Metal, D3D11). That same
QRhi layer is also the low-level extension point for advanced rendering. The
[Qt Widgets](https://doc.qt.io/qt-6/qtwidgets-index.html) application shell is
kept, and the work is confined to the `R*` classes under `cpp/solvcon/pilot/`:

- `R3DWidget` -- hosts a
  [`Qt3DExtras::Qt3DWindow`](https://doc.qt.io/qt-6/qt3dextras-qt3dwindow.html)
  inside a [`QWidget`](https://doc.qt.io/qt-6/qwidget.html) via
  [`QWidget::createWindowContainer`](https://doc.qt.io/qt-6/qwidget.html#createWindowContainer),
  and owns the axis-gizmo overlay built from a custom [frame
  graph](https://doc.qt.io/qt-6/qt3drender-framegraph.html)
  ([`QViewport`](https://doc.qt.io/qt-6/qt3drender-qviewport.html),
  [`QLayerFilter`](https://doc.qt.io/qt-6/qt3drender-qlayerfilter.html),
  [`QClearBuffers`](https://doc.qt.io/qt-6/qt3drender-qclearbuffers.html),
  [`QCameraSelector`](https://doc.qt.io/qt-6/qt3drender-qcameraselector.html),
  [`QRenderSurfaceSelector`](https://doc.qt.io/qt-6/qt3drender-qrendersurfaceselector.html)).
- `RScene` / `RWorld` -- the
  [`Qt3DCore::QEntity`](https://doc.qt.io/qt-6/qt3dcore-qentity.html) scene
  graph and bounding-box tracking.
- `RStaticMesh` -- custom geometry: mesh vertices, per-vertex colours, and
  index buffers uploaded through
  [`Qt3DCore::QGeometry`](https://doc.qt.io/qt-6/qt3dcore-qgeometry.html),
  [`QAttribute`](https://doc.qt.io/qt-6/qt3dcore-qattribute.html), and
  [`QBuffer`](https://doc.qt.io/qt-6/qt3dcore-qbuffer.html) (about 50
  [`QAttribute`](https://doc.qt.io/qt-6/qt3dcore-qattribute.html) uses), drawn with
  [`QGeometryRenderer`](https://doc.qt.io/qt-6/qt3drender-qgeometryrenderer.html).
- `RAxisMark` -- axis arrows and labels from
  [`Qt3DExtras::QConeMesh`](https://doc.qt.io/qt-6/qt3dextras-qconemesh.html)
  and
  [`QExtrudedTextMesh`](https://doc.qt.io/qt-6/qt3dextras-qextrudedtextmesh.html).
- `RCameraController` -- orbit and first-person controllers built directly on
  [`Qt6::3DInput`](https://doc.qt.io/qt-6/qt3dinput.html)
  ([`QAxis`](https://doc.qt.io/qt-6/qt3dinput-qaxis.html),
  [`QButtonAxisInput`](https://doc.qt.io/qt-6/qt3dinput-qbuttonaxisinput.html),
  [`QAnalogAxisInput`](https://doc.qt.io/qt-6/qt3dinput-qanalogaxisinput.html),
  [`QAction`](https://doc.qt.io/qt-6/qt3dinput-qaction.html),
  [`QMouseDevice`](https://doc.qt.io/qt-6/qt3dinput-qmousedevice.html),
  [`QKeyboardDevice`](https://doc.qt.io/qt-6/qt3dinput-qkeyboarddevice.html)).
- Materials:
  [`Qt3DExtras::QDiffuseSpecularMaterial`](https://doc.qt.io/qt-6/qt3dextras-qdiffusespecularmaterial.html)
  and
  [`QPerVertexColorMaterial`](https://doc.qt.io/qt-6/qt3dextras-qpervertexcolormaterial.html).

The rest of the pilot uses plain Qt widgets that do not touch Qt 3D: docking,
the Python console (`RPythonConsoleDockWidget`), the 2D viewer (`R2DWidget`,
`RWorldRenderer2d`), the manager, and the pybind11 bindings (`wrap_pilot.cpp`).

## System design

[Qt Quick 3D](https://doc.qt.io/qt-6/qtquick3d-index.html) maps onto the
viewer's needs as follows:

- **Custom geometry** -- subclass
  [`QQuick3DGeometry`](https://doc.qt.io/qt-6/qquick3dgeometry.html) in C++ and
  feed it the same vertex / colour / index buffers we already assemble in
  `RStaticMesh`. This is a first-class, documented C++ path.
- **Per-vertex colour** -- supplied as a custom vertex attribute and consumed
  by a
  [`CustomMaterial`](https://doc.qt.io/qt-6/qml-qtquick3d-custommaterial.html)
  shader, replacing [`QPerVertexColorMaterial`](https://doc.qt.io/qt-6/qt3dextras-qpervertexcolormaterial.html).
- **Widget embedding** -- host a
  [`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html) in a
  [`QQuickWidget`](https://doc.qt.io/qt-6/qquickwidget.html), which renders
  into the widget's framebuffer and composes correctly with neighbouring
  widgets (unlike
  [`createWindowContainer`](https://doc.qt.io/qt-6/qwidget.html#createWindowContainer),
  used today, which can draw over sibling widgets).
- **Axis-gizmo overlay** -- a second [`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html) with its own camera, instead of a
  hand-rolled frame graph.
- **Camera control** -- Qt Quick 3D ships
  [`OrbitCameraController`](https://doc.qt.io/qt-6/qml-qtquick3d-helpers-orbitcameracontroller.html)
  and
  [`WasdController`](https://doc.qt.io/qt-6/qml-qtquick3d-helpers-wasdcontroller.html);
  our bespoke controllers can be reimplemented on top of these or with
  widget-level input handling.

The principal caveat is that **Qt Quick 3D is
[QML](https://doc.qt.io/qt-6/qtqml-index.html)-first.** Its public C++ API
(notably [`QQuick3DGeometry`](https://doc.qt.io/qt-6/qquick3dgeometry.html)) has
grown, but scene assembly is still expected to happen in QML, and a fully
programmatic C++-only scene is not the intended workflow. The realistic shape
of the migration is therefore a thin QML scene defining the
[`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html), cameras, lights,
and materials, with the heavy data -- mesh geometry and colour fields -- pushed
from C++ through [types registered with
`QML_ELEMENT`](https://doc.qt.io/qt-6/qtqml-cppintegration-definetypes.html) and
[`QQuick3DGeometry`](https://doc.qt.io/qt-6/qquick3dgeometry.html) subclasses.
This is a structural change from today's all-C++ scene graph, but it keeps the
numerics and buffer management in C++ where they belong.

Qt Quick 3D is not a closed high-level engine: because it is itself a QRhi
renderer, the same QRhi layer is the viewer's **low-level extension point** for
anything the scene graph and materials cannot express -- custom shaders,
compute, or direct draw calls -- without leaving the engine or standing up a
second renderer. The custom code runs on the one QRhi the whole window already
uses, so it shares one graphics device, one backend, and one shader toolchain
([`QShader`](https://doc.qt.io/qt-6/qshader.html) / `.qsb`, baked by the
[`qsb`](https://doc.qt.io/qt-6/qtshadertools-qsb.html) tool already in the
dependency prefix). Most custom shading -- including the boundary highlight's
per-vertex colour -- is met by a [`CustomMaterial`](https://doc.qt.io/qt-6/qml-qtquick3d-custommaterial.html)
or a post-processing
[`Effect`](https://doc.qt.io/qt-6/qml-qtquick3d-effect.html) without touching
raw QRhi at all. When that is not enough, three documented seams inject custom
QRhi passes, from most to least integrated:

1. **Inside the 3D scene** --
   [`QQuick3DRenderExtension`](https://doc.qt.io/qt-6/qquick3drenderextension.html)
   plus
   [`QSSGRenderExtension`](https://doc.qt.io/qt-6/qssgrenderextension.html)
   (Qt 6.7+). Subclass the extension, return a
   [`QSSGRenderExtension`](https://doc.qt.io/qt-6/qssgrenderextension.html) from
   `updateSpatialNode()`, and implement `prepareData()` / `prepareRender()` /
   `render()`; a
   [`QSSGFrameData`](https://doc.qt.io/qt-6/qssgframedata.html) hands over the
   live QRhi, command buffer, render target, depth, and camera. The
   `Standalone` mode renders into a texture that can feed a material or effect;
   the `Main` mode injects into the scene's pass (`PreColor` / `PostColor`),
   depth-tested against scene geometry. This is the right seam for custom
   shaders or compute drawn *within* the mesh view.
2. **Beside the View3D** --
   [`QSGRenderNode`](https://doc.qt.io/qt-6/qsgrendernode.html) (QRhi-based
   since Qt 6.6), a custom scene-graph node issuing QRhi draw calls inline with
   the 2D scene graph and composited with the
   [`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html).
3. **Window underlay/overlay** -- the
   [`QQuickWindow`](https://doc.qt.io/qt-6/qquickwindow.html) `beforeRendering`
   / `beforeRenderPassRecording` (underlay) and `afterRenderPassRecording` /
   `afterRendering` (overlay) signals, for raw QRhi before or after the whole
   scene. Simplest, but with no depth mixing against the scene.

The shared QRhi is reached through
[`QQuickWindow::rendererInterface()`](https://doc.qt.io/qt-6/qquickwindow.html#rendererInterface)
-> `getResource(window, QSGRendererInterface::RhiResource)` (see
[`QSGRendererInterface`](https://doc.qt.io/qt-6/qsgrendererinterface.html)).
The trade-off is API stability: QRhi has been **semi-public** since Qt 6.6 --
documented and usable, but with only a limited source/binary-compatibility
guarantee, like the [QPA](https://doc.qt.io/qt-6/qpa.html) classes. The
render-extension types ([`QSSGRenderExtension`](https://doc.qt.io/qt-6/qssgrenderextension.html),
[`QSSGFrameData`](https://doc.qt.io/qt-6/qssgframedata.html)) are more volatile
still: they live in the `QtQuick3DRuntimeRender` module under a version-pinned
include path and can shift between minor releases.
[`QSGRenderNode`](https://doc.qt.io/qt-6/qsgrendernode.html) (seam 2) and the
window signals (seam 3) are the steadier options. The Qt 6.11
build in the dependency environment already ships all three seams, so they are
available without further dependency work.

**PySide6 stays compatible.** It ships a
[`QtQuick3D`](https://doc.qt.io/qtforpython-6/PySide6/QtQuick3D/index.html)
binding that even supports subclassing
[`QQuick3DGeometry`](https://doc.qt.io/qtforpython-6/PySide6/QtQuick3D/QQuick3DGeometry.html)
in Python, and since `R3DWidget` stays a
[`QWidget`](https://doc.qt.io/qt-6/qwidget.html) bridged through libpyside6, the
migration needs no Python-side change beyond rebuilding PySide6 against the new
module.

## Migration plan

The migration is staged so the pilot stays buildable and runnable at every
step. The original Qt 3D viewer was never production-proven, so the move to Qt
Quick 3D is also an opportunity to improve the visualization system. Each step
below should be a self-contained, reviewable change.

We expect unexpected needs to surface during the migration, and will adjust the
plan and this document as that happens.

### 1. Spike the data path

Build a throwaway [`QQuickWidget`](https://doc.qt.io/qt-6/qquickwidget.html)
hosting a [`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html) with a
single [`QQuick3DGeometry`](https://doc.qt.io/qt-6/qquick3dgeometry.html)
subclass fed from the existing `SimpleArray` buffers, plus a
[`CustomMaterial`](https://doc.qt.io/qt-6/qml-qtquick3d-custommaterial.html)
that reads a per-vertex colour attribute. Confirm an offscreen grab
([`QQuickWidget::grabFramebuffer`](https://doc.qt.io/qt-6/qquickwidget.html#grabFramebuffer))
matches the Qt 3D output, and settle the C++/QML boundary -- geometry providers
registered with
[`QML_ELEMENT`](https://doc.qt.io/qt-6/qtqml-cppintegration-definetypes.html)
and driven from C++, the QML scene kept thin. This de-risks the QML-first
constraint before any production code changes.

### 2. Scaffold the Qt Quick 3D backend

Add `find_package(Qt6 ... Quick Quick3D)` and a
[`qt_add_qml_module`](https://doc.qt.io/qt-6/qt-add-qml-module.html) block to
`cpp/binary/pilot/CMakeLists.txt`, and gate the new backend behind a build flag
(e.g. `PILOT_QUICK3D`) so the Qt 3D pilot keeps building during the port. The
scdv dependency build already ships `qtquick3d`
(`contrib/dependency/*/build-scdv-*.sh`). Stand up an empty
[`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html) inside `R3DWidget`
in place of the
[`Qt3DExtras::Qt3DWindow`](https://doc.qt.io/qt-6/qt3dextras-qt3dwindow.html)
container.

### 3. Port the mesh and boundary geometry

Replace `RStaticMesh`'s [`Qt3DCore::QGeometry`](https://doc.qt.io/qt-6/qt3dcore-qgeometry.html)
/ [`QAttribute`](https://doc.qt.io/qt-6/qt3dcore-qattribute.html) /
[`QBuffer`](https://doc.qt.io/qt-6/qt3dcore-qbuffer.html) wireframe with a
[`QQuick3DGeometry`](https://doc.qt.io/qt-6/qquick3dgeometry.html) subclass
(position attribute, `Lines` primitive), reusing the `SimpleArray` ->
[`QByteArray`](https://doc.qt.io/qt-6/qbytearray.html) assembly. Port
`RBoundary`'s coloured ribbon to a `ColorSemantic` attribute consumed by the
[`CustomMaterial`](https://doc.qt.io/qt-6/qml-qtquick3d-custommaterial.html).
Validate the result against the sample meshes exercised in
`solvcon/pilot/_mesh.py`.

### 4. Rebuild the scene and the axis overlay

Recreate the scene root as a [`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html)
with camera and lights, and port the colour-field path
(`R3DWidget::updateColorField`, per-cell triangles) and the world geometry
(`RLines` / `RVertices`, refreshed through `updateWorld`). Replace the
hand-rolled frame-graph axis gizmo with a second
[`View3D`](https://doc.qt.io/qt-6/qml-qtquick3d-view3d.html) overlay on its own
orthographic camera, porting `RAxisMark`'s
[`QConeMesh`](https://doc.qt.io/qt-6/qt3dextras-qconemesh.html) arrows and
[`QExtrudedTextMesh`](https://doc.qt.io/qt-6/qt3dextras-qextrudedtextmesh.html)
labels to Qt Quick 3D models and text.

### 5. Restore camera interaction

Reimplement `ROrbitCameraController` and `RFirstPersonCameraController` on the
Qt Quick 3D [`OrbitCameraController`](https://doc.qt.io/qt-6/qml-qtquick3d-helpers-orbitcameracontroller.html)
/ [`WasdController`](https://doc.qt.io/qt-6/qml-qtquick3d-helpers-wasdcontroller.html)
or on widget-level input, preserving the current key/mouse bindings and the
`fitCameraToScene` framing.

### 6. Reconnect the Python bindings and tests

Keep the pybind11 `R3DWidget` API (`updateMesh`, `showMesh`, `showBoundary`,
`updateColorField`, `showMark`, `updateWorld`, and the pixmap grab) unchanged
in `wrap_pilot.cpp`, repointing the grab at
[`QQuickWidget::grabFramebuffer`](https://doc.qt.io/qt-6/qquickwidget.html#grabFramebuffer).
Confirm `make run_pilot_pytest` and the GUI tests
(`tests/test_pilot_mesh_info.py`) pass on the new backend.

### 7. Cut over and remove Qt 3D

Make Qt Quick 3D the default, delete the Qt 3D classes and the `Qt6::3D*`
`find_package` calls from `cpp/binary/pilot/CMakeLists.txt`, drop the build
flag, and remove the Qt 3D modules from the dependency build. The pilot stays
runnable throughout, so this final step only retires code that nothing uses any
more.

## References

- [Qt Quick 3D Architecture](https://doc.qt.io/qt-6/qtquick3d-architecture.html)
- [Qt Quick 3D - Custom Geometry
  Example](https://doc.qt.io/qt-6/qtquick3d-customgeometry-example.html)
- [Programmable Materials, Effects, Geometry, and Texture
  data](https://doc.qt.io/qt-6/qtquick3d-custom.html)
- [Graphics in Qt 6.0: QRhi, Qt Quick, Qt Quick
  3D](https://www.qt.io/blog/graphics-in-qt-6.0-qrhi-qt-quick-qt-quick-3d)
- [What's New in Qt 6.8 (Qt 3D deprecation
  announcement)](https://doc.qt.io/qt-6/whatsnew68.html)
- [Changes to Qt 3D](https://doc.qt.io/qt-6/qt3d-changes-qt6.html)
- [What's New in Qt 6.6 (QRhi
  semi-public)](https://doc.qt.io/qt-6/whatsnew66.html)
- [What's New in Qt 6.7 (render
  extensions)](https://doc.qt.io/qt-6/whatsnew67.html)
- [Scene Graph - Custom QSGRenderNode
  example](https://doc.qt.io/qt-6/qtquick-scenegraph-customrendernode-example.html)
- [Scene Graph - RHI Under QML
  example](https://doc.qt.io/qt-6/qtquick-scenegraph-rhiunderqml-example.html)
- [Removing Qt 3D from the release configuration (qt-project
  dev list)](https://lists.qt-project.org/pipermail/development/2024-March/045127.html)
- [PySide6.QtQuick3D
  module](https://doc.qt.io/qtforpython-6/PySide6/QtQuick3D/index.html)
- [Qt Quick 3D Custom Geometry example (Qt for
  Python)](https://doc.qt.io/qtforpython-6/examples/example_quick3d_customgeometry.html)
- [Qt Quick Scene Graph (render
  loops)](https://doc.qt.io/qt-6/qtquick-visualcanvas-scenegraph.html)
- [Shiboken module (C++/Python
  interop)](https://doc.qt.io/qtforpython-6/shiboken6/shibokenmodule.html)

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
