# Domain Visualizer

The pilot's domain viewer ({cpp:class}`~solvcon::RDomainWidget`) shows 2D and
3D unstructured-mesh domains and fields, rendered through Qt's
[QRhi](https://doc.qt.io/qt-6/qrhi.html). It is hosted as a sub-window in the
pilot and is fully controllable from Python: populate the scene, navigate the
camera, toggle layers, and capture frames from code the same way the mouse
drives them.

## Showing a domain

- **Mesh wireframe**: `updateMesh(mesh)` draws the unstructured mesh (2D faces
  or 3D cells) as a wireframe; `showMesh(on)` toggles it.
- **Colored field**: `updateColorField(vertices, colors, indices)` draws
  per-vertex-colored triangles over the domain. The field is swappable at
  runtime: call it again to replace the previous one.
- **Boundary highlight**: `showBoundary(ibc, on)` highlights boundary set
  `ibc`.
- **Orientation guide**: `showAxis(on)` shows a small axis triad in the corner,
  two axes for a 2D domain and three for a 3D one, oriented by the camera. It
  is hidden by default.

## Camera navigation

The viewer has one camera with three modes; each suits a different domain.
Choose the mode from the **View > Camera** menu, or set `cameraMode` from
Python.

| Menu item (View > Camera) | `cameraMode` | Domain | Drag | Wheel / pinch |
| --- | --- | --- | --- | --- |
| Orbit camera (3D) | `"orbit"` | 3D | swing the eye around the center | dolly toward or away from the center |
| First-person camera (3D) | `"fps"` | 3D | look around in place | dolly along the view direction |
| Pan / zoom camera (2D) | `"pan"` | 2D | pan in the plane | zoom the orthographic view |

**Orbit is the default.** It is a turntable orbit: the up axis stays fixed, so
the horizon never rolls and a hard pitch past vertical eases off instead of
flipping the view. Loading a 2D domain switches to pan/zoom automatically,
because the orthographic projection a 2D domain uses ignores the orbit dolly
and wants the in-plane wheel zoom instead.

### Mouse, wheel, and pinch

- **Left-drag** rotates: orbit the center in orbit mode, look around in
  first-person mode, or pan in 2D pan/zoom mode.
- **Middle- or right-drag** pans in any mode.
- **Wheel** zooms; what it does depends on the mode (see the table above).
- **Pinch** on a trackpad or touchscreen zooms in any mode.

### Keyboard

- **W / A / S / D** or the **arrow keys** move the camera: forward and back,
  and strafe left and right.
- **Esc** reframes the whole domain (fit to scene).

The **View > Camera move** submenu offers the same nudges as clickable actions,
plus fixed-step yaw and pitch rotation and a reset.

### From Python

The camera exposes the same primitives the mouse and wheel drive, so a domain
navigates identically from code:

```python
from solvcon import pilot

mgr = pilot.RManager.instance.setUp()
viewer = mgr.add3DWidget()

viewer.updateMesh(mesh)             # show the wireframe
viewer.cameraMode = "orbit"         # orbit a 3D domain (the default)
viewer.fitCameraToScene()           # frame the whole domain

viewer.rotateCamera(40.0, 15.0)     # orbit by a pixel delta
viewer.panCamera(20.0, 0.0)         # pan by a pixel delta
viewer.zoomCamera(3.0)              # zoom by wheel notches
viewer.pinchCamera(1.5)             # zoom by a pinch factor (>1 zooms in)
```

The pose is readable and settable directly as `(x, y, z)` tuples through
`cameraPosition`, `cameraTarget`, and `cameraUp`, so a view can be saved and
restored, or set to an exact framing.

## Capturing frames

- `saveImage(path)` renders the current frame offscreen and writes it to an
  image file.
- `clipImage()` copies the current frame to the clipboard.

Both grab the frame deterministically inside the Qt event loop, which is what
the pytest screenshot tests rely on.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
