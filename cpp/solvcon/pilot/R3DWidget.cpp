/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/R3DWidget.hpp> // Must be the first include.
#include <solvcon/pilot/RAxisMark.hpp>
#include <solvcon/pilot/RStaticMesh.hpp>

#include <Qt3DRender/QCameraLens>
#include <Qt3DRender/QClearBuffers>
#include <Qt3DRender/QCameraSelector>
#include <Qt3DRender/QDepthTest>
#include <Qt3DRender/QLayer>
#include <Qt3DRender/QLayerFilter>
#include <Qt3DRender/QMultiSampleAntiAliasing>
#include <Qt3DRender/QRenderStateSet>
#include <Qt3DRender/QRenderSurfaceSelector>
#include <Qt3DRender/QViewport>

#include <Qt3DExtras/QForwardRenderer>

#include <QRectF>

namespace solvcon
{

R3DWidget::R3DWidget(Qt3DExtras::Qt3DWindow * window, RScene * scene, QWidget * parent, Qt::WindowFlags f)
    : QWidget(parent, f)
    , m_view(nullptr == window ? new Qt3DExtras::Qt3DWindow : window)
    , m_scene(nullptr == scene ? new RScene : scene)
    , m_container(createWindowContainer(m_view, this, Qt::Widget))
{
    m_view->setRootEntity(m_scene);

    cameraController()->setCamera(m_view->camera());
    cameraController()->reset();

    setupAxisGizmo();

    if (Toggle::instance().fixed().get_show_axis())
    {
        showMark();
    }
}

void R3DWidget::setupAxisGizmo()
{
    QColor const clear_color = m_view->defaultFrameGraph()->clearColor();

    m_axis_layer = new Qt3DRender::QLayer(m_scene);
    m_axis_layer->setRecursive(true);

    // A dedicated camera for the gizmo.
    m_axis_camera = new Qt3DRender::QCamera(m_scene);
    // Orthographic projection keeps the triad undistorted and constant-size.
    m_axis_camera->lens()->setOrthographicProjection(
        -1.8f, 1.8f, -1.8f, 1.8f, 0.1f, 100.0f);

    auto make_state = [](Qt3DRender::QFrameGraphNode * parent)
    {
        auto * state = new Qt3DRender::QRenderStateSet(parent);
        auto * depth = new Qt3DRender::QDepthTest(state);
        depth->setDepthFunction(Qt3DRender::QDepthTest::Less);
        state->addRenderState(depth);
        state->addRenderState(new Qt3DRender::QMultiSampleAntiAliasing(state));
        return state;
    };

    auto * surface_selector = new Qt3DRender::QRenderSurfaceSelector();
    surface_selector->setSurface(m_view);

    // Main branch: full window, everything except the axis layer.
    auto * main_viewport = new Qt3DRender::QViewport(surface_selector);
    main_viewport->setNormalizedRect(QRectF(0.0f, 0.0f, 1.0f, 1.0f));
    auto * main_clear = new Qt3DRender::QClearBuffers(main_viewport);
    main_clear->setBuffers(Qt3DRender::QClearBuffers::ColorDepthBuffer);
    main_clear->setClearColor(clear_color);
    auto * main_camera_selector = new Qt3DRender::QCameraSelector(main_clear);
    main_camera_selector->setCamera(m_view->camera());
    auto * main_filter = new Qt3DRender::QLayerFilter(main_camera_selector);
    main_filter->addLayer(m_axis_layer);
    main_filter->setFilterMode(Qt3DRender::QLayerFilter::DiscardAnyMatchingLayers);
    make_state(main_filter);

    // Gizmo branch: floating corner viewport, only the axis layer.
    m_axis_viewport = new Qt3DRender::QViewport(surface_selector);
    // Lower-left fallback; updateAxisViewport() refines this to keep the
    // gizmo square once the actual window size is known.
    m_axis_viewport->setNormalizedRect(QRectF(0.0f, 0.85f, 0.15f, 0.15f));
    auto * axis_clear = new Qt3DRender::QClearBuffers(m_axis_viewport);
    axis_clear->setBuffers(Qt3DRender::QClearBuffers::DepthBuffer);
    auto * axis_camera_selector = new Qt3DRender::QCameraSelector(axis_clear);
    axis_camera_selector->setCamera(m_axis_camera);
    auto * axis_filter = new Qt3DRender::QLayerFilter(axis_camera_selector);
    axis_filter->addLayer(m_axis_layer);
    axis_filter->setFilterMode(Qt3DRender::QLayerFilter::AcceptAnyMatchingLayers);
    make_state(axis_filter);

    m_view->setActiveFrameGraph(surface_selector);

    // The gizmo follows every main-camera move to stay in sync.
    Qt3DRender::QCamera * const cam = m_view->camera();
    connect(cam, &Qt3DRender::QCamera::positionChanged, this, [this]
            { updateAxisCamera(); });
    connect(cam, &Qt3DRender::QCamera::viewCenterChanged, this, [this]
            { updateAxisCamera(); });
    connect(cam, &Qt3DRender::QCamera::upVectorChanged, this, [this]
            { updateAxisCamera(); });

    updateAxisViewport();
    updateAxisCamera();
}

void R3DWidget::updateAxisViewport()
{
    if (nullptr == m_axis_viewport)
    {
        return;
    }
    constexpr float size = 160.0f; // gizmo edge length in pixels
    constexpr float margin = 12.0f; // gap from the window edges in pixels
    float const w = static_cast<float>(m_view->width());
    float const h = static_cast<float>(m_view->height());
    if (w <= 0.0f || h <= 0.0f)
    {
        return;
    }
    // Qt3D origin at upper-left with y growing downward, so the lower-left
    // corner is near (0, 1).
    m_axis_viewport->setNormalizedRect(QRectF(
        margin / w, (h - size - margin) / h, size / w, size / h));
}

void R3DWidget::updateAxisCamera()
{
    if (nullptr == m_axis_camera)
    {
        return;
    }
    constexpr float dist = 10.0f;
    QVector3D const view_dir = camera()->viewVector().normalized();
    m_axis_camera->setViewCenter(QVector3D(0.0f, 0.0f, 0.0f));
    m_axis_camera->setPosition(-view_dir * dist);
    m_axis_camera->setUpVector(camera()->upVector());
}

void R3DWidget::showMark()
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RAxisMark))
        {
            child->deleteLater();
        }
    }
    new RAxisMark(m_scene, m_axis_layer);
}

void R3DWidget::updateMesh(std::shared_ptr<StaticMesh> const & mesh)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RStaticMesh) || typeid(*child) == typeid(RBoundary))
        {
            child->deleteLater();
        }
    }
    new RStaticMesh(mesh, m_scene);
    m_mesh = mesh;
}

void R3DWidget::showMesh(bool show)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RStaticMesh))
        {
            child->setEnabled(show);
        }
    }
}

void R3DWidget::showBoundary(int ibc, bool show)
{
    // Remove an existing boundary highlight for this set so a re-show stays
    // single and a hide simply leaves none behind.
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RBoundary) && static_cast<RBoundary *>(child)->ibc() == ibc)
        {
            child->deleteLater();
        }
    }
    if (show && nullptr != m_mesh)
    {
        new RBoundary(m_mesh, ibc, m_scene);
    }
}

void R3DWidget::updateWorld(std::shared_ptr<WorldFp64> const & world)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if ((typeid(*child) == typeid(RLines)) || (typeid(*child) == typeid(RVertices)))
        {
            child->deleteLater();
        }
    }
    new RVertices(world, m_scene);
    new RLines(world, m_scene);

    fitCameraToScene();
}

void R3DWidget::updateColorField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices)
{
    for (Qt3DCore::QNode * child : m_scene->childNodes())
    {
        if (typeid(*child) == typeid(RColorField))
        {
            child->deleteLater();
        }
    }
    new RColorField(vertices, colors, indices, m_scene);

    fitCameraToScene();
}

void R3DWidget::fitCameraToScene()
{
    QVector3D box_min_pt = m_scene->minPoint(); // get the bottom-left corner of bounding box
    QVector3D box_max_pt = m_scene->maxPoint(); // get the top-right corner of bounding box
    QVector3D box_center = (box_min_pt + box_max_pt) * 0.5f; // center point of the bounding box

    /*
     * Calculate the camera distance to fully view the bounding box based on
     * the vertical field of view (FOV) and the vertical extent of the box.
     *
     * Using the relation: tan(fov / 2) = (half of vertical size) / (distance to center),
     * we can rearrange and solve for the distance:
     *
     *     (distance to center) = (half of vertical size) / tan(fov / 2)
     *
     * This gives the minimum distance from the box center along the view direction within
     * the specified FOV. After that, set the far plane properly to ensurce enclose whole
     * bounding box.
     */
    float fov = 45.0f;
    float half_height = (box_max_pt - box_min_pt).length() * 0.5f;
    float dist = half_height / std::tan(qDegreesToRadians(fov) / 2.0f);

    cameraController()->setPosition(box_center + QVector3D(0, 0, dist));
    cameraController()->setViewCenter(box_center);
    cameraController()->setFarPlane(dist + half_height * 2);
}

void R3DWidget::resizeEvent(QResizeEvent * event)
{
    QWidget::resizeEvent(event);
    m_view->resize(event->size());
    m_container->resize(event->size());
    updateAxisViewport();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
