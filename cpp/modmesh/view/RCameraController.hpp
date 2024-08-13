#pragma once

/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/view/common_detail.hpp> // Must be the first include.

#include <QOrbitCameraController>
#include <QFirstPersonCameraController>
#include <QCamera>

#include <Qt3DLogic/QFrameAction>
#include <Qt3DInput/QActionInput>
#include <Qt3DInput/QAction>
#include <Qt3DInput/QAnalogAxisInput>
#include <Qt3DInput/QButtonAxisInput>
#include <Qt3DInput/QAxis>
#include <Qt3DInput/QLogicalDevice>

namespace modmesh
{

struct CameraInputState
{
    float rxAxisValue = 0;
    float ryAxisValue = 0;
    float txAxisValue = 0;
    float tyAxisValue = 0;
    float tzAxisValue = 0;

    bool leftMouseButtonActive = false;
    bool middleMouseButtonActive = false;
    bool rightMouseButtonActive = false;

    bool altKeyActive = false;
    bool shiftKeyActive = false;
};

enum class CameraControllerType
{
    FirstPerson,
    Orbit
};

class RCameraInputListener : public Qt3DCore::QEntity
{
    Q_OBJECT

public:
    using callback_type = std::function<void(const CameraInputState &, float)>;

    RCameraInputListener(
        Qt3DInput::QKeyboardDevice * keyboardDevice,
        Qt3DInput::QMouseDevice * mouseDevice,
        callback_type callback,
        QNode * parent = nullptr);

private:

    void init();

    void initMouseListeners() const;

    void initKeyboardListeners() const;

    Qt3DLogic::QFrameAction * m_frame_action;
    Qt3DInput::QLogicalDevice * m_logical_device;
    Qt3DInput::QKeyboardDevice * m_keyboard_device;
    Qt3DInput::QMouseDevice * m_mouse_device;
    // invoked each frame to update the camera position
    callback_type m_callback;
    // rotation
    Qt3DInput::QAxis * m_rx_axis;
    Qt3DInput::QAxis * m_ry_axis;
    // translation
    Qt3DInput::QAxis * m_tx_axis;
    Qt3DInput::QAxis * m_ty_axis;
    Qt3DInput::QAxis * m_tz_axis;
    Qt3DInput::QAction * m_left_mouse_button_action;
    Qt3DInput::QAction * m_middle_mouse_button_action;
    Qt3DInput::QAction * m_right_mouse_button_action;
    Qt3DInput::QAction * m_shift_button_action;
    Qt3DInput::QAction * m_alt_button_action;
    Qt3DInput::QAction * m_ctrl_button_action;
    Qt3DInput::QActionInput * m_left_mouse_button_input;
    Qt3DInput::QActionInput * m_middle_mouse_button_input;
    Qt3DInput::QActionInput * m_right_mouse_button_input;
    Qt3DInput::QActionInput * m_shift_button_input;
    Qt3DInput::QActionInput * m_alt_button_input;
    Qt3DInput::QActionInput * m_ctrl_button_input;
    // mouse rotation input
    Qt3DInput::QAnalogAxisInput * m_mouse_rx_input;
    Qt3DInput::QAnalogAxisInput * m_mouse_ry_input;
    // mouse translation input (wheel)
    Qt3DInput::QAnalogAxisInput * m_mouse_tz_x_input;
    Qt3DInput::QAnalogAxisInput * m_mouse_tz_y_input;
    // keyboard translation input
    Qt3DInput::QButtonAxisInput * m_keyboard_tx_pos_input;
    Qt3DInput::QButtonAxisInput * m_keyboard_ty_pos_input;
    Qt3DInput::QButtonAxisInput * m_keyboard_tz_pos_input;
    Qt3DInput::QButtonAxisInput * m_keyboard_tx_neg_input;
    Qt3DInput::QButtonAxisInput * m_keyboard_ty_neg_input;
    Qt3DInput::QButtonAxisInput * m_keyboard_tz_neg_input;
};

class CameraController
{
public:
    virtual ~CameraController() = default;

    virtual void updateCameraPosition(const CameraInputState & state, float dt) = 0;

    virtual Qt3DRender::QCamera * getCamera() = 0;

    virtual float getLinearSpeed() = 0;

    virtual float getLookSpeed() = 0;

    virtual CameraControllerType getType() = 0;

    QVector3D position() { return getCamera()->position(); }

    QVector3D viewVector() { return getCamera()->viewVector(); }

    QVector3D viewCenter() { return getCamera()->viewCenter(); }

    QVector3D upVector() { return getCamera()->upVector(); }

protected:
    RCameraInputListener * m_listener = nullptr;

private:
    Qt3DExtras::QAbstractCameraController * asQtCameraController()
    {
        return dynamic_cast<Qt3DExtras::QAbstractCameraController *>(this);
    }
};

class RFirstPersonCameraController : public Qt3DExtras::QFirstPersonCameraController
    , public CameraController
{
    Q_OBJECT

public:
    explicit RFirstPersonCameraController(QNode * parent = nullptr);

    Qt3DRender::QCamera * getCamera() override { return camera(); }
    float getLinearSpeed() override { return linearSpeed(); }
    float getLookSpeed() override { return lookSpeed(); }

private:
    static constexpr auto lookSpeedFactorOnShiftPressed = 0.2f;

    void moveCamera(const InputState & state, float dt) override {}

    void updateCameraPosition(const CameraInputState & input, float dt) override;

    CameraControllerType getType() override { return CameraControllerType::FirstPerson; }
};

class ROrbitCameraController : public Qt3DExtras::QOrbitCameraController
    , public CameraController
{
    Q_OBJECT

public:
    explicit ROrbitCameraController(QNode * parent = nullptr);

    Qt3DRender::QCamera * getCamera() override { return camera(); }
    float getLinearSpeed() override { return linearSpeed(); }
    float getLookSpeed() override { return lookSpeed(); }

private:
    void moveCamera(const InputState & state, float dt) override {}

    void updateCameraPosition(const CameraInputState & input, float dt) override;

    void zoom(float zoomValue) const;

    void orbit(float pan, float tilt) const;

    static float clamp(float value);

    static float zoomDistanceSquared(QVector3D firstPoint, QVector3D secondPoint);

    CameraControllerType getType() override { return CameraControllerType::Orbit; }
};

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
