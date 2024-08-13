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

#include <modmesh/view/RCameraController.hpp>

#include <QShortcut>
#include <Qt3DInput/QKeyboardDevice>
#include <Qt3DInput/QMouseDevice>
#include <Qt3DRender/QCamera>

#include "R3DWidget.hpp"

namespace modmesh
{

RCameraInputListener::RCameraInputListener(
    Qt3DInput::QKeyboardDevice * keyboardDevice,
    Qt3DInput::QMouseDevice * mouseDevice,
    callback_type callback,
    QNode * parent)
    : QEntity(parent)
    , m_frame_action(new Qt3DLogic::QFrameAction)
    , m_logical_device(new Qt3DInput::QLogicalDevice)
    , m_keyboard_device(keyboardDevice)
    , m_mouse_device(mouseDevice)
    , m_callback(std::move(callback))
    , m_rx_axis(new Qt3DInput::QAxis)
    , m_ry_axis(new Qt3DInput::QAxis)
    , m_tx_axis(new Qt3DInput::QAxis)
    , m_ty_axis(new Qt3DInput::QAxis)
    , m_tz_axis(new Qt3DInput::QAxis)
    , m_left_mouse_button_action(new Qt3DInput::QAction)
    , m_middle_mouse_button_action(new Qt3DInput::QAction)
    , m_right_mouse_button_action(new Qt3DInput::QAction)
    , m_shift_button_action(new Qt3DInput::QAction)
    , m_alt_button_action(new Qt3DInput::QAction)
    , m_ctrl_button_action(new Qt3DInput::QAction)
    , m_left_mouse_button_input(new Qt3DInput::QActionInput)
    , m_middle_mouse_button_input(new Qt3DInput::QActionInput)
    , m_right_mouse_button_input(new Qt3DInput::QActionInput)
    , m_shift_button_input(new Qt3DInput::QActionInput)
    , m_alt_button_input(new Qt3DInput::QActionInput)
    , m_ctrl_button_input(new Qt3DInput::QActionInput)
    , m_mouse_rx_input(new Qt3DInput::QAnalogAxisInput)
    , m_mouse_ry_input(new Qt3DInput::QAnalogAxisInput)
    , m_mouse_tz_x_input(new Qt3DInput::QAnalogAxisInput)
    , m_mouse_tz_y_input(new Qt3DInput::QAnalogAxisInput)
    , m_keyboard_tx_pos_input(new Qt3DInput::QButtonAxisInput)
    , m_keyboard_ty_pos_input(new Qt3DInput::QButtonAxisInput)
    , m_keyboard_tz_pos_input(new Qt3DInput::QButtonAxisInput)
    , m_keyboard_tx_neg_input(new Qt3DInput::QButtonAxisInput)
    , m_keyboard_ty_neg_input(new Qt3DInput::QButtonAxisInput)
    , m_keyboard_tz_neg_input(new Qt3DInput::QButtonAxisInput)
{
    init();

    connect(
        m_frame_action,
        &Qt3DLogic::QFrameAction::triggered,
        this,
        [this](const float dt)
        {
            CameraInputState state{};
            const bool isCtrlPressed = m_ctrl_button_action->isActive();

            state.rxAxisValue = m_rx_axis->value();
            state.ryAxisValue = m_ry_axis->value();
            state.txAxisValue = m_tx_axis->value();
            state.tyAxisValue = isCtrlPressed ? 0.0f : m_ty_axis->value();
            state.tzAxisValue = isCtrlPressed ? m_ty_axis->value() : 0.0f;

            state.leftMouseButtonActive = m_left_mouse_button_action->isActive();
            state.middleMouseButtonActive = m_middle_mouse_button_action->isActive();
            state.rightMouseButtonActive = m_right_mouse_button_action->isActive();

            state.altKeyActive = m_alt_button_action->isActive();
            state.shiftKeyActive = m_shift_button_action->isActive();

            m_callback(state, dt);
        });
}

void RCameraInputListener::init() {
    initMouseListeners();
    initKeyboardListeners();

    m_logical_device->addAction(m_left_mouse_button_action);
    m_logical_device->addAction(m_middle_mouse_button_action);
    m_logical_device->addAction(m_right_mouse_button_action);
    m_logical_device->addAction(m_alt_button_action);
    m_logical_device->addAction(m_shift_button_action);
    m_logical_device->addAction(m_ctrl_button_action);

    m_logical_device->addAxis(m_rx_axis);
    m_logical_device->addAxis(m_ry_axis);
    m_logical_device->addAxis(m_tx_axis);
    m_logical_device->addAxis(m_ty_axis);
    m_logical_device->addAxis(m_tz_axis);

    connect(this, &QEntity::enabledChanged, m_logical_device, &Qt3DInput::QLogicalDevice::setEnabled);
    connect(this, &QEntity::enabledChanged, m_frame_action, &Qt3DLogic::QFrameAction::setEnabled);

    for (const auto axis : {m_rx_axis, m_ry_axis, m_tx_axis, m_ty_axis, m_tz_axis})
    {
        connect(this, &QEntity::enabledChanged, axis, &Qt3DInput::QAxis::setEnabled);
    }

    addComponent(m_frame_action);
    addComponent(m_logical_device);
}

void RCameraInputListener::initMouseListeners() const
{
    // left mouse button
    m_left_mouse_button_input->setButtons(QList<int>{Qt::LeftButton});
    m_left_mouse_button_input->setSourceDevice(m_mouse_device);
    m_left_mouse_button_action->addInput(m_left_mouse_button_input);

    // middle mouse button
    m_middle_mouse_button_input->setButtons(QList<int>{Qt::MiddleButton});
    m_middle_mouse_button_input->setSourceDevice(m_mouse_device);
    m_middle_mouse_button_action->addInput(m_middle_mouse_button_input);

    // right mouse button
    m_right_mouse_button_input->setButtons(QList<int>{Qt::RightButton});
    m_right_mouse_button_input->setSourceDevice(m_mouse_device);
    m_right_mouse_button_action->addInput(m_right_mouse_button_input);

    // mouse x rotation - reads mouse movement in x direction
    m_mouse_rx_input->setAxis(Qt3DInput::QMouseDevice::X);
    m_mouse_rx_input->setSourceDevice(m_mouse_device);
    m_rx_axis->addInput(m_mouse_rx_input);

    // mouse y rotation - reads mouse movement in y direction
    m_mouse_ry_input->setAxis(Qt3DInput::QMouseDevice::Y);
    m_mouse_ry_input->setSourceDevice(m_mouse_device);
    m_ry_axis->addInput(m_mouse_ry_input);

    // mouse x translation - need for mouses with trackball
    m_mouse_tz_x_input->setAxis(Qt3DInput::QMouseDevice::WheelX);
    m_mouse_tz_x_input->setSourceDevice(m_mouse_device);
    m_tz_axis->addInput(m_mouse_tz_x_input);

    // mouse z translation - mouse wheel
    m_mouse_tz_y_input->setAxis(Qt3DInput::QMouseDevice::WheelY);
    m_mouse_tz_y_input->setSourceDevice(m_mouse_device);
    m_tz_axis->addInput(m_mouse_tz_y_input);
}

void RCameraInputListener::initKeyboardListeners() const
{
    // shift button
    m_shift_button_input->setButtons(QList<int>{Qt::Key_Shift});
    m_shift_button_input->setSourceDevice(m_keyboard_device);
    m_shift_button_action->addInput(m_shift_button_input);

    // alt button
    m_alt_button_input->setButtons(QList<int>{Qt::Key_Alt});
    m_alt_button_input->setSourceDevice(m_keyboard_device);
    m_alt_button_action->addInput(m_alt_button_input);

    // ctrl button
    m_ctrl_button_input->setButtons(QList<int>{Qt::Key_Control});
    m_ctrl_button_input->setSourceDevice(m_keyboard_device);
    m_ctrl_button_action->addInput(m_ctrl_button_input);

    // keyboard positive x translation
    m_keyboard_tx_pos_input->setButtons(QList<int>{Qt::Key_D, Qt::Key_Right});
    m_keyboard_tx_pos_input->setScale(1.0f);
    m_keyboard_tx_pos_input->setSourceDevice(m_keyboard_device);
    m_tx_axis->addInput(m_keyboard_tx_pos_input);

    // keyboard positive y translation
    m_keyboard_ty_pos_input->setButtons(QList<int>{Qt::Key_W, Qt::Key_Up});
    m_keyboard_ty_pos_input->setScale(1.0f);
    m_keyboard_ty_pos_input->setSourceDevice(m_keyboard_device);
    m_ty_axis->addInput(m_keyboard_ty_pos_input);

    // keyboard negative x translation
    m_keyboard_tx_neg_input->setButtons(QList<int>{Qt::Key_A, Qt::Key_Left});
    m_keyboard_tx_neg_input->setScale(-1.0f);
    m_keyboard_tx_neg_input->setSourceDevice(m_keyboard_device);
    m_tx_axis->addInput(m_keyboard_tx_neg_input);

    // keyboard negative y translation
    m_keyboard_ty_neg_input->setButtons(QList<int>{Qt::Key_S, Qt::Key_Down});
    m_keyboard_ty_neg_input->setScale(-1.0f);
    m_keyboard_ty_neg_input->setSourceDevice(m_keyboard_device);
    m_ty_axis->addInput(m_keyboard_ty_neg_input);
}

RFirstPersonCameraController::RFirstPersonCameraController(QNode * parent)
    : QFirstPersonCameraController(parent)
{
    auto callback = [this](const CameraInputState & state, const float dt)
    {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

void RFirstPersonCameraController::updateCameraPosition(const CameraInputState & input, const float dt)
{
    constexpr auto positiveY = QVector3D(0.f, 1.f, 0.f);

    if (camera() == nullptr)
        return;

    const auto translationInput = QVector3D(input.txAxisValue, input.tyAxisValue, input.tzAxisValue);

    camera()->translate(translationInput * linearSpeed() * dt);

    if (input.leftMouseButtonActive)
    {
        const float adjustedLookSpeed = lookSpeed() * (input.shiftKeyActive ? lookSpeedFactorOnShiftPressed : 1.0f);

        camera()->pan(input.rxAxisValue * adjustedLookSpeed * dt, positiveY);
        camera()->tilt(input.ryAxisValue * adjustedLookSpeed * dt);
    }
}

ROrbitCameraController::ROrbitCameraController(QNode * parent)
    : QOrbitCameraController(parent)
{
    auto callback = [this](const CameraInputState & state, const float dt)
    {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

void ROrbitCameraController::updateCameraPosition(const CameraInputState & input, const float dt)
{
    if (camera() == nullptr)
        return;

    // mouse input
    if (input.leftMouseButtonActive)
    {
        if (input.rightMouseButtonActive)
        {
            zoom(input.tzAxisValue);
        }
        else
        {
            const auto translation = QVector3D(
                clamp(input.rxAxisValue + input.txAxisValue),
                clamp(input.ryAxisValue + input.tyAxisValue),
                0);

            camera()->translate(translation * linearSpeed() * dt);
        }

        return;
    }

    if (input.rightMouseButtonActive)
    {
        orbit(input.rxAxisValue * dt, input.ryAxisValue * dt);
    }

    // keyboard Input
    if (input.altKeyActive)
    {
        orbit(input.txAxisValue * dt, input.tzAxisValue * dt);
    }
    else if (input.shiftKeyActive)
    {
        zoom(input.tzAxisValue * linearSpeed() * dt);
    }
    else
    {
        const float x = clamp(input.txAxisValue + (input.leftMouseButtonActive ? input.rxAxisValue : 0));
        const float y = clamp(input.tyAxisValue + (input.leftMouseButtonActive ? input.ryAxisValue : 0));
        const auto translation = QVector3D(x, y, input.tzAxisValue) * linearSpeed() * dt;

        camera()->translate(translation);
    }
}

void ROrbitCameraController::zoom(const float zoomValue) const
{
    const float limitSquared = zoomInLimit() * zoomInLimit();
    const float distanceSquared = zoomDistanceSquared(camera()->position(), camera()->viewCenter());

    const float z = distanceSquared > limitSquared ? zoomValue : -0.5f;

    camera()->translate(QVector3D(0, 0, z), Qt3DRender::QCamera::DontTranslateViewCenter);
}

void ROrbitCameraController::orbit(const float pan, const float tilt) const
{
    constexpr auto positiveY = QVector3D(0.f, 1.f, 0.f);

    camera()->panAboutViewCenter(pan * lookSpeed(), positiveY);
    camera()->tiltAboutViewCenter(tilt * lookSpeed());
}

float ROrbitCameraController::clamp(const float value)
{
    return std::min(std::max(value, -1.f), 1.f);
}

float ROrbitCameraController::zoomDistanceSquared(const QVector3D firstPoint, const QVector3D secondPoint)
{
    const QVector3D vector = secondPoint - firstPoint;

    return vector.lengthSquared();
}

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
