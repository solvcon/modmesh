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

#include <Qt3DInput/QKeyboardDevice>
#include <Qt3DInput/QMouseDevice>
#include <Qt3DRender/QCamera>

namespace modmesh {

RCameraInputListener::RCameraInputListener(
    Qt3DInput::QKeyboardDevice* keyboardDevice,
    Qt3DInput::QMouseDevice* mouseDevice,
    callback_type callback,
    QNode * parent
)
    : QEntity(parent)
    , m_frameAction(new Qt3DLogic::QFrameAction)
    , m_logicalDevice(new Qt3DInput::QLogicalDevice)

    , m_keyboardDevice(keyboardDevice)
    , m_mouseDevice(mouseDevice)
    , m_callback(std::move(callback))

    , m_rxAxis(new Qt3DInput::QAxis)
    , m_ryAxis(new Qt3DInput::QAxis)
    , m_txAxis(new Qt3DInput::QAxis)
    , m_tyAxis(new Qt3DInput::QAxis)
    , m_tzAxis(new Qt3DInput::QAxis)

    , m_leftMouseButtonAction(new Qt3DInput::QAction)
    , m_middleMouseButtonAction(new Qt3DInput::QAction)
    , m_rightMouseButtonAction(new Qt3DInput::QAction)

    , m_shiftButtonAction(new Qt3DInput::QAction)
    , m_altButtonAction(new Qt3DInput::QAction)

    , m_leftMouseButtonInput(new Qt3DInput::QActionInput)
    , m_middleMouseButtonInput(new Qt3DInput::QActionInput)
    , m_rightMouseButtonInput(new Qt3DInput::QActionInput)

    , m_shiftButtonInput(new Qt3DInput::QActionInput)
    , m_altButtonInput(new Qt3DInput::QActionInput)

    , m_mouseRxInput(new Qt3DInput::QAnalogAxisInput)
    , m_mouseRyInput(new Qt3DInput::QAnalogAxisInput)
    , m_mouseTzXInput(new Qt3DInput::QAnalogAxisInput)
    , m_mouseTzYInput(new Qt3DInput::QAnalogAxisInput)

    , m_keyboardTxPosInput(new Qt3DInput::QButtonAxisInput)
    , m_keyboardTyPosInput(new Qt3DInput::QButtonAxisInput)
    , m_keyboardTzPosInput(new Qt3DInput::QButtonAxisInput)
    , m_keyboardTxNegInput(new Qt3DInput::QButtonAxisInput)
    , m_keyboardTyNegInput(new Qt3DInput::QButtonAxisInput)
    , m_keyboardTzNegInput(new Qt3DInput::QButtonAxisInput)
{
    init();

    connect(m_frameAction, &Qt3DLogic::QFrameAction::triggered,
    this, [this] (const float dt) {
        Qt3DExtras::QAbstractCameraController::InputState state{};

        state.rxAxisValue = m_rxAxis->value();
        state.ryAxisValue = m_ryAxis->value();
        state.txAxisValue = m_txAxis->value();
        state.tyAxisValue = m_tyAxis->value();
        state.tzAxisValue = m_tzAxis->value();

        state.leftMouseButtonActive = m_leftMouseButtonAction->isActive();
        state.middleMouseButtonActive = m_middleMouseButtonAction->isActive();
        state.rightMouseButtonActive = m_rightMouseButtonAction->isActive();

        state.altKeyActive = m_altButtonAction->isActive();
        state.shiftKeyActive = m_shiftButtonAction->isActive();

        m_callback(state, dt);
    });
}

void RCameraInputListener::init()
{
    initMouseListeners();
    initKeyboardListeners();

    m_logicalDevice->addAction(m_leftMouseButtonAction);
    m_logicalDevice->addAction(m_middleMouseButtonAction);
    m_logicalDevice->addAction(m_rightMouseButtonAction);
    m_logicalDevice->addAction(m_altButtonAction);
    m_logicalDevice->addAction(m_shiftButtonAction);

    m_logicalDevice->addAxis(m_rxAxis);
    m_logicalDevice->addAxis(m_ryAxis);
    m_logicalDevice->addAxis(m_txAxis);
    m_logicalDevice->addAxis(m_tyAxis);
    m_logicalDevice->addAxis(m_tzAxis);

    connect(this, &QEntity::enabledChanged, m_logicalDevice, &Qt3DInput::QLogicalDevice::setEnabled);
    connect(this, &QEntity::enabledChanged, m_frameAction, &Qt3DLogic::QFrameAction::setEnabled);

    for (const auto axis: {m_rxAxis, m_ryAxis, m_txAxis, m_tyAxis, m_tzAxis})
    {
        connect(this, &QEntity::enabledChanged, axis, &Qt3DInput::QAxis::setEnabled);
    }

    addComponent(m_frameAction);
    addComponent(m_logicalDevice);
}

void RCameraInputListener::initMouseListeners() const {
    // left mouse button
    m_leftMouseButtonInput->setButtons(QList<int> { Qt::LeftButton });
    m_leftMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_leftMouseButtonAction->addInput(m_leftMouseButtonInput);

    // middle mouse button
    m_middleMouseButtonInput->setButtons(QList<int> { Qt::MiddleButton });
    m_middleMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_middleMouseButtonAction->addInput(m_middleMouseButtonInput);

    // right mouse button
    m_rightMouseButtonInput->setButtons(QList<int> { Qt::RightButton });
    m_rightMouseButtonInput->setSourceDevice(m_mouseDevice);
    m_rightMouseButtonAction->addInput(m_rightMouseButtonInput);

    // mouse x rotation - reads mouse movement in x direction
    m_mouseRxInput->setAxis(Qt3DInput::QMouseDevice::X);
    m_mouseRxInput->setSourceDevice(m_mouseDevice);
    m_rxAxis->addInput(m_mouseRxInput);

    // mouse y rotation - reads mouse movement in y direction
    m_mouseRyInput->setAxis(Qt3DInput::QMouseDevice::Y);
    m_mouseRyInput->setSourceDevice(m_mouseDevice);
    m_ryAxis->addInput(m_mouseRyInput);

    // mouse x translation - need for mouses with trackball
    m_mouseTzXInput->setAxis(Qt3DInput::QMouseDevice::WheelX);
    m_mouseTzXInput->setSourceDevice(m_mouseDevice);
    m_tzAxis->addInput(m_mouseTzXInput);

    // mouse z translation - mouse wheel
    m_mouseTzYInput->setAxis(Qt3DInput::QMouseDevice::WheelY);
    m_mouseTzYInput->setSourceDevice(m_mouseDevice);
    m_tzAxis->addInput(m_mouseTzYInput);
}

void RCameraInputListener::initKeyboardListeners() const {
    // shift button
    m_shiftButtonInput->setButtons(QList<int> { Qt::Key_Shift });
    m_shiftButtonInput->setSourceDevice(m_keyboardDevice);
    m_shiftButtonAction->addInput(m_shiftButtonInput);

    // alt button
    m_altButtonInput->setButtons(QList<int> { Qt::Key_Alt });
    m_altButtonInput->setSourceDevice(m_keyboardDevice);
    m_altButtonAction->addInput(m_altButtonInput);

    // keyboard positive x translation
    m_keyboardTxPosInput->setButtons(QList<int> { Qt::Key_D, Qt::Key_Right });
    m_keyboardTxPosInput->setScale(1.0f);
    m_keyboardTxPosInput->setSourceDevice(m_keyboardDevice);
    m_txAxis->addInput(m_keyboardTxPosInput);

    // keyboard positive y translation
    m_keyboardTyPosInput->setButtons(QList<int> { Qt::Key_E, Qt::Key_PageUp });
    m_keyboardTyPosInput->setScale(1.0f);
    m_keyboardTyPosInput->setSourceDevice(m_keyboardDevice);
    m_tyAxis->addInput(m_keyboardTyPosInput);

    // keyboard positive z translation
    m_keyboardTzPosInput->setButtons(QList<int> { Qt::Key_W, Qt::Key_Up });
    m_keyboardTzPosInput->setScale(1.0f);
    m_keyboardTzPosInput->setSourceDevice(m_keyboardDevice);
    m_tzAxis->addInput(m_keyboardTzPosInput);

    // keyboard negative x translation
    m_keyboardTxNegInput->setButtons(QList<int> { Qt::Key_A, Qt::Key_Left });
    m_keyboardTxNegInput->setScale(-1.0f);
    m_keyboardTxNegInput->setSourceDevice(m_keyboardDevice);
    m_txAxis->addInput(m_keyboardTxNegInput);
    
    // keyboard negative y translation
    m_keyboardTyNegInput->setButtons(QList<int> { Qt::Key_Q, Qt::Key_PageDown });
    m_keyboardTyNegInput->setScale(-1.0f);
    m_keyboardTyNegInput->setSourceDevice(m_keyboardDevice);
    m_tyAxis->addInput(m_keyboardTyNegInput);

    // keyboard negative z translation
    m_keyboardTzNegInput->setButtons(QList<int> { Qt::Key_S, Qt::Key_Down });
    m_keyboardTzNegInput->setScale(-1.0f);
    m_keyboardTzNegInput->setSourceDevice(m_keyboardDevice);
    m_tzAxis->addInput(m_keyboardTzNegInput);
}


RFirstPersonCameraController::RFirstPersonCameraController(QNode * parent) : QFirstPersonCameraController(parent)
{
    auto callback = [this] (const InputState &state, const float dt) {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

void RFirstPersonCameraController::updateCameraPosition(const InputState &input, const float dt)
{
    Qt3DRender::QCamera *_camera = camera();

    if (_camera == nullptr)
        return;

    const auto translationInput = QVector3D(input.txAxisValue, input.tyAxisValue, input.tzAxisValue);

    _camera->translate(translationInput * linearSpeed() * dt);

    if (input.leftMouseButtonActive)
    {
        const float _lookSpeed = lookSpeed() * (input.shiftKeyActive ? lookSpeedFactorOnShiftPressed : 1.0f);

        _camera->pan(input.rxAxisValue * _lookSpeed * dt, upVector);
        _camera->tilt(input.ryAxisValue * _lookSpeed * dt);
    }
}


ROrbitCameraController::ROrbitCameraController(QNode * parent) : QOrbitCameraController(parent)
{
    auto callback = [this] (const InputState &state, const float dt) {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

void ROrbitCameraController::updateCameraPosition(const InputState& input, const float dt)
{
    Qt3DRender::QCamera *_camera = camera();
    const float _linearSpeed = linearSpeed();

    if (_camera == nullptr)
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
            // translate
            const auto inversion = QVector3D(
                inverseXTranslate() ? -1.f : 1.f,
                inverseYTranslate() ? -1.f : 1.f,
                1.f
            );
            const auto translation = QVector3D(
                clamp(input.rxAxisValue + input.txAxisValue),
                clamp(input.ryAxisValue + input.tyAxisValue),
                0
            );

            _camera->translate(inversion * translation * _linearSpeed * dt);
        }

        return;
    }

    if (input.rightMouseButtonActive)
    {
        orbit(
            (inversePan() ? -1.0f : 1.0f) * input.rxAxisValue * dt,
            (inverseTilt() ? -1.0f : 1.0f) * input.ryAxisValue * dt
        );
    }

    // keyboard Input
    if (input.altKeyActive)
    {
        orbit(input.txAxisValue * dt, input.tzAxisValue * dt);
    }
    else if (input.shiftKeyActive)
    {
        zoom(input.tzAxisValue * _linearSpeed * dt);
    }
    else
    {
        // translate
        const float x = clamp(input.txAxisValue + (input.leftMouseButtonActive ? input.rxAxisValue : 0));
        const float y = clamp(input.tyAxisValue + (input.leftMouseButtonActive ? input.ryAxisValue : 0));
        const auto translation = QVector3D(x, y, input.tzAxisValue) * _linearSpeed * dt;

        const auto option = zoomTranslateViewCenter() ? Qt3DRender::QCamera::TranslateViewCenter : Qt3DRender::QCamera::DontTranslateViewCenter;

        _camera->translate(translation, option);
    }
}

void ROrbitCameraController::zoom(const float zoomValue) const {
    const float _zoomInLimitSquared = zoomInLimit() * zoomInLimit();
    const float _zoomDistanceSquared = zoomDistanceSquared(camera()->position(), camera()->viewCenter());

    const float z = _zoomDistanceSquared > _zoomInLimitSquared ? zoomValue : -0.5f;

    camera()->translate(QVector3D(0, 0, z), Qt3DRender::QCamera::DontTranslateViewCenter);
}

void ROrbitCameraController::orbit(const float pan, const float tilt) const {
    camera()->panAboutViewCenter(pan * lookSpeed(), upVector());
    camera()->tiltAboutViewCenter(tilt * lookSpeed());
}

float ROrbitCameraController::clamp(const float value) {
    return std::min(std::max(value, -1.f), 1.f);
}

float ROrbitCameraController::zoomDistanceSquared(const QVector3D firstPoint, const QVector3D secondPoint)
{
    const QVector3D vector = secondPoint - firstPoint;

    return vector.lengthSquared();
}

}
