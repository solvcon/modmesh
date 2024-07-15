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
#include <QVector3D>

#include <Qt3DLogic/QFrameAction>
#include <Qt3DInput/QActionInput>
#include <Qt3DInput/QAction>
#include <Qt3DInput/QAnalogAxisInput>
#include <Qt3DInput/QButtonAxisInput>
#include <Qt3DInput/QAxis>
#include <Qt3DInput/QLogicalDevice>

namespace modmesh {

class RCameraInputListener : public Qt3DCore::QEntity
{
    Q_OBJECT

public:
    using callback_type = std::function<void (const Qt3DExtras::QAbstractCameraController::InputState&, float)>;

    RCameraInputListener(
        Qt3DInput::QKeyboardDevice* keyboardDevice,
        Qt3DInput::QMouseDevice* mouseDevice,
        callback_type callback,
        QNode *parent = nullptr
    );

private:

    Qt3DLogic::QFrameAction *m_frameAction;
    Qt3DInput::QLogicalDevice *m_logicalDevice;

    Qt3DInput::QKeyboardDevice* m_keyboardDevice;
    Qt3DInput::QMouseDevice* m_mouseDevice;

    // invoked each frame to update the camera position
    callback_type m_callback;

    // rotation
    Qt3DInput::QAxis *m_rxAxis;
    Qt3DInput::QAxis *m_ryAxis;

    // translation
    Qt3DInput::QAxis *m_txAxis;
    Qt3DInput::QAxis *m_tyAxis;
    Qt3DInput::QAxis *m_tzAxis;

    Qt3DInput::QAction *m_leftMouseButtonAction;
    Qt3DInput::QAction *m_middleMouseButtonAction;
    Qt3DInput::QAction *m_rightMouseButtonAction;

    Qt3DInput::QAction *m_shiftButtonAction;
    Qt3DInput::QAction *m_altButtonAction;

    Qt3DInput::QActionInput *m_leftMouseButtonInput;
    Qt3DInput::QActionInput *m_middleMouseButtonInput;
    Qt3DInput::QActionInput *m_rightMouseButtonInput;

    Qt3DInput::QActionInput *m_shiftButtonInput;
    Qt3DInput::QActionInput *m_altButtonInput;

    // mouse rotation input
    Qt3DInput::QAnalogAxisInput *m_mouseRxInput;
    Qt3DInput::QAnalogAxisInput *m_mouseRyInput;

    // mouse translation input (wheel)
    Qt3DInput::QAnalogAxisInput *m_mouseTzXInput;
    Qt3DInput::QAnalogAxisInput *m_mouseTzYInput;

    // keyboard translation input
    Qt3DInput::QButtonAxisInput *m_keyboardTxPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTyPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTzPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTxNegInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTyNegInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTzNegInput;

    void init();

    void initMouseListeners() const;

    void initKeyboardListeners() const;
};


class CameraControllerMixin
{
public:
    virtual ~CameraControllerMixin() = default;

    virtual void updateCameraPosition(const Qt3DExtras::QAbstractCameraController::InputState &state, float dt) = 0;

protected:
    RCameraInputListener* m_listener = nullptr;
};


class RFirstPersonCameraController : public Qt3DExtras::QFirstPersonCameraController, public CameraControllerMixin
{
    Q_OBJECT

public:
    explicit RFirstPersonCameraController(QNode *parent = nullptr);

private:
    static constexpr auto upVector = QVector3D(0.f, 1.f, 0.f);
    static constexpr auto lookSpeedFactorOnShiftPressed = 0.2f;

    void moveCamera(const InputState &state, float dt) override {}

    void updateCameraPosition(const InputState &input, float dt) override;
};


class ROrbitCameraController : public Qt3DExtras::QOrbitCameraController, public CameraControllerMixin
{
    Q_OBJECT

public:
    explicit ROrbitCameraController(QNode *parent = nullptr);

private:
    void moveCamera(const InputState &state, float dt) override {}

    void updateCameraPosition(const InputState &input, float dt) override;

    void zoom(float zoomValue) const;

    void orbit(float pan, float tilt) const;

    static float clamp(float value);

    static float zoomDistanceSquared(QVector3D firstPoint, QVector3D secondPoint);
};

}