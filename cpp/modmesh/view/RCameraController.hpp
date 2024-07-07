#pragma once

#include <modmesh/view/common_detail.hpp> // Must be the first include.

#include <QOrbitCameraController>
#include <QFirstPersonCameraController>

#include <Qt3DLogic/QFrameAction>
#include <Qt3DInput/QActionInput>
#include <Qt3DInput/QAction>
#include <Qt3DInput/QAnalogAxisInput>
#include <Qt3DInput/QButtonAxisInput>
#include <Qt3DInput/QAxis>
#include <Qt3DInput/QLogicalDevice>

namespace modmesh {

/*
 * CameraInputListener
 */
class RCameraInputListener : public Qt3DCore::QEntity
{
    Q_OBJECT


public:
    using callback_type = std::function<void (const Qt3DExtras::QAbstractCameraController::InputState&, float)>;

    RCameraInputListener(Qt3DInput::QKeyboardDevice* keyboardDevice, Qt3DInput::QMouseDevice* mouseDevice,
        callback_type callback, QNode *parent = nullptr);

private:
    Qt3DLogic::QFrameAction *m_frameAction;
    Qt3DInput::QLogicalDevice *m_logicalDevice;

    Qt3DInput::QKeyboardDevice* m_keyboardDevice;
    Qt3DInput::QMouseDevice* m_mouseDevice;
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

    Qt3DInput::QAnalogAxisInput *m_mouseRxInput;
    Qt3DInput::QAnalogAxisInput *m_mouseRyInput;
    Qt3DInput::QAnalogAxisInput *m_mouseTzXInput;
    Qt3DInput::QAnalogAxisInput *m_mouseTzYInput;

    Qt3DInput::QButtonAxisInput *m_keyboardTxPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTyPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTzPosInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTxNegInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTyNegInput;
    Qt3DInput::QButtonAxisInput *m_keyboardTzNegInput;

    void init();
};


class CameraControllerMixin
{
public:
    virtual ~CameraControllerMixin() = default;

    virtual void updateCameraPosition(const Qt3DExtras::QAbstractCameraController::InputState &state, float dt) = 0;

protected:
    RCameraInputListener* m_listener = nullptr;
};

/*
 * FirstPersonCameraController
 */
class RFirstPersonCameraController : public Qt3DExtras::QFirstPersonCameraController, public CameraControllerMixin
{
    Q_OBJECT

public:
    explicit RFirstPersonCameraController(QNode *parent = nullptr);

private:
    void moveCamera(const InputState &state, float dt) override {}

    void updateCameraPosition(const InputState &state, float dt) override;
};

/*
 * OrbitCameraController
 */
class ROrbitCameraController : public Qt3DExtras::QOrbitCameraController, public CameraControllerMixin
{
    Q_OBJECT

public:
    explicit ROrbitCameraController(QNode *parent = nullptr);

private:
    void moveCamera(const InputState &state, float dt) override {}

    void updateCameraPosition(const InputState &state, float dt) override;
};

}