
#include <modmesh/view/RCameraController.hpp>

#include <Qt3DInput/QKeyboardDevice>
#include <Qt3DInput/QMouseDevice>
#include <Qt3DRender/QCamera>

namespace modmesh {

/*
 * CameraInputListener
 */

RCameraInputListener::RCameraInputListener(
    Qt3DInput::QKeyboardDevice* keyboardDevice, Qt3DInput::QMouseDevice* mouseDevice,
    callback_type callback, QNode * parent)
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

    // shift button
    m_shiftButtonInput->setButtons(QList<int> { Qt::Key_Shift });
    m_shiftButtonInput->setSourceDevice(m_keyboardDevice);
    m_shiftButtonAction->addInput(m_shiftButtonInput);

    //// Axes

    // mouse x rotation
    m_mouseRxInput->setAxis(Qt3DInput::QMouseDevice::X);
    m_mouseRxInput->setSourceDevice(m_mouseDevice);
    m_rxAxis->addInput(m_mouseRxInput);

    // mouse y rotation
    m_mouseRyInput->setAxis(Qt3DInput::QMouseDevice::Y);
    m_mouseRyInput->setSourceDevice(m_mouseDevice);
    m_ryAxis->addInput(m_mouseRyInput);

    // mouse x translation
    m_mouseTzXInput->setAxis(Qt3DInput::QMouseDevice::WheelX);
    m_mouseTzXInput->setSourceDevice(m_mouseDevice);
    m_tzAxis->addInput(m_mouseTzXInput);

    // mouse z translation
    m_mouseTzYInput->setAxis(Qt3DInput::QMouseDevice::WheelY);
    m_mouseTzYInput->setSourceDevice(m_mouseDevice);
    m_tzAxis->addInput(m_mouseTzYInput);

    // keyboard positive x translation
    m_keyboardTxPosInput->setButtons(QList<int> { Qt::Key_D });
    m_keyboardTxPosInput->setScale(1.0f);
    m_keyboardTxPosInput->setSourceDevice(m_keyboardDevice);
    m_txAxis->addInput(m_keyboardTxPosInput);

    // keyboard positive z translation
    m_keyboardTzPosInput->setButtons(QList<int> { Qt::Key_W });
    m_keyboardTzPosInput->setScale(1.0f);
    m_keyboardTzPosInput->setSourceDevice(m_keyboardDevice);
    m_tzAxis->addInput(m_keyboardTzPosInput);

    // keyboard positive y translation
    m_keyboardTyPosInput->setButtons(QList<int> { Qt::Key_E });
    m_keyboardTyPosInput->setScale(1.0f);
    m_keyboardTyPosInput->setSourceDevice(m_keyboardDevice);
    m_tyAxis->addInput(m_keyboardTyPosInput);

    // keyboard negative x translation
    m_keyboardTxNegInput->setButtons(QList<int> { Qt::Key_A });
    m_keyboardTxNegInput->setScale(-1.0f);
    m_keyboardTxNegInput->setSourceDevice(m_keyboardDevice);
    m_txAxis->addInput(m_keyboardTxNegInput);

    // keyboard negative z translation
    m_keyboardTzNegInput->setButtons(QList<int> { Qt::Key_S });
    m_keyboardTzNegInput->setScale(-1.0f);
    m_keyboardTzNegInput->setSourceDevice(m_keyboardDevice);
    m_tzAxis->addInput(m_keyboardTzNegInput);

    // keyboard negative y translation
    m_keyboardTyNegInput->setButtons(QList<int> { Qt::Key_Q });
    m_keyboardTyNegInput->setScale(-1.0f);
    m_keyboardTyNegInput->setSourceDevice(m_keyboardDevice);
    m_tyAxis->addInput(m_keyboardTyNegInput);

    // logical device
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

    connect(this, &QEntity::enabledChanged,
            m_logicalDevice, &Qt3DInput::QLogicalDevice::setEnabled);
    connect(this, &QEntity::enabledChanged,
            m_frameAction, &Qt3DLogic::QFrameAction::setEnabled);

    for (const auto axis: {m_rxAxis, m_ryAxis, m_txAxis, m_tyAxis, m_tzAxis}) {
        connect(this, &QEntity::enabledChanged,
                         axis, &Qt3DInput::QAxis::setEnabled);
    }

    addComponent(m_frameAction);
    addComponent(m_logicalDevice);
}


/*
 * FirstPersonCameraController
 */

RFirstPersonCameraController::RFirstPersonCameraController(QNode * parent) : QFirstPersonCameraController(parent) {
    auto callback = [this] (const InputState &state, const float dt) {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

void RFirstPersonCameraController::updateCameraPosition(const InputState &state, const float dt)
{
    Qt3DRender::QCamera *theCamera = camera();

    if (theCamera == nullptr)
        return;

    theCamera->translate(QVector3D(state.txAxisValue * linearSpeed(),
                                  state.tyAxisValue * linearSpeed(),
                                  state.tzAxisValue * linearSpeed()) * dt);
    
    if (state.leftMouseButtonActive)
    {
        float theLookSpeed = lookSpeed();
        
        if (state.shiftKeyActive)
        {
            theLookSpeed *= 0.2f;
        }

        const QVector3D upVector(0.0f, 1.0f, 0.0f);

        theCamera->pan(state.rxAxisValue * theLookSpeed * dt, upVector);
        theCamera->tilt(state.ryAxisValue * theLookSpeed * dt);
    }
}

/*
 * OrbitCameraController
 */

ROrbitCameraController::ROrbitCameraController(QNode * parent) : QOrbitCameraController(parent) {
    auto callback = [this] (const InputState &state, const float dt) {
        updateCameraPosition(state, dt);
    };

    m_listener = new RCameraInputListener(keyboardDevice(), mouseDevice(), callback, this);
}

inline float clampInputs(float input1, float input2)
{
    float axisValue = input1 + input2;
    return (axisValue < -1) ? -1 : (axisValue > 1) ? 1 : axisValue;
}

inline float zoomDistance(QVector3D firstPoint, QVector3D secondPoint)
{
    return (secondPoint - firstPoint).lengthSquared();
}

void ROrbitCameraController::updateCameraPosition(const InputState& state, const float dt) {
    Qt3DRender::QCamera *_camera = camera();
    const float _zoomInLimit = zoomInLimit();
    const float _lookSpeed = lookSpeed();
    const float _linearSpeed = linearSpeed();

    if (_camera == nullptr)
        return;

    // Mouse input
    if (state.leftMouseButtonActive)
    {
        if (state.rightMouseButtonActive)
        {
            if ( zoomDistance(camera()->position(), _camera->viewCenter()) > _zoomInLimit * _zoomInLimit)
            {
                // Dolly up to limit
                _camera->translate(QVector3D(0, 0, state.ryAxisValue), _camera->DontTranslateViewCenter);
            }
            else
            {
                _camera->translate(QVector3D(0, 0, -0.5), _camera->DontTranslateViewCenter);
            }
        } else
        {
            // Translate
            _camera->translate(QVector3D((inverseXTranslate() ? -1.0f : 1.0f) * clampInputs(state.rxAxisValue, state.txAxisValue) * linearSpeed(),
                                           (inverseYTranslate() ? -1.0f : 1.0f) * clampInputs(state.ryAxisValue, state.tyAxisValue) * linearSpeed(),
                                           0) * dt);
        }
        return;
    }
    else if (state.rightMouseButtonActive)
    {
        // Orbit
        _camera->panAboutViewCenter((inversePan() ? -1.0f : 1.0f) * (state.rxAxisValue * _lookSpeed) * dt, upVector());
        _camera->tiltAboutViewCenter((inverseTilt() ? -1.0f : 1.0f) * (state.ryAxisValue * _lookSpeed) * dt);
    }

    // Keyboard Input
    if (state.altKeyActive)
    {
        // Orbit
        _camera->panAboutViewCenter((state.txAxisValue * _lookSpeed) * dt, upVector());
        _camera->tiltAboutViewCenter((state.tyAxisValue * _lookSpeed) * dt);
    }
    else if (state.shiftKeyActive)
    {
        if (zoomDistance(camera()->position(), _camera->viewCenter()) > _zoomInLimit * _zoomInLimit)
        {
            // Dolly
            _camera->translate(QVector3D(0, 0, state.tzAxisValue * linearSpeed() * dt), _camera->DontTranslateViewCenter);
        }
        else
        {
            _camera->translate(QVector3D(0, 0, -0.5), _camera->DontTranslateViewCenter);
        }
    }
    else
    {
        // Translate
        _camera->translate(QVector3D(clampInputs(state.leftMouseButtonActive ? state.rxAxisValue : 0, state.txAxisValue) * _linearSpeed,
                                      clampInputs(state.leftMouseButtonActive ? state.ryAxisValue : 0, state.tyAxisValue) * _linearSpeed,
                                      state.tzAxisValue * _linearSpeed) * dt,
                             zoomTranslateViewCenter() ? _camera->TranslateViewCenter : _camera->DontTranslateViewCenter);
    }
}

}
