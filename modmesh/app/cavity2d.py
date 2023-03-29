__package__ = "modmesh.app"

import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from .. import view


def load_app():
    view.mgr.pycon.writeToHistory("""
# Use the functions for more examples:
ctrl.start()  # Start computing. Multiple blank windows will appear, and the
results will be displayed once the execution is terminated.
<ctrl> + C # Terminate in terminal


ctrl = mm.app.cavity2d.run(interval=10, max_steps=50, profiling=True)
""")
    cmd = "ctrl = mm.app.cavity2d.run(interval=10, max_steps=50)"
    view.mgr.pycon.command = cmd


###############################################################################

class var:
    N = 32
    dt = 0.001
    tol_p = 1e2
    tol_v = 1e-10
    Re = 100


class grid:
    N = var.N
    dx = 1.0 / (N - 1)
    dy = 1.0 / (N - 1)
    P = np.zeros((N + 1, N + 1))
    U = np.zeros((N + 1, N + 1))
    V = np.zeros((N + 1, N + 1))
    U_1 = np.zeros((N + 1, N + 1))
    V_1 = np.zeros((N + 1, N + 1))
    U_2 = np.zeros((N + 1, N + 1))
    V_2 = np.zeros((N + 1, N + 1))


class prev:
    N = grid.N
    U = np.zeros((N + 1, N + 1))
    V = np.zeros((N + 1, N + 1))
    P = np.zeros((N + 1, N + 1))


class col:
    N = grid.N
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    P = np.zeros((N, N))


# benchmark
class benchmark:
    N = var.N
    x_b = [0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000]
    x_b = x_b + [0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]
    y_b = [0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000]
    y_b = y_b + [0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
    Re_100_v = [0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507]
    Re_100_v = Re_100_v + [0.17527, 0.05454, -0.24533, -0.22445, -0.16914]
    Re_100_v = Re_100_v + [-0.10313, -0.08864, -0.07391, -0.05906, 0]
    Re_100_u = [0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662]
    Re_100_u = Re_100_u + [-0.21090, -0.20581, -0.13641, 0.00332, 0.23151]
    Re_100_u = Re_100_u + [0.68717, 0.73722, 0.78871, 0.84123, 1]
    pt = np.zeros(N)
    for i in range(1, N):
        pt[i] = (i - 1) / (N - 1)


def meet_poisson():
    residual = 0
    for i in range(1, grid.N):
        for j in range(1, grid.N):
            ux = (grid.U_2[i, j] - grid.U_2[i - 1, j]) / grid.dx
            vy = (grid.V_2[i, j] - grid.V_2[i, j - 1]) / grid.dy
            poisson_LHP = (ux + vy) / var.dt
            poisson_RHP = (grid.P[i + 1, j] + grid.P[i - 1, j] + grid.P[i, j + 1] + grid.P[i, j - 1] - 4 * grid.P[i, j]) / (grid.dx * grid.dx)  # noqa: E501
            residual += abs(poisson_LHP - poisson_RHP)

    # clear_output(wait=True)
    log(f"[meet Poisson]residual:  {residual}")
    if residual < var.tol_p:
        return True

    return False


def log(msg):
    sys.stdout.write(msg)
    sys.stdout.write('\n')
    view.mgr.pycon.writeToHistory(msg)
    view.mgr.pycon.writeToHistory('\n')


def is_steady():
    vt = 0
    for i in range(1, grid.N):
        for j in range(1, grid.N):
            vt += abs(prev.U[i, j] - grid.U[i, j])
            vt += abs(prev.V[i, j] - grid.V[i, j])
            prev.U[i, j] = grid.U[i, j]
            prev.V[i, j] = grid.V[i, j]

    log(f"[is_steady] velocity deviation: {vt}")

    if vt < var.tol_v:
        return True

    return False


def collocate():
    for i in range(grid.N):
        for j in range(grid.N):
            col.U[i, j] = 0.5 * (grid.U[i, j] + grid.U[i, j + 1])
            col.V[i, j] = 0.5 * (grid.V[i, j] + grid.V[i + 1, j])
            col.P[i, j] = (grid.P[i, j] + grid.P[i + 1, j] + grid.P[i, j + 1] + grid.P[i + 1, j + 1]) * 0.25  # noqa: E501


def moniter(timestep):
    #   clear_output(wait=True)
    bm = benchmark()
    print("Timestep: ", timestep)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    U_mid = np.transpose(col.U)[:, int((grid.N + 1) / 2)]
    V_mid = np.transpose(col.V)[int((grid.N + 1) / 2)]
    velocity = np.transpose(np.sqrt(np.multiply(col.U, col.U) + np.multiply(col.V, col.V)))  # noqa: E501
    ax[0].plot(bm.Re_100_u, bm.y_b, 'bo', U_mid, bm.pt)
    ax[0].set(ylim=(0, 1))
    ax[1].plot(bm.x_b, bm.Re_100_v, 'bo', bm.pt, V_mid)
    ax[1].set(xlim=(0, 1))
    ax[2].imshow(velocity)
    ax[2].invert_yaxis()
    plt.show()


# Computation-1
def setBC(option):
    N = grid.N
    if option == 'P':
        grid.P[0, :] = grid.P[1, :]  # west
        grid.P[N, :] = grid.P[N - 1, :]  # east
        grid.P[:, 0] = grid.P[:, 1]  # south
        grid.P[:, N] = grid.P[:, N - 1]  # north

    elif option == 'U':
        grid.U[:, N] = 2 - grid.U[:, N - 1]  # north
        grid.U[:, 0] = - grid.U[:, 1]  # south
        grid.U[0, :] = 0  # west
        grid.U[N - 1, :] = 0  # east

        grid.U_1[:, N] = 2 - grid.U_1[:, N - 1]  # north
        grid.U_1[:, 0] = - grid.U_1[:, 1]  # south
        grid.U_1[0, :] = 0  # west
        grid.U_1[N-1, :] = 0  # east

        grid.U_2[:, N] = 2 - grid.U_2[:, N - 1]  # north
        grid.U_2[:, 0] = - grid.U_2[:, 1]  # south
        grid.U_2[0, :] = 0  # west
        grid.U_2[N - 1, :] = 0  # east

    elif option == 'V':
        grid.V[0, :] = - grid.V[1, :]  # west
        grid.V[N, :] = - grid.V[N - 1, :]  # east
        grid.V[:, 0] = 0  # south
        grid.V[:, N - 1] = 0  # north

        grid.V_1[0, :] = - grid.V_1[1, :]  # west
        grid.V_1[N, :] = - grid.V_1[N - 1, :]  # east
        grid.V_1[:, 0] = 0  # south
        grid.V_1[:, N - 1] = 0  # north

        grid.V_2[0, :] = - grid.V_2[1, :]  # west
        grid.V_2[N, :] = - grid.V_2[N - 1, :]  # east
        grid.V_2[:, 0] = 0  # south
        grid.V_2[:, N-1] = 0  # north


def solve_U1():
    for i in range(1, grid.N - 1):
        for j in range(1, grid.N):
            u = grid.U[i, j]
            v = (grid.V[i, j] + grid.V[i + 1, j] + grid.V[i, j - 1] + grid.V[i + 1, j - 1]) / 4  # noqa: E501
            ux = (grid.U[i + 1, j] - grid.U[i - 1, j]) / (2 * grid.dx)
            uy = (grid.U[i, j + 1] - grid.U[i, j - 1]) / (2 * grid.dy)
            u2x = (grid.U[i + 1, j] + grid.U[i - 1, j] - 2 * grid.U[i, j]) / (grid.dx * grid.dx)  # noqa: E501
            u2y = (grid.U[i, j + 1] + grid.U[i, j - 1] - 2 * grid.U[i, j]) / (grid.dy * grid.dy)  # noqa: E501

            C = u * ux + v * uy
            D = (u2x + u2y) / var.Re

            px = (grid.P[i + 1, j] - grid.P[i, j]) / grid.dx
            grid.U_1[i, j] = (- C + D - px) * var.dt + grid.U[i, j]


def solve_V1():
    for i in range(1, grid.N):
        for j in range(1, grid.N - 1):
            u = (grid.U[i - 1, j + 1] + grid.U[i, j + 1] + grid.U[i - 1, j] + grid.U[i, j]) / 4.0  # noqa: E501
            v = grid.V[i, j]
            vx = (grid.V[i + 1, j] - grid.V[i - 1, j]) / (2 * grid.dx)
            vy = (grid.V[i, j + 1] - grid.V[i, j - 1]) / (2 * grid.dy)
            v2x = (grid.V[i + 1, j] + grid.V[i - 1, j] - 2 * grid.V[i, j]) / (grid.dx * grid.dx)  # noqa: E501
            v2y = (grid.V[i, j + 1] + grid.V[i, j - 1] - 2 * grid.V[i, j]) / (grid.dy * grid.dy)  # noqa: E501

            C = u * vx + v * vy
            D = (v2x + v2y) / var.Re

            py = (grid.P[i, j + 1] - grid.P[i, j]) / grid.dy
            grid.V_1[i][j] = (- C + D - py) * var.dt + grid.V[i, j]


def solve_U2():
    for i in range(1, grid.N - 1):
        for j in range(1, grid.N):
            px = (grid.P[i + 1, j] - grid.P[i, j]) / grid.dx
            grid.U_2[i, j] = px * var.dt + grid.U_1[i, j]


def solve_V2():
    for i in range(1, grid.N):
        for j in range(1, grid.N - 1):
            py = (grid.P[i, j + 1] - grid.P[i, j]) / grid.dy
            grid.V_2[i, j] = py * var.dt + grid.V_1[i, j]


def solve_P():
    iteration = 0
    while not meet_poisson():
        iteration = iteration + 1
        # clear_output(wait=True)
        # print("[solve_P]iteration: ", iteration)
        for i in range(1, grid.N):
            for j in range(1, grid.N):
                ux = (grid.U_2[i, j] - grid.U_2[i - 1, j]) / grid.dx
                vy = (grid.V_2[i, j] - grid.V_2[i, j - 1]) / grid.dy
                poisson_LHP = (ux + vy) / var.dt
                grid.P[i, j] = 0.25 * (grid.P[i + 1, j] + grid.P[i - 1, j] + grid.P[i, j + 1] + grid.P[i, j - 1] - poisson_LHP * grid.dx * grid.dx)  # noqa: E501
    meet_poisson()


def solve_U():
    for i in range(1, grid.N - 1):
        for j in range(1, grid.N):
            px = (grid.P[i + 1, j] - grid.P[i, j]) / grid.dx
            grid.U[i, j] = - px * var.dt + grid.U_2[i, j]


def solve_V():
    for i in range(1, grid.N):
        for j in range(1, grid.N-1):
            py = (grid.P[i, j+1] - grid.P[i, j]) / grid.dy
            grid.V[i, j] = - py * var.dt + grid.V_2[i, j]


# Computation-2
def set_BC(option=3):
    if option == 0:
        setBC('P')

    elif option == 1:
        setBC('U')

    elif option == 2:
        setBC('V')

    elif option == 3:
        setBC('U')
        setBC('V')
        setBC('P')


def step_1(option):
    if option == 1:
        solve_U1()
    elif option == 2:
        solve_V1()


def step_2(option):
    if option == 1:
        solve_U2()
    elif option == 2:
        solve_V2()


def step_3(option):
    solve_P()


def step_4(option):
    if option == 1:
        solve_U()
    elif option == 2:
        solve_V()


def projection():
    timestep = 0
    set_BC()
    while ((timestep == 0) | (is_steady() == False)):  # noqa: E712

        timestep += 1

        step_1(1)
        step_1(2)

        step_2(1)
        step_2(2)

        set_BC(1)
        set_BC(2)

        step_3(0)
        set_BC(0)

        step_4(1)
        step_4(2)
        set_BC(1)
        set_BC(2)

        collocate()
        moniter(timestep)

# projection()
###############################################################################


class Plot:
    def __init__(self, figsize=(5, 5), N=31):
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.canvas.figure.subplots()
        test = np.random.rand(31, 31)
        self.ax.imshow(test)


class Controller:
    def __init__(self, max_steps, use_sub=None, profiling=False):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.timer = None

        self._main = QtWidgets.QWidget()

        self.plt = Plot(figsize=(10, 10))
        layout = QtWidgets.QVBoxLayout(self._main)
        layout.addWidget(self.plt.canvas)
        layout.addWidget(NavigationToolbar(self.plt.canvas, self._main))

        self.profiling = profiling

    def show(self):
        self._main.show()

    def start(self):
        # self.timer.start()
        projection()

    def stop(self):
        self.timer.stop()


def run(interval=10, max_steps=50, no_view_mgr=False, **kw):
    ctrl = Controller(max_steps=max_steps, **kw)
    ctrl.show()

    return ctrl


if __name__ == "__main__":
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication()

        ctrl = run(interval=10, max_steps=50, no_view_mgr=True, profiling=True)
        ctrl.start()

        # The trick to close the event loop of app automatically
        # The timer will emit a closeAllWindows event after 20 seconds
        # after the app is executed.
        QTimer.singleShot(20000, app.closeAllWindows)

        sys.exit(app.exec())

    except ImportError:
        print("Something wrong when importing PySide6.")
        print("Do you install PySide6?")
        sys.exit(1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

###############################################################################
