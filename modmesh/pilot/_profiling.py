from PySide6 import QtCore, QtWidgets

from ._gui_common import PilotFeature

__all__ = [
    'Profiling'
]


class Profiling(PilotFeature):
    """
    Create profiling windows.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        # self._diag.setDirectory(self._get_initial_path())
        self._diag.setWindowTitle('Open profiling file')

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.profilingMenu,
            text="Open profiling result",
            tip="Open JSON file of profiling result",
            func=self.open_profiling_result,
        )

        self._add_menu_item(
            menu=self._mgr.profilingMenu,
            text="View profiling dashboard",
            tip="View the dashboard of profiling result",
            func=self.profiling_dashboard,
        )

    def open_profiling_result(self):
        self._diag.open(self, QtCore.SLOT('on_finished()'))

    def profiling_dashboard(self):
        pass

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)
        print("Open file: ", filenames)
