from PUI.PySide6 import *

class ParametersView(PuiInQt):
    class TableAdapter:
        def __init__(self, state, data):
            self.state = state
            self._data = data

        def data(self, row, col):
            param = self._data[row]
            return [param.key, param.value][col]

        def setData(self, row, col, value):
            if col==1:
                self._data[row].value = value

        def editable(self, row, col):
            return col > 0

        def columnHeader(self, col):
            return ["Key","Value"][col]

        def rowCount(self):
            return len(self._data)

        def columnCount(self):
            return 2

    def __init__(self, container, state):
        super().__init__(container)
        self.state = state

    def content(self):
        data = State()
        if not self.state.filter:
            data.params = self.state.params
        else:
            data.params = [it for it in self.state.params if self.state.filter in it.key]
        with VBox():
            TextField(self.state("filter"))
            Table(self.TableAdapter(self.state, data.params))

def openParametersView(params):
    state  = State()
    state.buffer = [f"line {i}" for i in range(50)]
    state.cmd_edit = ""
    state.config_modal = False
    state.filter = ""
    state.params = params
    pv = ParametersView(Window(size=(640, 480)), state)
    pv.redraw()

if __name__=="__main__":
    class Example():
        def __init__(self):
            self.app = QtWidgets.QApplication([])

        def run(self):
            openParametersView()
            self.app.exec()

    root = Example()
    root.run()
