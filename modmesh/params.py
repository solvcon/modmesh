#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2023, Buganini Chiu <buganini@b612.tw>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the pstake nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import PUI
from PUI.PySide6 import State, PuiInQt, Window, VBox, Table, TextField
from PySide6 import QtWidgets

if tuple([int(x) for x in PUI.__version__.split(".")]) != (0, 3):
    raise ValueError("PUI too old")


class ParameterView(PuiInQt):
    class TableAdapter:
        def __init__(self, state, data):
            self.state = state
            self._data = data

        def data(self, row, col):
            param = self._data[row]
            return [param.key, param.value][col]

        def setData(self, row, col, value):
            if col == 1:
                self._data[row].value = value

        def editable(self, row, col):
            return col > 0

        def columnHeader(self, col):
            return ["Key", "Value"][col]

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
            data.params = [
                it
                for it in self.state.params
                if self.state.filter in it.key
            ]
        with VBox():
            TextField(self.state("filter"))
            Table(self.TableAdapter(self.state, data.params))


def openParameterView(params):
    state = State()
    state.filter = ""
    state.params = params
    pv = ParameterView(Window(size=(640, 480)), state)
    pv.redraw()


if __name__ == "__main__":
    class Example():
        def __init__(self):
            self.app = QtWidgets.QApplication([])

        def run(self):
            from PUI.PySide6 import StateDict
            paramsState = StateDict()
            paramsState["a.b.foo_int64"] = 5566
            paramsState["a.b.bar_double"] = 77.88
            paramsList = [
                paramsState("a.b.foo_int64"),
                paramsState("a.b.bar_double"),
            ]
            openParameterView(paramsList)
            self.app.exec()

    root = Example()
    root.run()
