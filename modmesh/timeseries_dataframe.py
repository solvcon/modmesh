# Copyright (c) 2024, Zong-han, Xie <zonghanxie@proton.me>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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


import os
import numpy as np

from .core import SimpleArrayUint64, SimpleArrayFloat64


__all__ = [
    'TimeSeriesDataFrame'
]


class TimeSeriesDataFrame(object):

    def __init__(self):

        self._columns = list()
        self._index_column = None
        self._index_column_name = None
        self._data = list()

    def read_from_text_file(
        self,
        txt_path,
        delimiter=',',
        timestamp_in_file=True,
        timestamp_column=None
    ):
        """
        Generate dataframe from a text file.

        :param txt_path: path to the text file.
        :type txt_path: str
        :param delimiter: delimiter.
        :type delimiter: str
        :param timestamp_in_file: If the text file containing index column,
        expected to be integer.
        :type timestamp_in_file: bool
        :prarm timestamp_column: Column which stores timestamp data.
        :type timestamp_column: str
        :return None
        """
        if not os.path.exists(txt_path):
            raise Exception("Text file '{}' does not exist".format(txt_path))

        nd_arr = np.genfromtxt(txt_path, delimiter=delimiter)[1:]
        index_column_num = 0 if timestamp_in_file else None

        with open(txt_path, 'r') as f:
            table_header = [x for x in f.readline().strip().split(delimiter)]
            if timestamp_in_file:
                if timestamp_column in table_header:
                    index_column_num = table_header.index(timestamp_column)
                self._index_column = SimpleArrayUint64(
                    array=nd_arr[:, index_column_num].astype(np.uint64)
                )
                self._index_column_name = table_header[index_column_num]
            else:
                self._index_column = SimpleArrayUint64(
                    array=np.arange(nd_arr.shape[0]).astype(np.uint64)
                )
                self._index_column_name = "Index"

            self._columns = table_header
            if index_column_num is not None:
                self._columns.pop(index_column_num)

        for i in range(nd_arr.shape[1]):
            if i != index_column_num:
                self._data.append(
                    SimpleArrayFloat64(array=nd_arr[:, i].copy())
                )

    def __getitem__(self, column_name):
        if column_name not in self._columns:
            raise Exception("Column '{}' does not exist".format(column_name))
        return self._data[self._columns.index(column_name)].ndarray

    @property
    def columns(self):
        return self._columns

    @property
    def shape(self):
        return (self._index_column.ndarray.shape[0], len(self._data))

    @property
    def index(self):
        return self._index_column.ndarray
