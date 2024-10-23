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
import unittest

import numpy as np

from modmesh import TimeSeriesDataFrame


class TimeSeriesDataFrameTC(unittest.TestCase):
    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")
    columns_sol = [
        'DATA_DELTA_VEL[1] ', 'DATA_DELTA_VEL[2] ',
        'DATA_DELTA_VEL[3] ', 'DATA_DELTA_ANGLE[1] ',
        'DATA_DELTA_ANGLE[2] ', 'DATA_DELTA_ANGLE[3]'
    ]

    columns_sol2 = [
        'DATA_DELTA_VEL[1] ', 'DATA_DELTA_VEL[2] ',
        'TIME_NANOSECONDS_TAI ', 'DATA_DELTA_VEL[3] ',
        'DATA_DELTA_ANGLE[1] ', 'DATA_DELTA_ANGLE[2] ',
        'DATA_DELTA_ANGLE[3]'
    ]

    def test_read_from_text_file_basic(self):
        tsdf = TimeSeriesDataFrame()

        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed.csv")
        )
        self.assertEqual(tsdf._columns, self.columns_sol)
        self.assertEqual(len(tsdf._columns), 6)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_column_name, 'TIME_NANOSECONDS_TAI ')

        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed_header_changed.csv"),
            delimiter=',',
            timestamp_column='TIME_NANOSECONDS_TAI '
        )

        self.assertEqual(tsdf._columns, self.columns_sol)
        self.assertEqual(len(tsdf._columns), 6)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_column_name, 'TIME_NANOSECONDS_TAI ')

        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed_header_changed.csv"),
            delimiter=',',
            timestamp_in_file=False
        )
        self.assertEqual(tsdf._columns, self.columns_sol2)
        self.assertEqual(len(tsdf._columns), 7)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_column_name, 'Index')

    def test_dataframe_attribute_columns(self):
        tsdf = TimeSeriesDataFrame()
        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed.csv")
        )
        self.assertEqual(tsdf.columns, self.columns_sol)

    def test_dataframe_attribute_shape(self):
        tsdf = TimeSeriesDataFrame()
        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed.csv")
        )
        self.assertEqual(tsdf.shape, (10, 6))

    def test_dataframe_attribute_index(self):
        tsdf = TimeSeriesDataFrame()
        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed.csv")
        )

        nd_arr = np.genfromtxt(
            os.path.join(self.DATADIR, "dlc_trimmed.csv"), delimiter=','
        )[1:]

        self.assertEqual(
            list(tsdf.index), list(nd_arr[:, 0].astype(np.uint64))
        )

    def test_dataframe_get_column(self):
        tsdf = TimeSeriesDataFrame()
        tsdf.read_from_text_file(
            os.path.join(self.DATADIR, "dlc_trimmed.csv")
        )

        one_column_data = tsdf['DATA_DELTA_VEL[1] ']

        nd_arr = np.genfromtxt(
            os.path.join(self.DATADIR, "dlc_trimmed.csv"), delimiter=','
        )[1:]

        self.assertEqual(list(one_column_data), list(nd_arr[:, 1]))
