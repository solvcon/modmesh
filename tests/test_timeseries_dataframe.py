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

import io
import unittest

import numpy as np

from modmesh.track import dataframe


class TimeSeriesDataFrameTC(unittest.TestCase):

    col_sol = ['DELTA_VEL[1]', 'DELTA_VEL[2]', 'DELTA_VEL[3]']
    col_sol2 = ['DELTA_VEL[1]', 'DELTA_VEL[2]', 'EPOCH', 'DELTA_VEL[3]']

    dlc_data = """EPOCH ,DELTA_VEL[1] ,DELTA_VEL[2] ,DELTA_VEL[3]
1.6025960102293e+18,-0.18792724609375,-0.00048828125,-0.0478515625
1.60259601024931e+18,-0.1903076171875,-0.0009765625,-0.0489501953125
1.60259601026931e+18,-0.18743896484375,0.0006103515625,-0.0498046875
1.60259601028932e+18,-0.18927001953125,-0.0009765625,-0.04840087890625
1.60259601030931e+18,-0.188720703125,-0.00103759765625,-0.0504150390625
1.60259601032931e+18,-0.18951416015625,-0.000732421875,-0.0489501953125
1.60259601034931e+18,-0.18902587890625,-0.000732421875,-0.0489501953125
1.6025960103693e+18,-0.1895751953125,-0.00128173828125,-0.04925537109375
1.60259601038931e+18,-0.18841552734375,6.103515625e-05,-0.0489501953125
1.60259601040931e+18,-0.1884765625,-0.00042724609375,-0.04840087890625
"""
    modified_dlc_data = """DELTA_VEL[1] ,DELTA_VEL[2] ,EPOCH ,DELTA_VEL[3]
-0.18792724609375,-0.00048828125,1602596010229299968,-0.0478515625
-0.1903076171875,-0.0009765625,1602596010249309952,-0.0489501953125
-0.18743896484375,0.0006103515625,1602596010269309952,-0.0498046875
-0.18927001953125,-0.0009765625,1602596010289319936,-0.04840087890625
-0.188720703125,-0.00103759765625,1602596010309309952,-0.0504150390625
-0.18951416015625,-0.000732421875,1602596010329309952,-0.0489501953125
-0.18902587890625,-0.000732421875,1602596010349309952,-0.0489501953125
-0.1895751953125,-0.00128173828125,1602596010369299968,-0.04925537109375
-0.18841552734375,6.103515625e-05,1602596010389309952,-0.0489501953125
-0.1884765625,-0.00042724609375,1602596010409309952,-0.04840087890625
"""
    unsorted_dlc_data = """EPOCH ,DELTA_VEL[1] ,DELTA_VEL[2] ,DELTA_VEL[3]
1.60259601024931e+18,-0.1903076171875,-0.0009765625,-0.0489501953125
1.60259601034931e+18,-0.18902587890625,-0.000732421875,-0.0489501953125
1.60259601040931e+18,-0.1884765625,-0.00042724609375,-0.04840087890625
1.60259601032931e+18,-0.18951416015625,-0.000732421875,-0.0489501953125
1.6025960102293e+18,-0.18792724609375,-0.00048828125,-0.0478515625
1.6025960103693e+18,-0.1895751953125,-0.00128173828125,-0.04925537109375
1.60259601030931e+18,-0.188720703125,-0.00103759765625,-0.0504150390625
1.60259601026931e+18,-0.18743896484375,0.0006103515625,-0.0498046875
1.60259601028932e+18,-0.18927001953125,-0.0009765625,-0.04840087890625
1.60259601038931e+18,-0.18841552734375,6.103515625e-05,-0.0489501953125
"""

    def test_read_from_text_file_basic(self):
        tsdf = dataframe.DataFrame()

        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf._columns, self.col_sol)
        self.assertEqual(len(tsdf._columns), 3)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'EPOCH')

        tsdf.read_from_text_file(
            io.StringIO(self.modified_dlc_data),
            delimiter=',',
            timestamp_column='EPOCH'
        )

        self.assertEqual(tsdf._columns, self.col_sol)
        self.assertEqual(len(tsdf._columns), 3)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'EPOCH')

        tsdf.read_from_text_file(
            io.StringIO(self.modified_dlc_data),
            delimiter=',',
            timestamp_in_file=False
        )
        self.assertEqual(tsdf._columns, self.col_sol2)
        self.assertEqual(len(tsdf._columns), 4)
        for i in range(len(tsdf._columns)):
            self.assertEqual(tsdf._data[i].ndarray.shape[0], 10)
        self.assertEqual(tsdf._index_name, 'Index')

    def test_dataframe_attribute_columns(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf.columns, self.col_sol)

    def test_dataframe_attribute_shape(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))
        self.assertEqual(tsdf.shape, (10, 3))

    def test_dataframe_attribute_index(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]

        self.assertEqual(
            list(tsdf.index), list(nd_arr[:, 0].astype(np.uint64))
        )

    def test_dataframe_get_column(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.dlc_data))

        col_data = tsdf['DELTA_VEL[1]']

        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]

        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

    def test_dataframe_sort(self):
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(io.StringIO(self.unsorted_dlc_data))

        # Test out-of-place sort
        reordered_tsdf = tsdf.sort(tsdf.columns, index_column=None,
                                   inplace=False)
        col_data = reordered_tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

        # Test inplace sort_by_index
        tsdf.sort_by_index()
        col_data = tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))

        # Test out-of-place sort with index_column
        tsdf.read_from_text_file(io.StringIO(self.unsorted_dlc_data),
                                 timestamp_in_file=False)

        reordered_tsdf = tsdf.sort(['EPOCH', 'DELTA_VEL[1]'],
                                   index_column='EPOCH', inplace=False)
        col_data = reordered_tsdf['DELTA_VEL[1]']
        nd_arr = np.genfromtxt(io.StringIO(self.dlc_data), delimiter=',')[1:]
        self.assertEqual(list(col_data), list(nd_arr[:, 1]))
