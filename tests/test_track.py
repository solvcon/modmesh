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


import pathlib
import tempfile
import unittest

from modmesh.track import dataset


class NasaDatasetTC(unittest.TestCase):

    def _write_csv(self, directory, name, content):
        path = pathlib.Path(directory) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_load_flight_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            imu_data = (
                "TIME_NANOSECONDS_TAI ,DATA_DELTA_VEL[1] ,"
                "DATA_DELTA_VEL[2] ,DATA_DELTA_VEL[3] ,"
                "DATA_DELTA_ANGLE[1] ,DATA_DELTA_ANGLE[2] ,"
                "DATA_DELTA_ANGLE[3]\n"
                "30,3.0,3.1,3.2,30.0,30.1,30.2\n"
                "10,1.0,1.1,1.2,10.0,10.1,10.2\n"
            )
            imu_csv = self._write_csv(
                tmpdir,
                "dlc.csv",
                imu_data,
            )
            lidar_data = (
                "TIME_NANOSECONDS_TAI ,OMPS_Range_M[1] ,"
                "OMPS_Range_M[2] ,OMPS_Range_M[3] ,OMPS_Range_M[4] ,"
                "OMPS_DopplerSpeed_MpS[1] ,OMPS_DopplerSpeed_MpS[2] ,"
                "OMPS_DopplerSpeed_MpS[3] ,OMPS_DopplerSpeed_MpS[4]\n"
                "40,4.0,4.1,4.2,4.3,40.0,40.1,40.2,40.3\n"
                "20,2.0,2.1,2.2,2.3,20.0,20.1,20.2,20.3\n"
            )
            lidar_csv = self._write_csv(
                tmpdir,
                "commercial_lidar.csv",
                lidar_data,
            )
            gt_data = (
                "TIME_NANOSECONDS_TAI ,truth_pos_CON_ECEF_ECEF_M[1] ,"
                "truth_pos_CON_ECEF_ECEF_M[2] ,"
                "truth_pos_CON_ECEF_ECEF_M[3] ,"
                "truth_vel_CON_ECEF_ECEF_MpS[1] ,"
                "truth_vel_CON_ECEF_ECEF_MpS[2] ,"
                "truth_vel_CON_ECEF_ECEF_MpS[3] ,"
                "truth_quat_CON2ECEF[1] ,truth_quat_CON2ECEF[2] ,"
                "truth_quat_CON2ECEF[3] ,truth_quat_CON2ECEF[4]\n"
                "25,2.5,2.6,2.7,25.0,25.1,25.2,0.25,0.26,0.27,0.28\n"
                "5,0.5,0.6,0.7,5.0,5.1,5.2,0.05,0.06,0.07,0.08\n"
            )
            gt_csv = self._write_csv(
                tmpdir,
                "truth.csv",
                gt_data,
            )

            dst = dataset.NasaDataset(url="", filename="")
            dst.imu_csv = imu_csv
            dst.lidar_csv = lidar_csv
            dst.gt_csv = gt_csv

            dst.load()

            self.assertEqual(len(dst.events), 6)

            expected_events = [
                (
                    5,
                    "ground_truth",
                    {
                        "truth_pos_CON_ECEF_ECEF_M[1]": 0.5,
                        "truth_pos_CON_ECEF_ECEF_M[2]": 0.6,
                        "truth_pos_CON_ECEF_ECEF_M[3]": 0.7,
                        "truth_vel_CON_ECEF_ECEF_MpS[1]": 5.0,
                        "truth_vel_CON_ECEF_ECEF_MpS[2]": 5.1,
                        "truth_vel_CON_ECEF_ECEF_MpS[3]": 5.2,
                        "truth_quat_CON2ECEF[1]": 0.05,
                        "truth_quat_CON2ECEF[2]": 0.06,
                        "truth_quat_CON2ECEF[3]": 0.07,
                        "truth_quat_CON2ECEF[4]": 0.08,
                    },
                ),
                (
                    10,
                    "imu",
                    {
                        "DATA_DELTA_VEL[1]": 1.0,
                        "DATA_DELTA_VEL[2]": 1.1,
                        "DATA_DELTA_VEL[3]": 1.2,
                        "DATA_DELTA_ANGLE[1]": 10.0,
                        "DATA_DELTA_ANGLE[2]": 10.1,
                        "DATA_DELTA_ANGLE[3]": 10.2,
                    },
                ),
                (
                    20,
                    "lidar",
                    {
                        "OMPS_Range_M[1]": 2.0,
                        "OMPS_Range_M[2]": 2.1,
                        "OMPS_Range_M[3]": 2.2,
                        "OMPS_Range_M[4]": 2.3,
                        "OMPS_DopplerSpeed_MpS[1]": 20.0,
                        "OMPS_DopplerSpeed_MpS[2]": 20.1,
                        "OMPS_DopplerSpeed_MpS[3]": 20.2,
                        "OMPS_DopplerSpeed_MpS[4]": 20.3,
                    },
                ),
                (
                    25,
                    "ground_truth",
                    {
                        "truth_pos_CON_ECEF_ECEF_M[1]": 2.5,
                        "truth_pos_CON_ECEF_ECEF_M[2]": 2.6,
                        "truth_pos_CON_ECEF_ECEF_M[3]": 2.7,
                        "truth_vel_CON_ECEF_ECEF_MpS[1]": 25.0,
                        "truth_vel_CON_ECEF_ECEF_MpS[2]": 25.1,
                        "truth_vel_CON_ECEF_ECEF_MpS[3]": 25.2,
                        "truth_quat_CON2ECEF[1]": 0.25,
                        "truth_quat_CON2ECEF[2]": 0.26,
                        "truth_quat_CON2ECEF[3]": 0.27,
                        "truth_quat_CON2ECEF[4]": 0.28,
                    },
                ),
                (
                    30,
                    "imu",
                    {
                        "DATA_DELTA_VEL[1]": 3.0,
                        "DATA_DELTA_VEL[2]": 3.1,
                        "DATA_DELTA_VEL[3]": 3.2,
                        "DATA_DELTA_ANGLE[1]": 30.0,
                        "DATA_DELTA_ANGLE[2]": 30.1,
                        "DATA_DELTA_ANGLE[3]": 30.2,
                    },
                ),
                (
                    40,
                    "lidar",
                    {
                        "OMPS_Range_M[1]": 4.0,
                        "OMPS_Range_M[2]": 4.1,
                        "OMPS_Range_M[3]": 4.2,
                        "OMPS_Range_M[4]": 4.3,
                        "OMPS_DopplerSpeed_MpS[1]": 40.0,
                        "OMPS_DopplerSpeed_MpS[2]": 40.1,
                        "OMPS_DopplerSpeed_MpS[3]": 40.2,
                        "OMPS_DopplerSpeed_MpS[4]": 40.3,
                    },
                ),
            ]

            for event, (timestamp, source, data) in zip(
                dst.events,
                expected_events,
            ):
                self.assertEqual(event.timestamp, timestamp)
                self.assertEqual(event.source, source)
                self.assertIsInstance(event.row, int)
                self.assertEqual(event.data.to_dict(), data)
                for key, value in data.items():
                    self.assertEqual(event.data[key], value)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
