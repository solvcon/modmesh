# Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
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


import unittest
import modmesh


class OasisRecordRectTC(unittest.TestCase):
    # Please refer the comment of modmesh::OasisRecordRect in oasis_device.cpp
    def test_to_byte(self):
        rec = modmesh.OasisRecordRect(70, 800, 180, 40)

        expected_record_bytes = '\x14\x7B\x00\x00\xB4\x01\x28\x8C\x01\xC0\x0C'
        record_bytes = rec.to_bytes()

        self.assertEqual(record_bytes, list(map(ord, expected_record_bytes)))


class OasisRecordPolyTC(unittest.TestCase):
    # Please refer the comment of modmesh::OasisRecordPoly in oasis_device.cpp
    def test_type_1_to_byte(self):
        rec = modmesh.OasisRecordPoly([
            [410, 720], [410, 920], [70, 920],
            [70, 880], [370, 880], [370, 760], [70, 760], [70, 720]])

        expected_record_bytes = '\x15\x3B\x00\x00\x01\x07' \
                                '\x90\x03\xA9\x05\x51\xD8' \
                                '\x04\xF1\x01\xD9\x04\x51' \
                                '\xB4\x06\xA0\x0B'
        record_bytes = rec.to_bytes()

        self.assertEqual(record_bytes, list(map(ord, expected_record_bytes)))

    def test_type_0_to_byte(self):
        rec = modmesh.OasisRecordPoly([
            [70, 720], [410, 720], [410, 920], [70, 920],
            [70, 880], [370, 880], [370, 760], [70, 760]])

        expected_record_bytes = '\x15\x3B\x00\x00\x00\x07' \
                                '\xA8\x05\x90\x03\xA9\x05' \
                                '\x51\xD8\x04\xF1\x01\xD9' \
                                '\x04\x8C\x01\xA0\x0B'
        record_bytes = rec.to_bytes()

        self.assertEqual(record_bytes, list(map(ord, expected_record_bytes)))


# For OASIS format, refer modmesh::OasisDevice comment in oasis_device.cpp
class OasisDeviceTC(unittest.TestCase):
    def oasis_bytes(self, records=None):
        magic_bytes = '%SEMI-OASIS\x0D\x0A'
        start = "\x01\x031.0\x00\xE8\x07" + '\x00' * 13
        cell = "\x03\x03\x54\x4F\x50\x0D\x00"
        end = '\x02' + '\x00' * 254 + '\x00'

        if records is None:
            return list(map(ord, magic_bytes + start + cell + end))
        else:
            return list(map(ord, magic_bytes + start + cell + records + end))

    def test_empty_oasis(self):
        device = modmesh.OasisDevice()
        oasis_bytes = device.to_bytes()

        self.assertEqual(oasis_bytes, self.oasis_bytes())

    def test_rect_oasis(self):
        device = modmesh.OasisDevice()
        rec = modmesh.OasisRecordRect(70, 800, 180, 40)

        device.add_rect_record(rec)

        oasis_bytes = device.to_bytes()
        rec_record_bytes = '\x14\x7B\x00\x00\xB4\x01\x28\x8C\x01\xC0\x0C'
        self.assertEqual(oasis_bytes, self.oasis_bytes(rec_record_bytes))

    def test_poly_oasis(self):
        device = modmesh.OasisDevice()
        rec = modmesh.OasisRecordPoly([
            [70, 720], [410, 720], [410, 920], [70, 920],
            [70, 880], [370, 880], [370, 760], [70, 760]])

        device.add_poly_record(rec)

        oasis_bytes = device.to_bytes()
        poly_record_bytes = '\x15\x3B\x00\x00\x00\x07' \
                            '\xA8\x05\x90\x03\xA9\x05' \
                            '\x51\xD8\x04\xF1\x01\xD9' \
                            '\x04\x8C\x01\xA0\x0B'
        self.assertEqual(oasis_bytes, self.oasis_bytes(poly_record_bytes))

    def test_mix_geometry_record_oasis(self):
        device = modmesh.OasisDevice()
        rec_poly = modmesh.OasisRecordPoly([
            [70, 720], [410, 720], [410, 920], [70, 920],
            [70, 880], [370, 880], [370, 760], [70, 760]])
        rec_rect = modmesh.OasisRecordRect(70, 800, 180, 40)

        device.add_poly_record(rec_poly)
        device.add_rect_record(rec_rect)

        oasis_bytes = device.to_bytes()
        rec_record_bytes = '\x14\x7B\x00\x00\xB4\x01\x28\x8C\x01\xC0\x0C'
        poly_record_bytes = '\x15\x3B\x00\x00\x00\x07' \
                            '\xA8\x05\x90\x03\xA9\x05' \
                            '\x51\xD8\x04\xF1\x01\xD9' \
                            '\x04\x8C\x01\xA0\x0B'
        mix_record_bytes = rec_record_bytes + poly_record_bytes
        self.assertEqual(oasis_bytes, self.oasis_bytes(mix_record_bytes))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
