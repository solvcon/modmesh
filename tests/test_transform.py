# Copyright (c) 2025, Chun-Hsu Lai <as2266317@gmail.com>
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

import numpy as np

import modmesh as mm


class TransformTB(mm.testing.TestBase):

    def setUp(self):
        pass

    def mm_complex(self, real=0.0, imag=0.0):
        pass

    def real_rng(self):
        pass

    def imag_rng(self):
        pass

    def mm_simplearraycomplex(self, size):
        pass

    def test_numpy_dft_comparison(self):
        input_size = 100

        mm_input = self.mm_simplearraycomplex(input_size)
        for i in range(input_size):
            mm_input[i] = self.mm_complex(self.real_rng(), self.imag_rng())

        np_input = np.array(mm_input, copy=False)

        mm_output = self.mm_simplearraycomplex(input_size)
        mm.transform.dft(mm_input, mm_output)

        np_output = np.fft.fft(np_input)

        for i in range(input_size):
            self.assert_allclose(mm_output[i].real, np_output[i].real)
            self.assert_allclose(mm_output[i].imag, np_output[i].imag)

    def test_numpy_fft_comparison(self):
        input_size = 100

        mm_input = self.mm_simplearraycomplex(input_size)
        for i in range(input_size):
            mm_input[i] = self.mm_complex(self.real_rng(), self.imag_rng())

        np_input = np.array(mm_input, copy=False)

        mm_output = self.mm_simplearraycomplex(input_size)
        mm.transform.fft(mm_input, mm_output)

        np_output = np.fft.fft(np_input)

        for i in range(input_size):
            self.assert_allclose(mm_output[i].real, np_output[i].real)
            self.assert_allclose(mm_output[i].imag, np_output[i].imag)

    def test_numpy_ifft_comparison(self):
        input_size = 100

        mm_input = self.mm_simplearraycomplex(input_size)
        for i in range(input_size):
            mm_input[i] = self.mm_complex(self.real_rng(), self.imag_rng())

        np_input = np.array(mm_input, copy=False)

        mm_output = self.mm_simplearraycomplex(input_size)
        mm.transform.ifft(mm_input, mm_output)

        np_output = np.fft.ifft(np_input)

        for i in range(input_size):
            self.assert_allclose(mm_output[i].real, np_output[i].real)
            self.assert_allclose(mm_output[i].imag, np_output[i].imag)


class TransformFp32TC(TransformTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'atol' not in kw:
            kw['atol'] = 1.e-2
        return super().assert_allclose(*args, **kw)

    def mm_complex(self, real=0.0, imag=0.0):
        return mm.complex64(real, imag)

    def mm_simplearraycomplex(self, size):
        return mm.SimpleArrayComplex64(size, mm.complex64())

    def real_rng(self):
        return np.float32(np.random.uniform(-1.0, 1.0))

    def imag_rng(self):
        return np.float32(np.random.uniform(-1.0, 1.0))

    def setUp(self):
        np.random.seed()


class TransformFp64TC(TransformTB, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'atol' not in kw:
            kw['atol'] = 1.e-10
        return super().assert_allclose(*args, **kw)

    def mm_complex(self, real=0.0, imag=0.0):
        return mm.complex128(real, imag)

    def mm_simplearraycomplex(self, size):
        return mm.SimpleArrayComplex128(size, mm.complex128())

    def real_rng(self):
        return np.random.uniform(-1.0, 1.0)

    def imag_rng(self):
        return np.random.uniform(-1.0, 1.0)

    def setUp(self):
        np.random.seed()
