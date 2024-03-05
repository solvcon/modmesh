/*
 * Copyright (c) 2024, An-Chi Liu <phy.tiger@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <modmesh/testhelper/pymod/testbuffer_pymod.hpp> // Must be the first include.

#include <modmesh/buffer/buffer.hpp>
#include <modmesh/buffer/pymod/SimpleArrayCaster.hpp>

namespace modmesh
{

namespace python
{

struct TestSimpleArrayHelper
{
}; /* end of TestSimpleArrayHelper */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapTestSimpleArrayHelper
    : public WrapBase<WrapTestSimpleArrayHelper, TestSimpleArrayHelper>
{

public:

    friend root_base_type;

protected:

    WrapTestSimpleArrayHelper(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        (*this)
            .def("test_cast_to_arrayint32", []() -> modmesh::SimpleArrayInt32
                 {
                     // clang-format off
                     SimpleArrayInt32 arr(100);
                     return arr;
                     // clang-format on
                 })
            .def("test_load_arrayin32_from_arrayplex", [](modmesh::SimpleArrayInt32 &) -> bool
                 { return true; })
            .def("test_load_arrayfloat64_from_arrayplex", [](modmesh::SimpleArrayFloat64 &) -> bool
                 { return true; })
            //
            ;
    }

}; /* end class WrapTestSimpleArrayHelper */

void wrap_TestSimpleArrayHelper(pybind11::module & mod)
{
    WrapTestSimpleArrayHelper::commit(mod, "TestSimpleArrayHelper", "TestSimpleArrayHelper");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: