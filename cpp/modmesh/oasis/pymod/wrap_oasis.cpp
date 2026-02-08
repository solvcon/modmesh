/*
 * Copyright (c) 2026, Han-Xuan Huang <c1ydehhx@gmail.com>
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

#include <modmesh/oasis/pymod/oasis_pymod.hpp> // Must be the first include.
#include <modmesh/modmesh.hpp>

#include <modmesh/oasis/oasis_device.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapOasisDevice
    : public WrapBase<WrapOasisDevice, oasis::OasisDevice, std::shared_ptr<oasis::OasisDevice>>
{

public:

    using base_type = WrapBase<WrapOasisDevice, oasis::OasisDevice, std::shared_ptr<oasis::OasisDevice>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapOasisDevice(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init())
            .def("write", &wrapped_type::write)
            .def("add_rect_record", &wrapped_type::add_rect_record, py::arg("rect_record"))
            .def("add_poly_record", &wrapped_type::add_poly_record, py::arg("polygon_record"));
    }
}; /* end class WrapOASISDevice */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapPolyRecord
    : public WrapBase<WrapPolyRecord, oasis::PolyRecord, std::shared_ptr<oasis::PolyRecord>>
{

public:

    using base_type = WrapBase<WrapPolyRecord, oasis::PolyRecord, std::shared_ptr<oasis::PolyRecord>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapPolyRecord(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<std::vector<std::pair<int, int>>>(), py::arg("coords"))
            .def("to_bytes", &wrapped_type::to_bytes);
    }
}; /* end class WrapPolyRecord */

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapRectRecord
    : public WrapBase<WrapRectRecord, oasis::RectRecord, std::shared_ptr<oasis::RectRecord>>
{

public:

    using base_type = WrapBase<WrapRectRecord, oasis::RectRecord, std::shared_ptr<oasis::RectRecord>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapRectRecord(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<std::pair<int, int>, int, int>(), py::arg("left-bottom"), py::arg("w"), py::arg("h"))
            .def("to_bytes", &wrapped_type::to_bytes);
    }
}; /* end class WrapPolygonRecord */

void wrap_oasis_device(pybind11::module & mod)
{
    WrapOasisDevice::commit(mod, "OasisDevice", "OASIS bytes device");
    WrapPolyRecord::commit(mod, "OasisPolyRecord", "OASIS polygon record");
    WrapRectRecord::commit(mod, "OasisRectRecord", "OASIS rectangle record");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
