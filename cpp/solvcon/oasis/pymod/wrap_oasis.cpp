/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/oasis/pymod/oasis_pymod.hpp> // Must be the first include.
#include <solvcon/solvcon.hpp>
#include <pybind11/stl.h>

#include <solvcon/oasis/oasis_device.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapOasisDevice
    : public WrapBase<WrapOasisDevice, OasisDevice, std::shared_ptr<OasisDevice>>
{

public:

    using base_type = WrapBase<WrapOasisDevice, OasisDevice, std::shared_ptr<OasisDevice>>;
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
            .def("to_bytes", &wrapped_type::to_bytes)
            .def("add_rect_record", &wrapped_type::add_rect_record, py::arg("rect_record"))
            .def("add_poly_record", &wrapped_type::add_poly_record, py::arg("polygon_record"));
    }
}; /* end class WrapOasisDevice */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapOasisRecordPoly
    : public WrapBase<WrapOasisRecordPoly, OasisRecordPoly, std::shared_ptr<OasisRecordPoly>>
{

public:

    using base_type = WrapBase<WrapOasisRecordPoly, OasisRecordPoly, std::shared_ptr<OasisRecordPoly>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapOasisRecordPoly(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<std::vector<std::pair<int, int>>>(), py::arg("coords"))
            .def("to_bytes", &wrapped_type::to_bytes);
    }
}; /* end class WrapOasisRecordPoly */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapOasisRecordRect
    : public WrapBase<WrapOasisRecordRect, OasisRecordRect, std::shared_ptr<OasisRecordRect>>
{

public:

    using base_type = WrapBase<WrapOasisRecordRect, OasisRecordRect, std::shared_ptr<OasisRecordRect>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapOasisRecordRect(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        (*this)
            .def(py::init<int, int, int, int>(), py::arg("lower"), py::arg("left"), py::arg("w"), py::arg("h"))
            .def("to_bytes", &wrapped_type::to_bytes);
    }
}; /* end class WrapOasisRecordRect */

void wrap_oasis_device(pybind11::module & mod)
{
    WrapOasisDevice::commit(mod, "OasisDevice", "OASIS bytes device");
    WrapOasisRecordPoly::commit(mod, "OasisRecordPoly", "OASIS polygon record");
    WrapOasisRecordRect::commit(mod, "OasisRecordRect", "OASIS rectangle record");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
