#include <pybind11/pybind11.h>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/buffer/pymod/buffer_pymod.hpp>

#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace modmesh
{

namespace python
{

void import_numpy()
{
    auto local_import_numpy = []()
    {
        import_array2("cannot import numpy", false); // or numpy c api segfault.
        return true;
    };
    if (!local_import_numpy())
    {
        throw pybind11::error_already_set();
    }
}

} // namespace python

} // namespace modmesh

PYBIND11_MODULE(modbuf, mod)
{
    modmesh::python::initialize_buffer(mod);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
