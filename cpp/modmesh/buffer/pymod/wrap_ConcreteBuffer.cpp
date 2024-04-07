/*
 * Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
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

#include <modmesh/buffer/pymod/buffer_pymod.hpp> // Must be the first include.
#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{

namespace python
{

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapConcreteBuffer
    : public WrapBase<WrapConcreteBuffer, ConcreteBuffer, std::shared_ptr<ConcreteBuffer>>
{

    friend root_base_type;

    WrapConcreteBuffer(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapConcreteBuffer */

WrapConcreteBuffer::WrapConcreteBuffer(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](size_t nbytes)
                { return wrapped_type::construct(nbytes); }),
            py::arg("nbytes"))
        .def(
            py::init(
                [](py::array & arr_in)
                {
                    return wrapped_type::construct(
                        arr_in.nbytes(), arr_in.mutable_data(), std::make_unique<ConcreteBufferNdarrayRemover>(arr_in));
                }),
            py::arg("array"))
        .def_timed("clone", &wrapped_type::clone)
        .def_property_readonly("nbytes", &wrapped_type::nbytes)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, int8_t val)
            { self.at(it) = val; })
        .def_buffer(
            [](wrapped_type & self)
            {
                return py::buffer_info(
                    self.data(), /* Pointer to buffer */
                    sizeof(int8_t), /* Size of one scalar */
                    py::format_descriptor<int8_t>::format(), /* Python struct-style format descriptor */
                    1, /* Number of dimensions */
                    {self.size()}, /* Buffer dimensions */
                    {1} /* Strides (in bytes) for each index */
                );
            })
        .def_property_readonly(
            "ndarray",
            [](wrapped_type & self)
            {
                namespace py = pybind11;
                return py::array(
                    py::detail::npy_format_descriptor<int8_t>::dtype(), /* Numpy dtype */
                    {self.size()}, /* Buffer dimensions */
                    {1}, /* Strides (in bytes) for each index */
                    self.data(), /* Pointer to buffer */
                    py::cast(self.shared_from_this()) /* Owning Python object */
                );
            })
        .def_property_readonly(
            "is_from_python",
            [](wrapped_type const & self)
            {
                return self.has_remover() && ConcreteBufferNdarrayRemover::is_same_type(self.get_remover());
            })
        //
        ;
}

class MODMESH_PYTHON_WRAPPER_VISIBILITY WrapBufferExpander
    : public WrapBase<WrapBufferExpander, BufferExpander, std::shared_ptr<BufferExpander>>
{

    friend root_base_type;

    WrapBufferExpander(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapBufferExpander */

WrapBufferExpander::WrapBufferExpander(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](size_t length)
                { return wrapped_type::construct(length); }),
            py::arg("length"))
        .def_timed(py::init([]()
                            { return wrapped_type::construct(); }))
        .def("reserve", &wrapped_type::reserve, py::arg("cap"))
        .def("expand", &wrapped_type::expand, py::arg("length"))
        .def_property_readonly("capacity", &wrapped_type::capacity)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, int8_t val)
            { self.at(it) = val; })
        .def("clone", &wrapped_type::clone)
        .def("copy_concrete", &wrapped_type::copy_concrete, py::arg("cap") = 0)
        .def("as_concrete", &wrapped_type::as_concrete, py::arg("cap") = 0)
        .def_property_readonly("is_concrete", &wrapped_type::is_concrete)
        //
        ;
}

void wrap_ConcreteBuffer(pybind11::module & mod)
{
    WrapConcreteBuffer::commit(mod, "ConcreteBuffer", "ConcreteBuffer");
    WrapBufferExpander::commit(mod, "BufferExpander", "BufferExpander");
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
