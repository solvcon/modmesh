#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <atomic>

#include <modmesh/modmesh.hpp>

#ifdef __GNUG__
#  define MODMESH_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#  define MODMESH_PYTHON_WRAPPER_VISIBILITY
#endif

namespace modmesh
{
namespace python
{

template < typename T >
bool dtype_is_type(pybind11::array const & arr)
{
    return pybind11::detail::npy_format_descriptor<T>::dtype().is(arr.dtype());
}

class WrapperProfilerStatus
{

public:

    static WrapperProfilerStatus & me()
    {
        static WrapperProfilerStatus instance;
        return instance;
    }

    WrapperProfilerStatus(WrapperProfilerStatus const & ) = delete;
    WrapperProfilerStatus(WrapperProfilerStatus       &&) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus const & ) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus       &&) = delete;
    ~WrapperProfilerStatus() = default;

    bool enabled() const { return m_enabled; }
    WrapperProfilerStatus & enable()  { m_enabled = true ; return *this; }
    WrapperProfilerStatus & disable() { m_enabled = false; return *this; }

private:

    WrapperProfilerStatus()
      : m_enabled(true)
    {}

    std::atomic<bool> m_enabled;

}; /* end class WrapperProfilerStatus */

struct mmtag {};

} /* end namespace python */
} /* end namespace modmesh */

namespace pybind11
{
namespace detail
{

template<> struct process_attribute<modmesh::python::mmtag>
  : process_attribute_default<modmesh::python::mmtag>
{

    static void precall(function_call & call)
    {
        if (modmesh::python::WrapperProfilerStatus::me().enabled())
        {
            modmesh::TimeRegistry::me().entry(get_name(call)).start();
        }
    }

    static void postcall(function_call & call, handle &)
    {
        if (modmesh::python::WrapperProfilerStatus::me().enabled())
        {
            modmesh::TimeRegistry::me().entry(get_name(call)).stop();
        }
    }

private:

    static std::string get_name(function_call const & call)
    {
        function_record const & r = call.func;
        return std::string(str(r.scope.attr("__name__"))) + std::string(".") + r.name;
    }

};

} /* end namespace detail */
} /* end namespace pybind11 */

namespace modmesh
{

namespace python
{

/**
 * Helper template for pybind11 class wrappers.
 */
template
<
    class Wrapper
  , class Wrapped
  , class Holder = std::unique_ptr<Wrapped>
  , class WrappedBase = Wrapped
>
/*
 * Use CRTP to detect type error during compile time.
 */
class
MODMESH_PYTHON_WRAPPER_VISIBILITY
WrapBase
{

public:

    using wrapper_type = Wrapper;
    using wrapped_type = Wrapped;
    using wrapped_base_type = WrappedBase;
    using holder_type = Holder;
    using root_base_type = WrapBase
    <
        wrapper_type
      , wrapped_type
      , holder_type
      , wrapped_base_type
    >;
    using class_ = typename std::conditional_t
    <
        std::is_same< Wrapped, WrappedBase >::value
      , pybind11::class_< wrapped_type, holder_type >
      , pybind11::class_< wrapped_type, wrapped_base_type, holder_type >
    >;

    static wrapper_type & commit(pybind11::module & mod)
    {
        static wrapper_type derived(mod);
        return derived;
    }

    static wrapper_type & commit(pybind11::module & mod, char const * pyname, char const * pydoc)
    {
        static wrapper_type derived(mod, pyname, pydoc);
        return derived;
    }

    WrapBase() = delete;
    WrapBase(WrapBase const & ) = default;
    WrapBase(WrapBase       &&) = delete;
    WrapBase & operator=(WrapBase const & ) = default;
    WrapBase & operator=(WrapBase       &&) = delete;
    ~WrapBase() = default;

#define DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(METHOD) \
    template< class... Args > \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD(Args&&... args) \
    { \
        m_cls.METHOD(std::forward<Args>(args)...); \
        return *static_cast<wrapper_type*>(this); \
    }

#define DECL_MM_PYBIND_CLASS_METHOD_TIMED(METHOD) \
    template< class... Args > \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD ## _timed(Args&&... args) \
    { \
        m_cls.METHOD(std::forward<Args>(args)..., mmtag()); \
        return *static_cast<wrapper_type*>(this); \
    }

#define DECL_MM_PYBIND_CLASS_METHOD(METHOD) \
    DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(METHOD) \
    DECL_MM_PYBIND_CLASS_METHOD_TIMED(METHOD)

    DECL_MM_PYBIND_CLASS_METHOD(def)
    DECL_MM_PYBIND_CLASS_METHOD(def_static)

    DECL_MM_PYBIND_CLASS_METHOD(def_readwrite)
    DECL_MM_PYBIND_CLASS_METHOD(def_readonly)
    DECL_MM_PYBIND_CLASS_METHOD(def_readwrite_static)
    DECL_MM_PYBIND_CLASS_METHOD(def_readonly_static)

    DECL_MM_PYBIND_CLASS_METHOD(def_property)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_static)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly_static)

    DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(def_buffer)

#undef DECL_MM_PYBIND_CLASS_METHOD_UNTIMED
#undef DECL_MM_PYBIND_CLASS_METHOD_TIMED
#undef DECL_MM_PYBIND_CLASS_METHOD

    class_ & cls() { return m_cls; }

protected:

    template <typename... Extra>
    WrapBase(pybind11::module & mod, char const * pyname, char const * pydoc, const Extra & ... extra)
      : m_cls(mod, pyname, pydoc, extra ...)
    {}

private:

    class_ m_cls;

}; /* end class WrapBase */

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
