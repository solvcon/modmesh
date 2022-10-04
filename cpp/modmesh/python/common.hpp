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
#include <pybind11/embed.h>

#include <atomic>

#include <modmesh/modmesh.hpp>

#ifdef __GNUG__
#define MODMESH_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#define MODMESH_PYTHON_WRAPPER_VISIBILITY PYBIND11_EXPORT
#endif

namespace modmesh
{

namespace python
{

namespace detail
{

template <class T>
std::string to_str(T const & self) { return Formatter() << self >> Formatter::to_str; }

} /* end namespace detail */

template <typename T>
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

    WrapperProfilerStatus(WrapperProfilerStatus const &) = delete;
    WrapperProfilerStatus(WrapperProfilerStatus &&) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus const &) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus &&) = delete;
    ~WrapperProfilerStatus() = default;

    bool enabled() const { return m_enabled; }
    WrapperProfilerStatus & enable()
    {
        m_enabled = true;
        return *this;
    }
    WrapperProfilerStatus & disable()
    {
        m_enabled = false;
        return *this;
    }

private:

    WrapperProfilerStatus()
        : m_enabled(true)
    {
    }

    std::atomic<bool> m_enabled;

}; /* end class WrapperProfilerStatus */

struct mmtag
{
};

} /* end namespace python */
} /* end namespace modmesh */

namespace pybind11
{
namespace detail
{

template <>
struct process_attribute<modmesh::python::mmtag>
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

// clang-format off
struct
MODMESH_PYTHON_WRAPPER_VISIBILITY
ConcreteBufferNdarrayRemover : ConcreteBuffer::remover_type
// clang-format on
{

    static bool is_same_type(ConcreteBuffer::remover_type const & other)
    {
        return typeid(other) == typeid(ConcreteBufferNdarrayRemover);
    }

    ConcreteBufferNdarrayRemover() = delete;

    explicit ConcreteBufferNdarrayRemover(pybind11::array arr_in)
        : ndarray(std::move(arr_in))
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *) const override {}

    pybind11::array ndarray;

}; /* end struct ConcreteBufferNdarrayRemover */

template <typename S>
std::enable_if_t<is_simple_array_v<S>, pybind11::array> to_ndarray(S && sarr)
{
    namespace py = pybind11;
    using T = typename std::remove_reference_t<S>::value_type;
    std::vector<size_t> const shape(sarr.shape().begin(), sarr.shape().end());
    std::vector<size_t> stride(sarr.stride().begin(), sarr.stride().end());
    for (size_t & v : stride) { v *= sarr.itemsize(); }
    return py::array(
        py::detail::npy_format_descriptor<T>::dtype() /* Numpy dtype */
        ,
        shape /* Buffer dimensions */
        ,
        stride /* Strides (in bytes) for each index */
        ,
        sarr.data() /* Pointer to buffer */
        ,
        py::cast(sarr.buffer().shared_from_this()) /* Owning Python object */
    );
}

template <typename T>
static SimpleArray<T> makeSimpleArray(pybind11::array_t<T> & ndarr)
{
    typename SimpleArray<T>::shape_type shape;
    for (ssize_t i = 0; i < ndarr.ndim(); ++i)
    {
        shape.push_back(ndarr.shape(i));
    }
    std::shared_ptr<ConcreteBuffer> const buffer = ConcreteBuffer::construct(
        ndarr.nbytes(), ndarr.mutable_data(), std::make_unique<ConcreteBufferNdarrayRemover>(ndarr));
    return SimpleArray<T>(shape, buffer);
}

/**
 * Helper template for pybind11 class wrappers.
 */
// clang-format off
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
// clang-format on
{

public:

    using wrapper_type = Wrapper;
    using wrapped_type = Wrapped;
    using wrapped_base_type = WrappedBase;
    using holder_type = Holder;
    // clang-format off
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
    // clang-format on

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
    WrapBase(WrapBase const &) = default;
    WrapBase(WrapBase &&) = delete;
    WrapBase & operator=(WrapBase const &) = default;
    WrapBase & operator=(WrapBase &&) = delete;
    ~WrapBase() = default;

#define DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(METHOD)                           \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD(Args &&... args)                                    \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)...);                            \
        return *static_cast<wrapper_type *>(this);                            \
    }

#define DECL_MM_PYBIND_CLASS_METHOD_TIMED(METHOD)                             \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD##_timed(Args &&... args)                            \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)..., mmtag());                   \
        return *static_cast<wrapper_type *>(this);                            \
    }

#define DECL_MM_PYBIND_CLASS_METHOD(METHOD)     \
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

    template <typename Func>
    wrapper_type & expose_SimpleArray(char const * name, Func && f)
    {
        namespace py = pybind11;

        using array_reference = typename std::invoke_result_t<Func, wrapped_type &>;
        static_assert(std::is_reference<array_reference>::value, "this_array_reference is not a reference");
        static_assert(!std::is_const<array_reference>::value, "this_array_reference cannot be const");
        using array_type = typename std::remove_reference_t<array_reference>;

        (*this)
            .def_property(
                name,
                [&f](wrapped_type & self) -> array_reference
                { return f(self); },
                [&f](wrapped_type & self, py::array_t<typename array_type::value_type> & ndarr)
                {
                    array_reference this_array = f(self);
                    if (this_array.nbytes() != static_cast<size_t>(ndarr.nbytes()))
                    {
                        std::ostringstream msg; // NOLINT(misc-const-correctness) tidy bug
                        msg << ndarr.nbytes() << " bytes of input array differ from "
                            << this_array.nbytes() << " bytes of internal array";
                        throw std::length_error(msg.str());
                    }
                    makeSimpleArray(ndarr).swap(this_array);
                })
            //
            ;

        return *static_cast<wrapper_type *>(this);
    }

    template <typename Func>
    wrapper_type & expose_SimpleArrayAsNdarray(char const * name, Func && f)
    {
        namespace py = pybind11;

        using array_reference = typename std::invoke_result_t<Func, wrapped_type &>;
        static_assert(std::is_reference<array_reference>::value, "this_array_reference is not a reference");
        static_assert(!std::is_const<array_reference>::value, "this_array_reference cannot be const");
        using array_type = typename std::remove_reference_t<array_reference>;

        (*this)
            .def_property(
                name,
                [&f](wrapped_type & self)
                { return to_ndarray(f(self)); },
                [&f](wrapped_type & self, py::array_t<typename array_type::value_type> & ndarr)
                {
                    array_reference this_array = f(self);
                    if (this_array.nbytes() != static_cast<size_t>(ndarr.nbytes()))
                    {
                        std::ostringstream msg; // NOLINT(misc-const-correctness) tidy bug
                        msg << ndarr.nbytes() << " bytes of input array differ from "
                            << this_array.nbytes() << " bytes of internal array";
                        throw std::length_error(msg.str());
                    }
                    this_array.swap(makeSimpleArray(ndarr));
                })
            //
            ;

        return *static_cast<wrapper_type *>(this);
    }

    class_ & cls() { return m_cls; }

protected:

    template <typename... Extra>
    WrapBase(pybind11::module & mod, char const * pyname, char const * pydoc, const Extra &... extra)
        : m_cls(mod, pyname, pydoc, extra...)
    {
    }

private:

    class_ m_cls;

}; /* end class WrapBase */

void import_numpy();
#ifdef __GNUC__
#pragma GCC diagnostic push
// Suppress the warning "greater visibility than the type of its field"
#pragma GCC diagnostic ignored "-Wattributes"
#endif
/**
 * Take a pybind11 module and an initializing function and only run the
 * initializing function once.
 */
template <typename T>
class OneTimeInitializer
{

public:

    OneTimeInitializer(OneTimeInitializer const &) = delete;
    OneTimeInitializer(OneTimeInitializer &&) = delete;
    OneTimeInitializer & operator=(OneTimeInitializer const &) = delete;
    OneTimeInitializer & operator=(OneTimeInitializer &&) = delete;
    ~OneTimeInitializer() = default;

    // Do not implement this function as a template. It should use a specialization in only one compilation unit.
    static OneTimeInitializer<T> & me();

    OneTimeInitializer<T> & operator()(
        pybind11::module & mod, std::function<void(pybind11::module &)> const & initializer)
    {
        if (!initialized())
        {
            m_mod = &mod;
            m_initializer = initializer;
            m_initializer(*m_mod);
        }
        m_initialized = true;
        return *this;
    }

    pybind11::module const & mod() const { return *m_mod; }
    pybind11::module & mod() { return *m_mod; }

    bool initialized() const { return m_initialized && nullptr != m_mod; }

private:

    OneTimeInitializer() = default;

    bool m_initialized = false;
    pybind11::module * m_mod = nullptr;
    std::function<void(pybind11::module &)> m_initializer;

}; /* end class OneTimeInitializer */
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
class MODMESH_PYTHON_WRAPPER_VISIBILITY Interpreter
{

public:

    static Interpreter & instance();

    Interpreter(Interpreter const &) = delete;
    Interpreter(Interpreter &&) = delete;
    Interpreter & operator=(Interpreter const &) = delete;
    Interpreter & operator=(Interpreter &&) = delete;
    ~Interpreter() { finalize(); };

    Interpreter & initialize();
    Interpreter & finalize();
    bool initialized() const { return nullptr != m_interpreter; }

    void preload_module(std::string const & name);
    void preload_modules(std::vector<std::string> const & names);

private:

    Interpreter() = default;
    void setup_path();

    pybind11::scoped_interpreter * m_interpreter = nullptr;

}; /* end class Interpreter */

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
