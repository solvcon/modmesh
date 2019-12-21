#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

#include "modmesh/modmesh.hpp"

#ifdef __GNUG__
#  define MODMESH_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#  define MODMESH_PYTHON_WRAPPER_VISIBILITY
#endif

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

    WrapBase() = delete;
    WrapBase(WrapBase const & ) = default;
    WrapBase(WrapBase       &&) = delete;
    WrapBase & operator=(WrapBase const & ) = default;
    WrapBase & operator=(WrapBase       &&) = delete;
    ~WrapBase() = default;

#define DECL_MM_PYBIND_CLASS_METHOD(METHOD) \
    template< class... Args > \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD(Args&&... args) { \
        m_cls.METHOD(std::forward<Args>(args)...); \
        return *static_cast<wrapper_type*>(this); \
    }

    DECL_MM_PYBIND_CLASS_METHOD(def)
    DECL_MM_PYBIND_CLASS_METHOD(def_readwrite)
    DECL_MM_PYBIND_CLASS_METHOD(def_property)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly_static)

#undef DECL_MM_PYBIND_CLASS_METHOD

    class_ & cls() { return m_cls; }

protected:

    WrapBase(pybind11::module & mod)
      : m_cls(mod, wrapper_type::PYNAME, wrapper_type::PYDOC)
    {
        static_assert
        (
            std::is_convertible<decltype(wrapper_type::PYNAME), const char *>::value
          , "wrapper_type::PYNAME is not char *"
        );
        static_assert
        (
            std::is_convertible<decltype(wrapper_type::PYDOC), const char *>::value
          , "wrapper_type::PYDOC is not char *"
        );
    }

private:

    class_ m_cls;

}; /* end class WrapBase */

template< typename Wrapper, typename GT >
class
MODMESH_PYTHON_WRAPPER_VISIBILITY
WrapGridBase
  : public WrapBase< Wrapper, GT >
{

public:

    using base_type = WrapBase< Wrapper, GT >;
    using wrapped_type = typename base_type::wrapped_type;

    friend typename base_type::root_base_type;

protected:

    WrapGridBase(pybind11::module & mod) : base_type(mod)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly_static
            (
                "NDIM"
              , [](py::object const &) { return wrapped_type::NDIM; }
            )
        ;

    }

}; /* end class WrapGridBase */

class WrapGrid1d
  : public WrapGridBase< WrapGrid1d, modmesh::Grid1d >
{

public:

    static constexpr char PYNAME[] = "Grid1d";
    static constexpr char PYDOC[] = "Grid1d";

    friend root_base_type;

    using base_type = WrapGridBase< WrapGrid1d, Grid1d >;

protected:

    WrapGrid1d(pybind11::module & mod) : base_type(mod)
    {}

}; /* end class WrapGrid1d */

class WrapGrid2d
  : public WrapGridBase< WrapGrid2d, modmesh::Grid2d >
{

public:

    static constexpr char PYNAME[] = "Grid2d";
    static constexpr char PYDOC[] = "Grid2d";

    friend root_base_type;

    using base_type = WrapGridBase< WrapGrid2d, Grid2d >;

protected:

    WrapGrid2d(pybind11::module & mod) : base_type(mod)
    {}

}; /* end class WrapGrid2d */

class WrapGrid3d
  : public WrapGridBase< WrapGrid3d, modmesh::Grid3d >
{

public:

    static constexpr char PYNAME[] = "Grid3d";
    static constexpr char PYDOC[] = "Grid3d";

    friend root_base_type;

    using base_type = WrapGridBase< WrapGrid3d, Grid3d >;

protected:

    WrapGrid3d(pybind11::module & mod) : base_type(mod)
    {}

}; /* end class WrapGrid3d */

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
