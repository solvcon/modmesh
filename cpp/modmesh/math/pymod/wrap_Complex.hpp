#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/numpy.h>

namespace pybind11
{

namespace detail
{

template <>
struct npy_format_descriptor<modmesh::Complex<double>>
{
    static constexpr auto name = const_name("complex128");
    static constexpr int value = npy_api::NPY_CDOUBLE_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex128");
    }

    static std::string format()
    {
        return "=Zd";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(typename std::remove_cv<modmesh::Complex<double>>::type),
                                  sizeof(modmesh::Complex<double>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        static PyObject * ptr = get_numpy_internals().get_type_info<modmesh::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& value)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                value = ((PyVoidScalarObject_Proxy *)obj)->obval;
                return true;
            }
        }
        return false;
    }
};

template <>
struct npy_format_descriptor<modmesh::Complex<float>>
{
    static constexpr auto name = const_name("complex64");
    static constexpr int value = npy_api::NPY_CFLOAT_;

    static pybind11::dtype dtype()
    {
        return pybind11::dtype("complex64");
    }

    static std::string format()
    {
        return "=Zf";
    }

    static void register_dtype(any_container<field_descriptor> fields)
    {
        register_structured_dtype(std::move(fields),
                                  typeid(typename std::remove_cv<modmesh::Complex<float>>::type),
                                  sizeof(modmesh::Complex<float>),
                                  &direct_converter);
    }

private:
    static PyObject * dtype_ptr()
    {
        static PyObject * ptr = get_numpy_internals().get_type_info<modmesh::Complex<double>>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject * obj, void *& value)
    {
        auto & api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
        {
            return false;
        }
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj)))
        {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr()))
            {
                value = ((PyVoidScalarObject_Proxy *)obj)->obval;
                return true;
            }
        }
        return false;
    }
};

} /* end namespace detail */

} /* end namespace pybind11 */
