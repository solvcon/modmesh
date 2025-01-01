#include <modmesh/mathtype/pymod/mathtype_pymod.hpp>

namespace modmesh
{

namespace python
{

struct mathtype_pymod_tag;

template <>
OneTimeInitializer<mathtype_pymod_tag> & OneTimeInitializer<mathtype_pymod_tag>::me()
{
    static OneTimeInitializer<mathtype_pymod_tag> instance;
    return instance;
}

void initialize_mathtype(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_Complex(mod);
    };

    OneTimeInitializer<mathtype_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
