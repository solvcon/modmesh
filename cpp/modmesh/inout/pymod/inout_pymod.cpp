#include <modmesh/inout/pymod/inout_pymod.hpp>

namespace modmesh
{

namespace python
{

struct inout_pymod_tag;

template <>
OneTimeInitializer<inout_pymod_tag> & OneTimeInitializer<inout_pymod_tag>::me()
{
    static OneTimeInitializer<inout_pymod_tag> instance;
    return instance;
}

void initialize_inout(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_Gmsh(mod);
    };

    OneTimeInitializer<inout_pymod_tag>::me()(mod, initialize_impl);
}
} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
