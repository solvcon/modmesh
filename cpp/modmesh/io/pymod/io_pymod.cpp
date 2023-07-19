#include <modmesh/io/pymod/io_pymod.hpp>

namespace modmesh
{

namespace python
{

struct io_pymod_tag;

template <>
OneTimeInitializer<io_pymod_tag> & OneTimeInitializer<io_pymod_tag>::me()
{
    static OneTimeInitializer<io_pymod_tag> instance;
    return instance;
}

void initialize_io(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_Gmsh(mod);
    };

    OneTimeInitializer<io_pymod_tag>::me()(mod, initialize_impl);
}
} /* end namespace python */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
