#include <modmesh/transform/fourier.hpp>

namespace modmesh
{

namespace transform
{

namespace detail
{

size_t bit_reverse(size_t n, const size_t bits)
{
    size_t reversed = 0;
    for (size_t i = 0; i < bits; i++)
    {
        if (n & (1 << i))
        {
            reversed |= 1 << (bits - 1 - i);
        }
    }
    return reversed;
}

} /* end namespace detail */

} /* end namespace transform */

} // namespace modmesh
