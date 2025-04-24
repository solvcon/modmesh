#pragma once

#include <modmesh/simd/simd_generic.hpp>

#if defined(__aarch64__)
#include <modmesh/simd/neon/neon.hpp>
#endif /* defined(__aarch64__) */

namespace modmesh
{

namespace simd
{

// Check if each element from start to end (excluded end) is within the range [min_val, max_val)
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
#if defined(__aarch64__)
    return neon::check_between<T>(start, end, min_val, max_val);
#else
    return generic::check_between<T>(start, end, min_val, max_val);
#endif /* defined(__aarch64__) */
}

} /* namespace simd */

} /* namespace modmesh */
