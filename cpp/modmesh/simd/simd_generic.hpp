#pragma once

namespace modmesh
{

namespace simd
{

namespace generic
{

template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    T const * ptr = start;
    while (ptr < end)
    {
        T idx = *ptr;
        if (idx < min_val || idx > max_val)
        {
            return ptr;
        }
        ++ptr;
    }
    return nullptr;
}

} /* namespace generic */

} /* namespace simd */

} /* namespace modmesh */
