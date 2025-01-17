#pragma once

#include <modmesh/math/math.hpp>
#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{

namespace transform
{

namespace detail
{

size_t bit_reverse(size_t n, const size_t bits);

} /* end namespace detail */

// TODO: The template of template is too complicate, we should find a way to make it easier.
template <template <typename> typename T1, typename T2>
void dft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
{
    size_t N = in.size();
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            T2 tmp = -2.0 * pi<T2> * i * j / N;
            T1<T2> twiddle_factor{.real_v = std::cos(tmp), .imag_v = std::sin(tmp)};
            out[i] += in[j] * twiddle_factor;
        }

        // Normalize dft output
        out[i] = out[i] / std::sqrt(static_cast<T2>(N));
    }
}

template <template <typename> class T1, typename T2>
void fft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
{
    auto N = in.size();
    const unsigned int bits = static_cast<unsigned int>(std::log2(N));

    // bit reversed reordering
    for (size_t i = 0; i < N; ++i)
    {
        out[detail::bit_reverse(i, bits)] = in[i];
    }

    // Cooly-Tukey FFT algorithm, radix-2
    for (size_t size = 2; size <= N; size *= 2)
    {
        size_t half_size = size / 2;
        T2 angle_inc = -2.0 * pi<T2> / size;

        for (size_t i = 0; i < N; i += size)
        {
            for (size_t k = 0; k < half_size; ++k)
            {
                // Twiddle factor = exp(-2 * pi * i * k / N)
                T2 angle = angle_inc * k;
                T1<T2> twiddle_factor{.real_v = std::cos(angle), .imag_v = std::sin(angle)};

                T1<T2> even(out[i + k]);
                T1<T2> odd(out[i + k + half_size] * twiddle_factor);

                out[i + k] = even + odd;
                out[i + k + half_size] = even - odd;
            }
        }
    }

    // Normalize fft output
    for (size_t i = 0; i < N; ++i)
    {
        out[i] = out[i] / std::sqrt(static_cast<T2>(N));
    }
}

} /* end namespace transform */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
