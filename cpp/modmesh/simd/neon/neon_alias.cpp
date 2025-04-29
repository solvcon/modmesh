/*
 * Copyright (c) 2025, Kuan-Hsien Lee <khlee870529@gmail.com>
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

#if defined(__aarch64__)

#include <modmesh/simd/neon/neon_alias.hpp>

namespace modmesh
{

namespace simd
{

namespace neon
{

#define utype_t(N) uint##N##_t
#define stype_t(N) int##N##_t

#define DECL_MM_IMPL_VDUPQ(N)                                    \
    template <>                                                  \
    type::vector_t<utype_t(N)> vdupq<utype_t(N)>(utype_t(N) val) \
    {                                                            \
        return vdupq_n_u##N(val);                                \
    }                                                            \
    template <>                                                  \
    type::vector_t<stype_t(N)> vdupq<stype_t(N)>(stype_t(N) val) \
    {                                                            \
        return vdupq_n_s##N(val);                                \
    }

DECL_MM_IMPL_VDUPQ(8)
DECL_MM_IMPL_VDUPQ(16)
DECL_MM_IMPL_VDUPQ(32)
DECL_MM_IMPL_VDUPQ(64)

#undef DECL_MM_IMPL_VDUPQ

#define DECL_MM_IMPL_VLD1Q(N)                                            \
    template <>                                                          \
    type::vector_t<utype_t(N)> vld1q<utype_t(N)>(utype_t(N) * ptr)       \
    {                                                                    \
        return vld1q_u##N(ptr);                                          \
    }                                                                    \
    template <>                                                          \
    type::vector_t<utype_t(N)> vld1q<utype_t(N)>(utype_t(N) const * ptr) \
    {                                                                    \
        return vld1q_u##N(ptr);                                          \
    }                                                                    \
    template <>                                                          \
    type::vector_t<stype_t(N)> vld1q<stype_t(N)>(stype_t(N) * ptr)       \
    {                                                                    \
        return vld1q_s##N(ptr);                                          \
    }                                                                    \
    template <>                                                          \
    type::vector_t<stype_t(N)> vld1q<stype_t(N)>(stype_t(N) const * ptr) \
    {                                                                    \
        return vld1q_s##N(ptr);                                          \
    }

DECL_MM_IMPL_VLD1Q(8)
DECL_MM_IMPL_VLD1Q(16)
DECL_MM_IMPL_VLD1Q(32)
DECL_MM_IMPL_VLD1Q(64)

#undef DECL_MM_IMPL_VLD1Q

#define DECL_MM_IMPL_VST1Q(N)                                                \
    template <>                                                              \
    void vst1q<utype_t(N)>(utype_t(N) * ptr, type::vector_t<utype_t(N)> vec) \
    {                                                                        \
        return vst1q_u##N(ptr, vec);                                         \
    }                                                                        \
    template <>                                                              \
    void vst1q<stype_t(N)>(stype_t(N) * ptr, type::vector_t<stype_t(N)> vec) \
    {                                                                        \
        return vst1q_s##N(ptr, vec);                                         \
    }

DECL_MM_IMPL_VST1Q(8)
DECL_MM_IMPL_VST1Q(16)
DECL_MM_IMPL_VST1Q(32)
DECL_MM_IMPL_VST1Q(64)

#undef DECL_MM_IMPL_VST1Q

#define DECL_MM_IMPL_VCGEQ(N)                                                                                        \
    template <>                                                                                                      \
    type::vector_t<utype_t(N)> vcgeq<utype_t(N)>(type::vector_t<utype_t(N)> vec_a, type::vector_t<utype_t(N)> vec_b) \
    {                                                                                                                \
        return vcgeq_u##N(vec_a, vec_b);                                                                             \
    }                                                                                                                \
    template <>                                                                                                      \
    type::vector_t<stype_t(N)> vcgeq<stype_t(N)>(type::vector_t<stype_t(N)> vec_a, type::vector_t<stype_t(N)> vec_b) \
    {                                                                                                                \
        return vcgeq_s##N(vec_a, vec_b);                                                                             \
    }

DECL_MM_IMPL_VCGEQ(8)
DECL_MM_IMPL_VCGEQ(16)
DECL_MM_IMPL_VCGEQ(32)
DECL_MM_IMPL_VCGEQ(64)

#undef DECL_MM_IMPL_VCGEQ

#define DECL_MM_IMPL_VCLTQ(N)                                                                                        \
    template <>                                                                                                      \
    type::vector_t<utype_t(N)> vcltq<utype_t(N)>(type::vector_t<utype_t(N)> vec_a, type::vector_t<utype_t(N)> vec_b) \
    {                                                                                                                \
        return vcltq_u##N(vec_a, vec_b);                                                                             \
    }                                                                                                                \
    template <>                                                                                                      \
    type::vector_t<stype_t(N)> vcltq<stype_t(N)>(type::vector_t<stype_t(N)> vec_a, type::vector_t<stype_t(N)> vec_b) \
    {                                                                                                                \
        return vcltq_s##N(vec_a, vec_b);                                                                             \
    }

DECL_MM_IMPL_VCLTQ(8)
DECL_MM_IMPL_VCLTQ(16)
DECL_MM_IMPL_VCLTQ(32)
DECL_MM_IMPL_VCLTQ(64)

#undef DECL_MM_IMPL_VCLTQ
#undef stype_t
#undef utype_t

} /* namespace neon */

} /* namespace simd */

} /* namespace modmesh */

#endif /* defined(__aarch64__) */
