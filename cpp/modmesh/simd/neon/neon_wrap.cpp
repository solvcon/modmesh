#if defined(__aarch64__)

#include <modmesh/simd/neon/neon_wrap.hpp>

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
