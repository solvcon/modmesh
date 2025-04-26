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

#include <modmesh/simd/simd_support.hpp>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#if defined(__aarch64__) || defined(__arm__)
#include <asm/hwcap.h>
#endif
#elif defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace modmesh
{

namespace simd
{

namespace detail
{

SimdFeature detect_simd()
{
    static SimdFeature CurrentFeature = SIMD_UNKNOWN;

    if (CurrentFeature != SIMD_UNKNOWN)
    {
        return CurrentFeature;
    }

#if defined(__aarch64__) || defined(__arm__)
// ARM architecture
#if defined(__linux__) || defined(__ANDROID__)
    unsigned long hwcaps = getauxval(AT_HWCAP);
#ifdef HWCAP_NEON
    if (hwcaps & HWCAP_NEON)
    {
        CurrentFeature = SIMD_NEON;
    }
#endif /* HWCAP_NEON */
#elif defined(__APPLE__)
    int neon_supported = 0;
    size_t size = sizeof(neon_supported);
    if (sysctlbyname("hw.optional.neon", &neon_supported, &size, NULL, 0) == 0 && neon_supported)
    {
        CurrentFeature = SIMD_NEON;
    }
#elif defined(_WIN32)
    if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE))
    {
        CurrentFeature = SIMD_NEON;
    }
#endif
#endif /* defined(__aarch64__) || defined(__arm__) */

    CurrentFeature = SIMD_NONE;

    return CurrentFeature;
}

} /* namespace detail */

} /* namespace simd */

} /* namespace modmesh */
