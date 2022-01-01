#pragma once

/*
 * Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
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

// Used in this file.
#include <cstdint>

// Shared by all code.
#include <algorithm>
#include <memory>
#include <iostream>
#include <sstream>
#include <map>

#define MODMESH_EXCEPT(CLS, EXC, MSG) throw EXC(#CLS ": " MSG);

namespace modmesh
{

template <typename I, typename R>
class NumberBase
{

public:

    using int_type = I;
    using uint_type = std::make_unsigned_t<I>;
    using size_type = uint_type;
    using real_type = R;

}; /* end class NumberBase */

/**
 * Spatial table basic information.  Any table-based data store for spatial
 * data should inherit this class template.
 */
template <uint8_t ND, typename I, typename R>
class SpaceBase
  : public NumberBase<I, R>
{

public:

    using dim_type = uint8_t;
    static constexpr const dim_type NDIM = ND;

    using number_base = NumberBase<I, R>;

    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using size_type = typename number_base::size_type;
    using serial_type = typename number_base::size_type;
    using real_type = typename number_base::real_type;

}; /* end class SpaceBase */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
