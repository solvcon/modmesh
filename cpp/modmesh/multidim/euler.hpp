#pragma once

/*
 * Copyright (c) 2024, Yung-Yu Chen <yyc@solvcon.net>
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

/**
 * The space-time CESE solver for the Euler equation.
 */

#include <modmesh/mesh/mesh.hpp>

namespace modmesh
{

class EulerCore
    : public NumberBase<int32_t, double>
    , public std::enable_shared_from_this<EulerCore>
{

private:

    class ctor_passkey
    {
    };

public:

    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

    template <typename... Args>
    static std::shared_ptr<EulerCore> construct(Args &&... args)
    {
        return std::make_shared<EulerCore>(std::forward<Args>(args)..., ctor_passkey());
    }

    EulerCore(std::shared_ptr<StaticMesh> const & mesh, real_type time_increment, ctor_passkey const &)
        : m_mesh(mesh)
        , m_time_increment(time_increment)
    {
    }

    EulerCore() = delete;
    EulerCore(EulerCore const &) = delete;
    EulerCore(EulerCore &&) = delete;
    EulerCore operator=(EulerCore const &) = delete;
    EulerCore operator=(EulerCore &&) = delete;
    ~EulerCore() = default;

private:

    std::shared_ptr<StaticMesh> m_mesh;
    real_type m_time_increment = 0.0;

}; /* end class EulerCore */

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
