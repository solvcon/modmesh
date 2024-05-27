#pragma once

/*
 * Copyright (c) 2024, Chunhsu Lai <as2266317@gmail.com>
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

#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <modmesh/base.hpp>
#include <modmesh/mesh/mesh.hpp>
#include <modmesh/buffer/buffer.hpp>
#include <modmesh/inout/inout_util.hpp>

namespace modmesh
{

namespace inout
{

class Plot3d
    : public NumberBase<int32_t, double>
{
    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

public:
    explicit Plot3d(const std::string & data);

    ~Plot3d() = default;

    Plot3d() = delete;
    Plot3d(Plot3d const & other) = delete;
    Plot3d(Plot3d && other) = delete;
    Plot3d & operator=(Plot3d const & other) = delete;
    Plot3d & operator=(Plot3d && other) = delete;

    std::shared_ptr<StaticMesh> to_block();
    void build_interior(const std::shared_ptr<StaticMesh> & blk);

private:
    inline uint_type coordinate_to_node_id(uint_type blk, uint_type x, uint_type y, uint_type z)
    {
        return blk > 0 ? m_blk_sizes(blk - 1) : 0 + x + y * m_x_shape(blk) + z * m_x_shape(blk) * m_y_shape(blk);
    }

    void parseCoordinates(const uint_type nblocks);
    void buildHexahedronElements(const uint_type nblocks);

    std::stringstream stream;

    SimpleArray<real_type> m_nds;
    SimpleArray<uint_type> m_x_shape;
    SimpleArray<uint_type> m_y_shape;
    SimpleArray<uint_type> m_z_shape;
    SimpleArray<uint_type> m_blk_sizes;

    std::unordered_map<uint_type, small_vector<uint_type>> m_elems;
};

} // namespace inout

} // namespace modmesh

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
